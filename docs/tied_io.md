# Tied input/output embeddings — design record

Status: **design agreed; the plain class is implemented, tying is
not.** Implementation naming settled after this record was written:
the component is called the **codec** — `OneHotCodec`
(`layers/one_hot_codec.py`, the plain class below; the soft-cap is on
by default after DB validation) and `EmbeddingCodec` (both
tied rows of the table below, upcoming), sharing `layers/codec.py`.
This is the decision log and migration plan from the 2026-07 design
discussions; the user-facing description of how texmo IO works
(including the parts designed here) lives in [`io.md`](io.md). The
immediate driver is RecurrentGemma (its LM head is the embedding table
transposed); the larger prize is the search: input + output layers
dominate the weight budget of small models (`bytes|dense.1.tanh` spends
768 of its 769 weights on them), and per search results `bits.4` beats
`bytes` up to >=20k weights largely because of that overhead.

## The InputOutput class

Input and output become one paired component owning a single embedding
table, the output head, the input scale, and the logit soft-cap. The
head stays **implicit in the spec** (no trailing `dense.N`): an
explicit head would allow degenerate forms and force a rename of every
conf in the DB for zero expressiveness. Logits are always produced by
the IO — untied dense head, adapter + E scoring, or (direct mode)
E scoring alone; a chain layer's output can never BE the logit vector
directly.

Classes:

| class | input | output | spec |
|---|---|---|---|
| bit-chunks, plain | one-hot ⊕ bp bits | learnable dense head + cap | `bits.4+bp`, `bytes` (unchanged) |
| bit-chunks, embedded | value-emb + pos-emb (added), scaled | adapter + tied E + cap | `bits.4.emb.8` |
| tokens, embedded | token-emb, scaled | adapter + tied E + cap | `tokens.256000.gemma.emb.2560` |
| bits.1 scalar | 1 value (+ pos) | single logit (sigmoid trick) | the `ntokens<=2` special case inside the classes above; promoted only if the code gets awkward |

`emb` **implies tied** — untied-embedding (ALBERT-style factorization
with a full head) doesn't get its own spelling. NOTE: this changes the
semantics of the already-committed `tokens.*.emb` input, which
currently pairs with an untied implicit head; it becomes the tied
class when the IO abstraction lands. The plain class is exactly
today's models plus soft-capping — no spec migration, full DB
continuity.

## Output structure (tied classes)

    query  = W_a @ h + b_a        # the ADAPTER: implicit dense K -> d,
                                  # with bias; added by default...
    logits = cap * tanh((query @ E_values.T) / cap)   # ...no per-vocab bias

- **Adapter added by default** (mirrors how the untied head is always
  present); only the explicit `.direct` mode below omits it. It
  accepts any last-hidden width K, so tied models compose
  with arbitrary chains and ALL existing mutations (last-layer
  resizes, appends, emb.d changes) remain valid — no special
  neighborhood logic. Crucially it decouples memory width from
  embedding width (`bytes.emb.2|rnn.8.tanh` = 8-dim state, 2-dim
  table), and it separates the prediction role from the state role, so
  the "don't end with a recurrent cell" concern doesn't apply in this
  mode. Its bias contributes `<b_a, E>` — a rank-limited learned prior
  over the vocabulary.
- **Direct mode** (`bytes.emb.8.direct`) skips the adapter:
  `logits = cap*tanh((h @ E.T)/cap)`. Hard rule: last hidden width ==
  d. This is the Gemma-exact form. A conditional adapter (auto-skip
  when K == d) was considered and REJECTED: a width mutation silently
  adding/removing a whole layer makes model behavior discontinuous in
  its dimensions and confuses the predictors. Mode is explicit in the
  spec; a direct-mode model is a one-hop neighbor of its adapter twin.
- **No bias on the E-multiplication** in either tied mode (matches
  Gemma; a full N-vector would reintroduce the parameters tying
  exists to eliminate).

## Input structure (tied classes)

- Table has `ntokens + npositions` rows of width d, `npositions =
  8/nbits`; position rows are omitted when npositions == 1 (bytes,
  tokens). Position embeddings are **added** to value embeddings
  (concat is a constrained special case of add; RoPE is categorically
  wrong here — it buys relative-shift equivariance, sub-byte position
  is absolute mod-8 structure). `+bp` disappears from emb spellings:
  positions are always embedded. `bits.4.emb.8` = (16+2)*8 = 144 table
  weights.
- **Input scale: one learned scalar, always on** (init sqrt(d)),
  multiplying the lookup. The two roles of a shared table want
  different magnitudes — loss pressure on logits drives rows to small
  per-dim scale, while the input side wants stream-scale activations.
  Gemma reconciles with a *fixed* sqrt(d) on input only (`x =
  E[id]*sqrt(d)`; head unscaled); that constant is only right under
  RMSNorm-everywhere, so texmo learns the scalar and the weight
  converter sets it to sqrt(2560) for Gemma. No output-side
  temperature (redundant against row scale).
- The (known) next position's embedding is NOT added on the output
  side: candidates share the position, so its score contribution
  `<query, q_pos>` is constant across them and **cancels in softmax**
  — provably a no-op. Position reaches the output only through the
  hidden state.
- Shift-pad initial vector: mean of the value rows (the uniform
  distribution pushed through the table); uniform 1/N in the plain
  class. BOS semantics lives in the token stream (tokenizer add_bos),
  not the padding.
- ntokens == 2 (bits.1): single logit = score difference
  `<query, e1 - e0>`, keeping the existing sigmoid-trick convention.

## Soft-capping — always on, all classes

`logits = 30*tanh(logits/30)` after the head. Infrastructure, not a
searchable knob: at equilibrium it is loss-invisible for our regime
(compression starts ~|logit|>10 where the cross-entropy delta is
~1e-6 b/token; the entropy floor is ~N*e^-60/ln2 ~ 1e-21 b/token), it
is monotone (greedy sampling unchanged), and its real effects are
bounded max loss (~86 bits/token instead of inf) and vanishing
gradients at extremes — fewer diverged runs. DB comparability with
pre-cap results is safe.

## Validity

- **Adapter mode: no new restrictions.** Any chain, any widths, any
  ending (the adapter separates roles); the empty chain `bytes.emb.2|`
  is a legitimate rank-d asymmetric bigram — the emb analog of the
  already-legal `bits.1+bp|`.
- **Direct mode:** (a) last hidden width == d (hard); (b) empty chain
  invalid — direct scoring of the input embedding gives a *symmetric*
  bigram, `logit(b|a) = E_a·E_b = logit(a|b)` identically, argmax
  "repeat the current token" for equal-norm rows; (c) soft endings
  principle: recurrent cells output their own state, so a recurrent
  ending forces state == logit query (a predictive embedding is not a
  sufficient statistic for the future) — legal but predictably weak;
  friendly endings are rmsnorm (Gemma's own) and the explicit bare
  `dense.d` adapter.
- **projects_input generalizes, one rule:** the IO output "projects"
  iff a learnable matrix sits in front of E — untied head: yes;
  adapter mode: yes (trailing bare dense collapses into the adapter,
  invalid as before); direct mode: NO (E is pinned, so a trailing bare
  `dense.d` is the meaningful d x d factorized adapter — the validity
  flip).

## Search mutations

The default adapter keeps the neighborhood logic intact; the only
additions are connective:

1. **Mode swap, plain <-> embedded:** direct transcription
   (`bytes|rnn.2.tanh` <-> `bytes.emb.d|rnn.2.tanh`), letting
   mutations refine from there rather than being clever about it.
2. **Mode swap, adapter <-> direct** when K == d (param delta = the
   adapter).
3. `emb.d` x2 mutations — always valid (the adapter absorbs any
   mismatch).

Accounting notes: tied models report num_weights without a full head,
so the small-model Pareto front re-ranks when these classes land —
re-testing bits.4-vs-bytes under honest accounting. Tying saves
PARAMETERS, not FLOPs (the head still scores N*d per position): the
timing model's output features keep their cost terms while num_weights
drops — the first place the two diverge.

## Worked example: minimal RNN

- `bytes.emb.2|rnn.2.tanh` — table 512 + scale 1 + rnn 10 + adapter 6
  = **529** weights.
- `bytes.emb.2|rnn.8.tanh` — 512 + 1 + 88 + 18 = **619**: 8-dim memory
  with a 2-dim table, impossible without the adapter.
- `bytes.emb.2.direct|rnn.2.tanh` — 523 (state doubles as query:
  legal, predictably weak).
- Smallest byte model: `bytes.emb.1|dense.1.tanh` = **261** (direct:
  259) vs **769** untied — the head savings that motivate all of this.

## Gemma fidelity mapping

`tokens.256000.gemma.emb.2560.direct` + final `rmsnorm` as the last
chain layer + cap 30 + input scale loaded as sqrt(2560):
`logits = 30*tanh((h @ E.T)/30)`, inputs `E[id]*sqrt(d)`, no biases
anywhere in the IO. Exact.

## Implementation order

1. IO abstraction + tokens class (adapter + direct + scale + cap) —
   the RecurrentGemma critical path; includes the tokens.*.emb
   semantic change to tied. The plain class falls out as today's head
   re-expressed (+ cap).
2. Embedded bit-chunks class + the mode-swap neighbors — search
   facing; separate commits.
3. Legacy Model/ModelDef retirement remains a prerequisite for
   deleting the old input plumbing (tracked separately).
