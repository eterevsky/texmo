# Input and output: from token ids to logits

A texmo model is a next-token predictor. Its input is a stream of
integer token indices; its output is a vector of logits over the same
token set at every position. Everything in between — the layer chain
written in the spec — operates on dense vectors. The **codec** — a
coder/decoder pair owning both ends of the model — converts between
the two worlds: it encodes ids to vectors at the front and decodes
hidden activations to logits at the back. Two implementations share
one API (`layers/codec.py` holds the common pieces):
`OneHotCodec` for the fixed-codebook kinds (kind 1 below, plus
`tokens.*.oh`) and `EmbeddingCodec` (upcoming) for the tied embedded
kinds (2 and 3).

In a spec like `bits.4+bp|rnn.8.gelu`, the part before `|` names the
codec; the part after names the hidden chain. The **output head** is the
final transformation that turns the last hidden layer's vector into
the logits — one score per token in the set. In the simplest case it
is a dense projection from the last hidden layer's activations to
`ntokens` scores; in the tied kinds below it scores the hidden state
against the embedding table instead. The head is **implicit**: it is fully determined by the
codec kind (which knows the token set) plus the width of the last hidden
layer, so it never appears in the spec. Logits are always produced by
the codec — a chain layer's output is never used as logits directly.

## The model contract

- **Tokens.** The training byte stream is tokenized either by a fixed
  rule (bytes, sub-byte bit chunks) or by a learned tokenset from
  `tokens/` (e.g. the converted Gemma BPE set). `ntokens` ranges from
  2 to 256,000.
- **Autoregression.** To predict the token at position t, the model
  sees tokens up to t-1 (inputs are shifted right). At position 0
  there is no history: the model receives the codec's *initial vector* —
  the encoding of "uniform distribution over tokens" — and its output
  there is the unconditional prior. This convention is
  tokenset-independent: training always uses the shift-pad shape, and
  the pad is a real position (attention layers attend to it like any
  other). Pretrained models with an explicit beginning-of-sequence
  token (Gemma) get it as a literal token in the id stream at
  *prompt/eval* time (the tokenizer's add_bos), with the meaningless
  pad-predicts-bos position simply excluded from scoring.
- **Loss.** Softmax cross-entropy over the logits. Training optimizes
  the mean loss **per token**, in bits. Evaluation reports the loss
  **per byte**, dividing by the tokenset's average bytes per token —
  which makes scores comparable across codec kinds with wildly
  different token granularities.
- **Binary special case.** When `ntokens <= 2` (bits.1) the head
  produces a single logit x, treated as `[x, 0]` — the log-odds /
  sigmoid trick.
- **Soft-capping.** All heads apply `logits = 30*tanh(logits/30)`.
  This is infrastructure, not a modeling knob: it is invisible to the
  loss at equilibrium and monotone (greedy sampling unchanged), but it
  bounds the maximum loss and kills gradients on runaway logits, so
  fewer runs diverge. *Status: implemented but opt-in
  (`parse_model2(..., cap=True)`) until its effect is measured against
  the uncapped result DB; the end state is always-on.*

## Kind 1: bit chunks, one-hot family — `bits.N[.oh][+bp]`, `bytes`

*Status: implemented as `OneHotCodec` (`layers/one_hot_codec.py`); the
current default for search models. Only the specs in actual use are
valid — `bytes`, `bits.1+bp`, `bits.2.oh+bp`, `bits.4.oh+bp` (and
`tokens.*.oh`); the other bits variants still parse and run for manual
experiments but `is_valid` rejects them, so search never proposes
them.*

Each byte is split into 8/N chunks of N bits (N in {1, 2, 4, 8};
`bytes` = `bits.8.oh`), so `ntokens = 2^N`. The input vector is either
the raw bits (N values in {0,1}) or a one-hot over the 2^N chunk
values (`.oh`). `+bp` appends the chunk's position within its byte,
encoded as bits — sub-byte models need to know the phase, since "third
bit of a byte" and "seventh bit of a byte" follow different statistics.

The head is a learnable dense taking the last hidden layer's
activations to `ntokens` scores, with bias. The initial vector — the
padding prepended to the shifted input so position 0 has something to
predict from (see the model contract above) — is max-entropy: uniform
1/2^N for one-hot, 0.5 per bit otherwise.

Why sub-byte at all: small models cannot afford 256-wide IO. A
`bytes|dense.1.tanh` model spends 768 of its 769 weights on the input
matrix and the head; `bits.1+bp` reads and predicts one bit at a time
with 4-dim IO. Empirically `bits.4` beats `bytes` up to at least 20k
weights, in large part because of exactly this overhead.

## Kind 2: embedded bit chunks — `bits.N.emb.d`, `bytes.emb.d`

*Status: designed as `EmbeddingCodec`, not yet implemented — see
[`tied_io.md`](tied_io.md) for the full rationale.*

Instead of one-hot vectors, chunk values get **learned embeddings**: a
table with one d-wide row per value plus one per within-byte position
(positions are *added* to values; the `+bp` flag disappears because
positions are always embedded).

The same table is **tied** into the output: the head scores the
(adapted) hidden state against the value rows,
`logits = cap*tanh((W_a h + b_a) @ E.T / cap)`. The adapter `W_a` — an
implicit dense from the last hidden width to d — is added by default:
it decouples the memory width from the embedding width
(`bytes.emb.2|rnn.8.tanh`: 8-dim state, 2-dim table) and frees the
last layer's output from doubling as the logit query. The explicit
`.direct` mode (`bytes.emb.8.direct`) omits the adapter and scores the
hidden state against the table directly — this requires the last
width to equal d, and is the exact shape used by tied transformer
LMs (Gemma ends its stack with a norm and scores against the table).

Why tying: the head is the single largest parameter block in a small
model, and it is largely redundant with the input table — both encode
"what does token v look like as a vector". Sharing them roughly halves
IO cost: the smallest byte model drops from 769 weights (untied) to
261. Because the same table now serves two roles that prefer
different magnitudes — the output role sets the row scale via loss
pressure on the logits, while the input side wants activation-scale
vectors — a single learned scalar multiplies the input lookup,
absorbing the difference in scale between the two roles.

## Kind 3: tokenized — `tokens.N.variation.emb.d[.direct]`

*Status: tokenizers and the one-hot mode (`tokens.*.oh`, part of
`OneHotCodec`) implemented; the embedded/tied mode belongs to
`EmbeddingCodec` and is designed, not yet implemented.*

For learned tokensets: `N.variation` names the tokenset file
`tokens{N}_{variation}.json`, resolved through the registry — e.g.
`tokens.256000.gemma.emb.2560` uses the converted Gemma BPE set (see
[`tokens.md`](tokens.md)). The layer itself never loads the file; the
spec fully describes the id space, and only the data sampler needs the
actual tokenizer.

Input is an embedding lookup (a one-hot mode exists for small custom
sets, but embedding is the default — a 256k one-hot would be absurd).
Output follows the same tied adapter/direct scheme as kind 2, minus
position embeddings (tokens are whole units; npositions = 1).
A tokenset may define a beginning-of-sequence token; when it does, the
tokenizer can prepend it to prompts (`add_bos`), so sequence starts
appear as a literal token in the id stream — the convention pretrained
models like Gemma rely on, alongside the synthetic initial vector that
training always uses.

This is the kind that runs pretrained RecurrentGemma:
`tokens.256000.gemma.emb.2560.direct` with a final `rmsnorm` in the
chain reproduces the reference head exactly (inputs scaled by
sqrt(2560), `logits = 30*tanh(h @ E.T / 30)`, no biases).

## Why the head is implicit

If the head were an explicit spec layer, every spec would end with a
projection to `ntokens`: `bits.4+bp|rnn.8.gelu` would be written
`bits.4+bp|rnn.8.gelu-dense.16`, with the trailing `dense.16`
producing the 16 logits. This was considered and rejected twice. It
adds no expressiveness (the codec must own the token-set width anyway),
it would let degenerate forms parse (`bits.4+bp|rnn.16.gelu` — a
recurrent cell whose state doubles as the logit vector), and it would
rename every conf in the results DB. The codec kind determines the head;
the spec stays a description of the *hidden* computation.
