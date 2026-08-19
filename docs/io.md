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
`tokens.*.oh`) and `EmbeddingCodec` for the tied embedded kinds
(2 and 3).

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
- **No soft-capping.** Logits are the raw head output. Nothing sits
  between the head and the cross-entropy — see the note below for the
  cap that used to.

### Removed: the logit soft-cap (2026-08-19)

From 2026-07 to 2026-08-19 every head applied
`logits = 30*tanh(logits/30)` (the RecurrentGemma value), on the
theory that it was free insurance: monotone, so greedy sampling is
unchanged, loss-invisible at equilibrium, but bounding the maximum
loss and killing gradients on runaway logits. It is gone, not
defaulted off — there is no `cap` flag anywhere.

Two findings retired it. **It did not buy convergence.** The original
2026-07 validation compared capped runs against the DB's existing
population, which selects for confs that already converged. A 3-way
study (no cap / correct cap / the miscompiled cap below) on an
*unbiased* sample of historically-divergent confs found the correct
cap roughly convergence-neutral: of 8 blow-ups it rescued 2 and
caused 2. On runs that converged either way it was loss-neutral to
within ±0.1%, confirming the equilibrium argument — which is exactly
why it also cannot help.

**It was a miscompilation magnet.** The cap's shape — a matmul
followed by a bias and a constant scale — is precisely the fusion
pattern that triggered an XLA:GPU cuBLASLt alpha/bias miscompilation,
which cost days of debugging (the miscompiled cap was the third arm
of the study). A component with no measured benefit is not worth
carrying a class of silent wrong-answer bugs for.

Data and per-conf numbers live in `scratch/cap_study/` (untracked
scratch, this machine only).

*Side effect, accepted:* `models/recurrentgemma-2b.json` previously
picked up the default cap, which matched Gemma's native
`final_softcap=30`. Without it the port's eval perplexity drifts
slightly from the reference. Greedy token matching is unaffected —
argmax is invariant under a monotone map — so the port's
token-for-token validation against HF still holds.

## Kind 1: bit chunks, one-hot family — `bits.N[.oh][+bp]`, `bytes`

*Status: implemented as `OneHotCodec` (`layers/one_hot_codec.py`); the
current default for search models. Only the specs in actual use are
valid — `bytes`, `bits.1+bp`, `bits.2.oh+bp`, `bits.4.oh+bp` (and
`tokens.*.oh`); the other bits variants still parse and run for
manual experiments but `is_valid` rejects them, so search never
proposes them.*

`+bm` was the minimal-width experiment: instead of transmitting the
position counter (`+bp`), a single byte-start marker bit (1 on the
first chunk of each byte), leaving the phase count to the model's own
state — UART framing, one input dimension narrower than `bits.1+bp`
(2 vs 4). Search verdict (2026-07): practically never better than
`+bp`, so it is retired — still parses and runs, but invalid, with no
incoming neighbor edges (its outgoing edge to `bits.1+bp` remains as
a migration bridge for the DB population).

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

*Status: implemented as `EmbeddingCodec`
(`layers/embedding_codec.py`) and search-reachable: every blessed
one-hot input has an `emb` mode-swap neighbor at the chain's final
width (and back), plus an emb domain ladder mirroring the one-hot
one. See [`tied_io.md`](tied_io.md) for the full rationale.*

Instead of one-hot vectors, chunk values get **learned embeddings**: a
table with one d-wide row per value plus one per within-byte position
(positions are *added* to values; the `+bp` flag disappears because
positions are always embedded).

The same table is **tied** into the output, and scoring is always
**direct**: the chain's last activation is scored against the value
rows, `logits = h @ E.T`, which requires the chain's
final width to equal d. There is no implicit adapter and no `.direct`
mode: a model that wants a projection before the table spells it as
an explicit trailing bare `dense.d` — the exact same bias-linear map
an implicit adapter would be, but visible in the spec and mutated by
the ordinary dense machinery (`bytes.emb.2|rnn.8.tanh-dense.2`:
8-dim state, 2-dim table). Direct scoring off a wider recurrent
output also works — the model can allocate orthogonal state subspaces
for the query and carry roles — so which form wins is the search's
question, not a baked-in default. The d in the spec must equal the
chain's final width — a spec that disagrees is invalid, exactly like
any other inconsistent spec (the parser never rewrites it; when the
search-facing pass lets mutations resize an emb model's last layer,
the mutation emits the matching d as part of the same edit). Chains
with no consistent width (suffix-scaled passthrough) are invalid.
Width-preserving chains — the Gemma shape, where the whole stack
inherits its width *from* the input — keep the spec's d load-bearing.
The empty chain (`bytes.emb.4|`) is the symmetric bigram
(`logits_j ∝ (E_t + P_p)·E_j`): handicapped on real text, but legal.

Why tying: the head is the single largest parameter block in a small
model, and it is largely redundant with the input table — both encode
"what does token v look like as a vector". Sharing them roughly halves
IO cost: the smallest byte model drops from 769 weights (untied) to
259 (`bytes.emb.1|dense.1.tanh` — the chain dense doubles as the
adapter). Because the same table now serves two roles that prefer
different magnitudes — the output role sets the row scale via loss
pressure on the logits, while the input side wants activation-scale
vectors — a single learned scalar multiplies the input lookup,
absorbing the difference in scale between the two roles.

## Kind 3: tokenized — `tokens.N.variation.emb.d`

*Status: implemented — tokenizers, the one-hot mode (`tokens.*.oh`,
part of `OneHotCodec`), and the embedded/tied mode
(`EmbeddingCodec`). Off-grid widths like `emb.2560` parse and run but
are load-only (`is_valid` requires a power of 2).*

For learned tokensets: `N.variation` names the tokenset file
`tokens{N}_{variation}.json`, resolved through the registry — e.g.
`tokens.256000.gemma.emb.2560` uses the converted Gemma BPE set (see
[`tokens.md`](tokens.md)). The layer itself never loads the file; the
spec fully describes the id space, and only the data sampler needs the
actual tokenizer.

Input is an embedding lookup (a one-hot mode exists for small custom
sets, but embedding is the default — a 256k one-hot would be absurd).
Output follows the same direct tied scoring as kind 2, minus
position embeddings (tokens are whole units; npositions = 1).
A tokenset may define a beginning-of-sequence token; when it does, the
tokenizer can prepend it to prompts (`add_bos`), so sequence starts
appear as a literal token in the id stream — the convention pretrained
models like Gemma rely on, alongside the synthetic initial vector that
training always uses.

This is the kind that runs pretrained RecurrentGemma:
`tokens.256000.gemma.emb.2560` with a final `rmsnorm` in the
chain reproduces the reference head exactly (inputs scaled by
sqrt(2560), `logits = 30*tanh(h @ E.T / 30)`, no biases).

## Fold tokensets: lossy folding with honest accounting

*Status: implemented 2026-07-19. Lossy set:
`tokens/tokens.32.fold.json` (`tokens.32.fold.oh`, also valid
as `.emb.d`); its lossless successor `tokens/tokens.64.shift.json`
(2026-07-20) is an ordinary tokens+sequences set, described below.
Generators: `scripts/make_fold32.py` / `make_shift64.py` compute the
mapping, any stored frequencies, and the residual from corpus byte
counts (`freq.txt`, produced by the Rust `count-bytes` command).*

A **fold** tokenset maps *every* byte to exactly one of its tokens —
a many-to-one projection, not a code. The 32-token set folds capitals
to lowercase, digits to `x`, quotes/brackets/underscore to `'`,
`!`/`:` to `.`, `;`/`-`/`/` to `,`, control bytes to space, and
everything else to `?`. Tokenization is a 256-entry table lookup
(`FoldTokenizer`), one token per byte; decoding prints each group's
head character.

**The accounting problem**: a model over folded text sees an easier
task, so its token cross-entropy is not comparable to a lossless
model's b/B — and the gap depends on knowledge (byte frequencies)
the model never had to learn. A trivial 1-token set scores 8 b/B
against uniform bytes, but ~4.7 b/B if corpus byte frequencies are
baked into the decode distribution: that knowledge has to be paid
for.

**The fix**: each fold set stores one number per token — the
corpus frequency of the group's head character within its group
(`head_freq`). Within a group, the head byte gets probability `p`
and the other members split `1 − p` equally. Composing the model's
token distribution with this within-group model yields a genuine
distribution over all 256 bytes, whose cross-entropy decomposes
exactly into the token loss plus a per-byte charge
(`-log2 P(byte | group)`). Two consequences:

- **Eval adds the charge**: `TokenSet.byte_loss` adds
  `stats.residual_bits_per_byte` (the corpus-average charge, 0.344
  b/B for the 32-token set; 0 for lossless sets) and every b/B
  conversion goes through it. Training is untouched — the charge is
  an additive constant with no gradients.
- **The codec counts the table as weights**: `+2·ntokens` (64) in
  `num_weights` — 32 head-token *selections* plus the 32 stored
  frequencies (the selection charge added 2026-07-29 to match
  hexbpe's one-weight-per-choice accounting; the full fold map
  assigns all 256 byte values, but one weight per group head is the
  agreed good-enough price). The only non-learned knowledge in the
  composite byte model, priced in; no `num_mults` cost. The value
  lives in the tokenset's `stats.extra_weights`.

Storing 32 numbers instead of the full 256-frequency table costs
only 0.03 b/B of accounting slack (0.344 vs 0.314 exact) because
most groups are `{letter, capital}` pairs where head + equal-split is
exact. The stored frequencies are deliberately *not* trainable: the
cross-entropy optimum of trainable values is exactly the empirical
frequency already baked in, and a static table keeps the charge a
per-tokenset constant, leaving within-tokenset rankings untouched.

Per-slice accounting (charging each eval chunk's actual bytes via
`FoldTokenizer.chunk_residual_bits`) is implemented but unused — the
corpus constant is exact in expectation over books3 slices.

**The 64-token successor, `tokens.64.shift`** (2026-07-20,
`tokens.64.shift.oh`), abandons folding entirely and is **fully
lossless** — the answer to fold-32's tax dominating at higher
weight counts. Lowercase letters and digits get their own tokens; a
**shift marker** (numbered ext token 0) encodes `A` as marker +
`a`, `\t` as marker + space, and six shifted-symbol pairs
(`% + [ ] `` \` → marker + `# - ( ) ' /`); 26 slots go to the most
frequent remaining bytes; every other byte (161 of them, 0.35% of
the corpus) spells its **hex code behind an escape** (numbered
token 1) using the existing digit/letter tokens —
`\xa0` = `[1, "a", "0"]`. Every byte round-trips exactly, so
`residual = 0`: the tail's information moves in-band, where a model
can *learn* it (an ignorant model pays ~8 bits per escaped byte,
about what a uniform-bucket charge would have been; a good one
beats it). `extra_weights = 64` — one weight per token-slot
selection (charged 2026-07-29 with the rest of the
one-per-choice convention; originally 0, which let the retired
set free-ride on the frontier against hexbpe-64's 68). The
cost is token inflation: 1.038 tokens/byte. Because every byte has
exactly one encoding, it is a plain tokens+sequences set: `type` is
`shift`, the generic DP `Tokenizer` handles it (the DP has no
choices and reproduces the fixed encoding; measured identical on
books3), and none of the fold machinery below applies. Numbered
specials rather than marker characters: control chars belong to
str→str *preprocessing* (capswords), numbered tokens to constructs
inside the tokenizer. Sequences list every byte without its own
token in increasing byte order, including bytes the corpus never
contained. The lossy-group accounting — head-frequency and uniform
charges, per-set weight surcharges via the tokenset's
`stats.extra_weights` (read by `codec.tokenset_extra_weights`) —
remains for fold-32 and any future lossy set.

**Retired 2026-07-28, re-enabled 2026-07-29**: `tokens.64.shift`
briefly lost its search eligibility to the hexbpe family below —
then the first day of hexbpe search showed shift *beating*
`tokens.64.hexbpe` by ~1–2% loss over most of the frontier above
~400 weights (every letter having its own token appears to matter).
It is schedulable again, bridged bidirectionally to
`tokens.64.hexbpe`; the retirement mechanism
(`search.RETIRED_INPUTS`, now empty) stays for the next candidate.

## Hexbpe tokensets: one schema for every size

*Status: implemented 2026-07-28. Sets:
`tokens/tokens{32,64,128,256}_hexbpe.json` (`tokens.N.hexbpe.emb.d`
/ `.oh`). Generator: the Rust `build-hexbpe` command; Python side:
`HexBpeTokenizer` + capswords2 in
[`processing.py`](../texmo/tokens/processing.py).*

The fold/shift generation solved lossiness but left each size a
bespoke design, carrying information that neither maps to weights
nor stays constant across sizes — which is what made the
weights→loss picture jagged around tokenset switches. **Hexbpe**
replaces them with a single schema for every `N >= 32`:

- **16 nibble tokens** (ids 0–15). Any byte without its own token
  spells as two nibbles — the fallback that makes every size
  lossless (`residual_bits_per_byte = 0`), with the tail's cost
  in-band where a model can learn it.
- **Selected bytes**: single-byte tokens, greedily chosen.
- **BPE merges**: pairs of existing tokens, greedily chosen; the
  JSON stores them as *string pairs* in rank order.

Selection and merging compete in one currency — emitted tokens
saved: a byte token saves `count(b)` (one nibble per occurrence), a
merge saves `count(AB)`. Comparing them by entropy instead would
smuggle in a model-strength assumption; token counts make the
greedy ordering well-defined (ties go to the byte — fewer stored
numbers).

**Weights**: `stats.extra_weights = selected_bytes + 2·merges` — one
number per byte choice, two per merge (each names two parents).
`num_weights` counts parameters, not bits, so tokenset decisions are
priced the same way; the codec reads the value from the tokenset
stats. This is the honest-accounting answer for *lossless* sets:
what the set knows about the corpus scales smoothly with N and is
charged like any other capacity.

**Processing is capswords2**: capswords plus `\x17`
(next-letter-uppercase, so `iPhone`/`McDonald` stop passing capitals
into the token space) and single inter-word space elision. Two
structural rules shape merges: a token's newline run may only touch
one end (paragraph starts stay usable as generation cut points), and
no letters after the last `\x16` (no token contains the *start* of a
following word).

**Stats semantics**: `bytes_per_token` charges tokens against **raw**
corpus bytes — the invariant every b/B is compared on —
while `scanned_bytes` is the processed sample length
(capswords2 output runs ~6% longer than raw) and only sizes
processed chunks.

**Encoding is the generic DP tokenizer**, not BPE by rank
(2026-07-29). Measured on the 256 set: the pure-Python merge loop
does 50k tokens/s vs DP's 791k — a 16x gap that scales with CPU
while the model scales with GPU — and DP emits only 0.53% fewer
tokens than the builder's merge-order encoding, so the stats skew is
far under the ±5% genre variation across corpus slices. The
literature agrees there is nothing to lose: unigram/DP-style
segmentation is equal-or-slightly-better than BPE-merge for the
model (Bostrom & Durrett 2020), and strict token-count minimization
is not itself a win either (Schmidt et al. 2024) — at N <= 256 the
difference is noise. `HexBpeTokenizer` (builder-exact BPE by rank)
remains for scripts that must reproduce the builder's counts.

| set | bytes | merges | extra weights | raw bytes/token |
| --- | ---: | ---: | ---: | ---: |
| 32  | 15 | 1   | 17  | 0.803 |
| 64  | 28 | 20  | 68  | 1.150 |
| 128 | 34 | 78  | 190 | 1.469 |
| 256 | 52 | 188 | 428 | 1.811 |

**shift-32** (2026-07-31, `tokens/tokens.32.shift.json`,
`tokens.32.shift`, type `shift_bucket`, processing `raw`): the
insight that killed the short-lived bucket-64 experiment (removed
before it ever ran): capswords2's word markers and case-marker split
exist *for multi-char tokens* — word-end markers let word tokens
absorb boundaries, all-caps markers let one token serve both
casings. A single-char-token set needs none of it; one shift marker
applied **at tokenization** (like shift-64, no processing pass)
does everything. 32 tokens = a numbered shift marker + the 24
letters except q/z + `space , . ' _` + two lossy faces: `0` (all
digits, uniform) and `*` (every other byte, uniform). 29 *semantic*
shift pairs — `a→A` … plus `space→\n`, `,→;`, `.→?`, `'→"`, `_→-`
— so capitalization is exact and the pair table transfers each
host's statistics to a genuinely similar partner. That table is
stored corpus knowledge: `extra_weights = 58` = 29 selections + 29
pairs, one per stored choice. Coverage: 97.5% of bytes lossless,
digits keep their *shape* (`1984` → four digit tokens), residual
0.151 b/B (digits 0.87% at log2 10 + catch-all 1.62% at log2 188),
1.047 tokens/byte. Against fold-32: 0.151 vs 0.348 residual at 58
vs 64 weights — fold stays searchable regardless. Generator:
`scripts/make_shift32.py` from the raw `freq.txt`.

**fold-128** (2026-08-03, `tokens/tokens.128.fold.json`,
`tokens.128.fold`): the simplest possible lossy rung — the 127 most
common **raw** bytes as singleton tokens plus one uniform
catch-all, no processing at all, exactly one token per byte
(`bytes_per_token = 1.0`). All letters, capitals and digits select
naturally; the bucket's 129 values carry 0.11% of the stream, so
the residual is a near-free 0.0076 b/B at `extra_weights = 127`
(selections only). Its lossless rival at the same size is
hexbpe-128 (190 weights, 1.469 raw B/token); the pair brackets the
compression-vs-simplicity trade at 128 tokens the way shift-64 and
hexbpe-64 do at 64. Generator: `scripts/make_bucket.py N
--processing raw` (the raw mode of the bucket generator; it refuses
to overwrite the curated `tokens.32.fold.json` without --force).
Edges: `tokens.64.shift <-> tokens.128.fold <-> bytes` and
`tokens.128.fold <-> tokens.128.hexbpe`, both codec modes.

Search-reachable (2026-07-28): the family hangs off the bit ladder
at its nearest rung and forms its own chain —
`bits.4.oh+bp <-> tokens.32.hexbpe.oh <-> tokens.64.hexbpe.oh <->
tokens.128.hexbpe.oh <-> tokens.256.hexbpe.oh`, with
`tokens.32.fold.oh <-> tokens.32.hexbpe.oh` bridging the old
family and `tokens.64.shift.oh <-> tokens.64.hexbpe.oh` hanging
shift off the chain at its own size. shift-32 sits between both
32-token incumbents and below its 64-token sibling:
`tokens.32.fold.oh <-> tokens.32.shift.oh <->
tokens.32.hexbpe.oh` and `tokens.32.shift.oh <->
tokens.64.shift.oh` (no direct bits.4 edge — reachable through
either incumbent). The same chains exist in emb mode at the shared
table width, and the usual oh<->emb mode swap applies. The loss predictor carries the residual charge and
log2(bytes/token) as global features; the timing model treats each
input as just another type key.

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
