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
- **Soft-capping.** All heads apply `logits = 30*tanh(logits/30)`.
  This is infrastructure, not a modeling knob: it is invisible to the
  loss at equilibrium and monotone (greedy sampling unchanged), but it
  bounds the maximum loss and kills gradients on runaway logits.
  *Status: on by default, validated against the result DB (2026-07,
  ~500 seed-paired confs): loss-neutral on healthy confs, rescues the
  logit-blow-up divergence mode, no effect on divergence born in
  hidden-layer state. `parse_model2(..., cap=False)` opts out for
  experiments.*

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
rows, `logits = cap*tanh(h @ E.T / cap)`, which requires the chain's
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
`tokens/tokens32_raw_fold.json` (`tokens.32.raw_fold.oh`, also valid
as `.emb.d`); its lossless successor `tokens/tokens64_shift.json`
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
- **The codec counts the table as weights**: `+ntokens` (32) in
  `num_weights` for variations ending in `fold` — the only
  non-learned knowledge in the composite byte model, priced in. No
  `num_mults` cost.

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

**The 64-token successor, `tokens64_shift`** (2026-07-20,
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
`residual = 0` and `extra_weights = 0`: the tail's information
moves in-band, where a model can *learn* it (an ignorant model pays
~8 bits per escaped byte, about what a uniform-bucket charge would
have been; a good one beats it) — no accounting caveats at all. The
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
charges, per-set weight surcharges via `codec.fold_extra_weights` —
remains for fold-32 and any future lossy set.

Search-reachable (2026-07-19): the fold set hangs off the bit ladder
at its nearest rung — `bits.4.oh+bp <-> tokens.32.raw_fold.oh` and
`bits.4.emb.X <-> tokens.32.raw_fold.emb.X` — and the usual oh<->emb
mode swap applies. The loss predictor carries the residual charge as
a global feature; the timing model treats the input as just another
type key.

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
