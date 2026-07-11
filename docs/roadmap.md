# Roadmap

## Grand goal

Given a parameter budget N and compute budget T, answer:
1. What is the best model architecture?
2. How should it be trained? — i.e. which metaparameters, what LR schedule,
   and potentially whether to use incremental training (adding layers
   gradually) instead of training from scratch.

## Architectures to add

### LSTM-family bake-off

Goal: figure out which modern LSTM tricks (matrix state,
exponential gating, multiplicative recurrent transitions) are
actually useful at small weight budgets. All three become layer-
type-swap neighbours of `lstm.X`.

* **matLSTM** (Beck et al. 2024, xLSTM paper, "mLSTM" in the
  paper — renamed here to disambiguate from the unrelated Krause-
  2016 mLSTM). LSTM with a matrix-valued cell state, covariance /
  outer-product write, and exponential input/forget gates (both
  input-dependent). Strict generalisation of vector LSTM. Tests
  whether content-dependent gating + matrix state work at small
  weight budgets where MSR didn't.
* **sLSTM** (Beck et al. 2024, xLSTM paper). Scalar-state companion
  to matLSTM — keeps the exponential gating + stabilisation tricks
  without the matrix cell. Tests whether the "modern LSTM tricks"
  alone (without matrix state) help at our scale.
* **mLSTM (multiplicative)** (Krause, Murray, Renals & Lu 2016).
  Input-modulated recurrent transitions — the recurrent transition
  function is per-input. Scalar state. Different family from the
  xLSTM mLSTM; the name collision is unfortunate.

Decision point after the three are on the board: if matLSTM beats
LSTM at our parameter budgets, follow up with **matGRU** — same
matrix-state machinery but with mGRU's single input-dependent
forget gate instead of LSTM-style input/forget pair. Novel; not in
any paper. Tests whether the GRU-beats-LSTM observation we already
see at scalar state carries over to matrix state.

### Other

* **DeltaNet.** Linear-attention with delta-rule updates:
  `S_t = S_{t-1}·(I − β k_t k_t^T) + β k_t v_t^T`, with `β_t` learned
  per token. Content-dependent forgetting in key-space; equivalent
  to online L2 regression of values against keys (i.e. a closed-form
  flavour of test-time training). Parallel-trainable via the
  Yang-et-al 2024 scan. Tests the "MSR's fixed γ is what hurt"
  hypothesis directly. Do after the LSTM-family bake-off.

* **S4D.** Simplest structured-state-space model: diagonal linear
  recurrence with complex eigenvalues, HiPPO initialisation.
  Different axis from MSR/DeltaNet — long-range memory via
  polynomial decay rather than gated/matrix state. Weight count is
  `O(d_state)` per layer; at `d_state=4` it's very competitive on
  parameter budget. Worth comparing against lrnn at 1000 weights.

* **TTT-MLP** (speculative; decide after DeltaNet). Hidden state =
  weights of a small MLP, updated each step by an inner SGD step on
  a self-supervised loss. Strictly more expressive than DeltaNet,
  but requires backprop through inner SGD at training time. At
  ~1000-weight budgets, in-place adaptation may matter more than at
  scale.

* **Attention / Transformer layer.** Use PyTorch's built-in
  `nn.MultiheadAttention` or `nn.TransformerEncoderLayer`. Key
  decisions: causal masking, position encoding (RoPE?), head count
  as a search axis. Deferred until at least one matrix-state
  layer is on the Pareto front — comparing attention against MSR
  alone isn't useful.

### Candidates to revisit

* **HGRN2** (Qin et al. 2024, "Hierarchically Gated Recurrent
  Network"). Adds depth-dependent gating — lower layers have
  lower forget rates (longer memory), upper layers shorter. The
  gain comes mostly in multi-layer models; less obviously useful
  at our typical 1-2 hidden-layer regime, so deferred until we
  have a baseline that benefits from depth.
* **LRU** (Orvieto et al. 2023, "Resurrecting Recurrent Neural
  Networks for Long Sequences", DeepMind). Linear recurrence with
  complex eigenvalues + input-dependent gating — essentially S4D-
  with-input-dependent-gates. Revisit after S4D so we have a
  baseline structured-recurrence to compare against.

## Infrastructure

* **Split `is_valid` into invalid vs search-ineligible.** Today one
  flag conflates "this model cannot run / makes no sense" with "this
  model runs fine but search shouldn't propose it" (off-whitelist
  bits variants, non-power-of-2 embedding widths like the load-only
  `tokens.256000.gemma.emb.2560`). Two predicates — `is_valid` and
  something like `in_search_space` — would let reports/eval treat
  load-only models as first-class while keeping the search grammar
  tight.

## Training procedures

* **Progressive layer-wise training.** For an N-hidden-layer model,
  the schedule is `2N − 1` phases: N "growth" phases (add l_i with
  prior layers frozen, fresh output projection) interleaved with
  `N − 1` "joint" phases (unfreeze everything trained so far, train
  jointly). Total step count: `(2N − 1) · 2^K` for tunable K.
  Encoded as one bit (`progressive: bool`) on `Configuration`.
  Search neighbor: progressive at `(2N − 1) · 2^K` steps is a
  neighbor of non-progressive at `2^M` steps where
  `2^M ≤ (2N − 1) · 2^K < 2^{M+1}`. An older version of this
  manager implemented something similar with stacked GRUs and
  reached lower loss than joint training, supporting the bet that
  small-model joint optimisation gets stuck in local minima that
  layer-wise training avoids.

## Run an off-the-shelf open-source LLM

(e.g. Gemma, Llama). Not a search target — purely about
exercising the runtime on a real model. Roughly half the
necessary building blocks already exist; what's missing:

- Attention (see above).
- Model load/save in the upstream checkpoint formats (HF
  safetensors at minimum; possibly GGUF).
- Adapt the existing Rust/Python tokenizer (DP-optimal,
  custom tokenset format) to consume the upstream model's
  SentencePiece/BPE vocab — mostly a format-conversion job.
- Model-specific quirks: RoPE variants, RMSNorm, SwiGLU, GQA,
  sliding-window attention, etc. Accumulate as we go.
