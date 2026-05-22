# Roadmap

## Grand goal

Given a parameter budget N and compute budget T, answer:
1. What is the best model architecture?
2. How should it be trained? — i.e. which metaparameters, what LR schedule,
   and potentially whether to use incremental training (adding layers
   gradually) instead of training from scratch.

## Architectures to add

* **mLSTM (xLSTM family).** LSTM with a matrix-valued cell state and
  scalar input/forget gates (both input-dependent). Strict
  generalisation of LSTM. Neighbor relation: `lstm.X <-> mlstm.X` as
  a type swap. Cheapest matrix-state experiment — directly tests
  whether content-dependent gating (which MSR lacked) is what makes
  matrix-state useful at small weight budgets.

* **DeltaNet.** Linear-attention with delta-rule updates:
  `S_t = S_{t-1}·(I − β k_t k_t^T) + β k_t v_t^T`, with `β_t` learned
  per token. Content-dependent forgetting in key-space; equivalent
  to online L2 regression of values against keys (i.e. a closed-form
  flavour of test-time training). Parallel-trainable via the
  Yang-et-al 2024 scan. Tests the "MSR's fixed γ is what hurt"
  hypothesis directly. Do after mLSTM.

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
