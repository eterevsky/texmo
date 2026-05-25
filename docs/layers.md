# Layers

All layers are implemented in both PyTorch (`{Name}Module`) and JAX
(`{Name}Jax`), with a shared `{Name}Def` descriptor. See
[`architecture.md`](architecture.md) for the abstractions and
[`backends.md`](backends.md) for backend trade-offs.

Layer files live in `texmo/layers/`. In specs, layers are separated by `-`
and the full model is `<input>|<layer1>-<layer2>-...`.

## Notation

- `x` — input at a timestep, shape `(input_size,)`
- `h` — hidden state, shape `(size,)`
- `W_ih`, `W_hh`, `b` — weight matrices and biases
- `σ` — sigmoid
- Hidden sizes must be powers of 2.
- All new layer weights are Xavier-uniform initialized for matrices and
  zero for biases. (Note: PyTorch's `nn.Linear`, `nn.RNN`, `nn.GRU`,
  `nn.LSTM` use their own uniform init, which is close but not identical.)

## Parameter-count convention

PyTorch's built-in `nn.RNN`, `nn.GRU`, `nn.LSTM` use **two bias vectors
per gate** (`bias_ih + bias_hh`), a historical quirk inherited from Lua
Torch. Our JAX implementations use **one bias per gate**, which is more
common in modern practice (GPT-2, LLaMA, etc.). `num_weights` on the
`LayerDef` reports the single-bias count; for PyTorch, the actual
module has more parameters by the extra bias vectors.

## Dense (`dense.<size>.<activation>`)

Standard feed-forward layer: `out = activation(W_ih @ x + b)`.

- Weights: `W_ih: (size, input_size)`, `b: (size,)`.
- Activations: `relu`, `gelu`, `tanh`.
- Stateless.
- `num_weights = size * input_size + size`.

## RNN (`rnn.<size>.<activation>`)

Elman RNN: `h = activation(W_ih @ x + W_hh @ h_prev + b)`.

- Weights: `W_ih: (size, input_size)`, `W_hh: (size, size)`, `b: (size,)`.
- Activations: `relu`, `gelu`, `tanh`.
- PyTorch: `tanh`/`relu` variants use cuDNN-fused `nn.RNN`; `gelu` falls
  back to a Python loop since `nn.RNN` doesn't support it.
- JAX: always uses `lax.scan` with the input projection hoisted out of
  the scan (single batched matmul over the full sequence for `W_ih @ x`).
- `num_weights = size * (input_size + size) + size`.

## GRU (`gru.<size>`)

Standard GRU:

    r = σ(W_ir x + W_hr h + b_r)       # reset gate
    z = σ(W_iz x + W_hz h + b_z)       # update gate
    n = tanh(W_in x + b_n + r * (W_hn h))   # candidate
    h_new = (1 - z) * n + z * h

- JAX layout: input projections for all three gates stacked into a single
  `w_ih` of shape `(3*size, input_size)`; hidden projections for `r` and
  `z` stacked into `w_hrz` of shape `(2*size, size)`; `w_hn` stays
  separate (it's gated by `r`, so it can't be fused with `r, z`).
- PyTorch: cuDNN-fused `nn.GRU`.
- `num_weights = 3 * size * (input_size + size) + 3 * size`.

## mGRU (`mgru.<size>`)

Minimal GRU variant with a single gate (update = 1 − forget):

    f = σ(W_fx x + W_fh h + b_f)
    hc = tanh(W_hx x + W_hh (f * h) + b_h)
    h_new = (1 - f) * h + f * hc

Two projections; biases live on the input-side linears. In our tests,
mGRU often matches or beats GRU at the same parameter count.

- JAX layout: stacked input projections `w_ih: (2*size, input_size)`;
  hidden projections `w_fh` and `w_hh` kept separate because `w_hh` is
  gated by `f`.
- PyTorch: custom Python-loop implementation (no built-in mGRU).
- `num_weights = 2 * (size * (input_size + size) + size)`.

## minGRU (`mingru.<size>`)

Even more minimal — gate and candidate depend only on the input:

    z = σ(W_z x + b_z)
    h_new = W_h x + b_h
    h = (1 - z) * h_prev + z * h_new

No hidden-to-hidden recurrence — the only time dependence is the
elementwise mixing. From https://arxiv.org/abs/2410.01201.

- JAX layout: stacked input projections `w_ih: (2*size, input_size)`.
- PyTorch: custom Python-loop implementation.
- `num_weights = 2 * size * input_size + 2 * size`.

## LSTM (`lstm.<size>`)

Standard LSTM with four gates (forget, input, output, candidate):

    f, i, o, g = split(W_ih x + W_hh h + b)
    f, i, o = σ(·);  g = tanh(g)
    c_new = f * c + i * g
    h_new = o * tanh(c_new)

- State is `(h, c)` — both needed for inference.
- JAX layout: all 4 gates stacked into one big pair of matrices
  `w_ih: (4*size, input_size)`, `w_hh: (4*size, size)` and a single bias
  `b: (4*size,)`. This gives one matmul per sequence (hoisted) and one
  matmul per timestep (inside scan) instead of 4 of each.
- PyTorch: cuDNN-fused `nn.LSTM`.
- `num_weights = 4 * size * (input_size + size) + 4 * size`.

## mulLSTM (`mullstm.<size>`)

Multiplicative LSTM from Krause, Murray, Renals & Lu 2016
("Multiplicative LSTM for Sequence Modelling"). Renamed here
to `mullstm` to disambiguate from `matlstm` (Beck-2024 xLSTM) --
the literature calls both "mLSTM" but they're unrelated
architectures.

The key idea: the previous hidden state `h_{t-1}` is element-wise
modulated by an input-dependent projection before reaching the
standard LSTM gates. Different inputs give different effective
recurrent transitions.

    m       = (W_mx x) ⊙ (W_mh h_{t-1})              # (size,)
    gates   = W_ih x + W_hh m + b                     # (4·size,)
    z       = tanh(gates[:s])
    i,f,o   = sigmoid(...)
    c_new   = f · c + i · z
    h_new   = o · tanh(c_new)

- State per sample: `(h, c)` -- identical to vanilla LSTM.
- Weights add a multiplicative pair on top of LSTM's gates:
  - `w_mx: (size, input_size)` -- multiplicative input projection.
  - `w_mh: (size, size)` -- multiplicative hidden projection.
  - Plus the usual `w_ih`, `w_hh`, `b` (4 stacked gates with bias).
- `num_weights = 5·size·(input_size + size) + 4·size` -- vanilla
  LSTM plus `size·(input_size + size)` extra for the multiplicative
  path. ~25% more parameters than `lstm.X` at the same size.
- Forward in JAX is a `lax.scan` with both input-only projections
  (`W_ih x` and `W_mx x`) hoisted out of the scan body.

## sLSTM (`slstm.<size>`)

Scalar-state sLSTM from xLSTM (Beck et al. 2024). Same weight
shape and memory-mixing structure as `lstm` (single head, full
`R_*` matrices), with sLSTM's "modern LSTM tricks" layered on
top: exponential input gate with max-tracking stabilisation,
normaliser carry, and an attention-like `c / n` readout instead
of the usual `tanh(c)`. JAX only.

    gates = W_ih x + W_hh h + b                      # (4·size,)
    z     = tanh(gates[:s])                          # cell input
    o     = sigmoid(gates[3s:4s])                    # output gate
    log_f = log_sigmoid(gates[2s:3s])
    m_new = max(log_f + m, gates[s:2s])              # stabiliser
    i_st  = exp(gates[s:2s] - m_new)                 # <= 1
    f_st  = exp(log_f + m - m_new)                   # <= 1
    c_new = f_st · c + i_st · z
    n_new = f_st · n + i_st
    h_new = o · (c_new / n_new)                      # paper eq (10)

- State per sample: `(h, c, n, m)` — all `(size,)` vectors. `n`
  and `m` add zero parameters; they're derived from gate values.
- `num_weights` is **identical to LstmDef**:
  `4·size·(input_size + size) + 4·size`. Useful comparison
  point: the head-to-head with `lstm.X` isolates the effect of
  the exp-gate + normaliser + stabiliser tricks at the same
  parameter budget.
- Single head only. Multi-head sLSTM (memory mixing within
  heads, not across) is deferred until single-head data tells us
  whether the bake-off is worth pursuing further.
- Forward is `lax.scan` with input projection `W_ih x` hoisted;
  everything else depends on the `h` carry so stays in the body.
- Output uses pure `c / n` per the paper (no `max(n, 1)`
  clipping); the stabiliser keeps both numerator and denominator
  in fp32 range during training.

## matLSTM (`matlstm.<size>`)

Matrix-state LSTM from xLSTM (Beck et al. 2024 —
"xLSTM: Extended Long Short-Term Memory"). Cell state is a `(D, D)`
matrix updated by an outer-product write, with scalar input /
forget / output gates and exponential gating + max-tracking
stabilisation for numerical bounds. JAX only.

    q     = W_q x
    k     = (W_k x) / sqrt(D)
    v     = W_v x
    i_pre, f_pre, o_pre = W_ifo x + b_ifo                # scalars
    log_f = log_sigmoid(f_pre)
    m_new = max(log_f + m, i_pre)                        # stabiliser
    i_st  = exp(i_pre - m_new)
    f_st  = exp(log_f + m - m_new)
    o     = sigmoid(o_pre)
    C_new = f_st · C + i_st · outer(v, k)
    n_new = f_st · n + i_st · k
    h     = o · (C_new @ q) / max(|n_new · q|, 1)

- State per sample: matrix `C: (D, D)`, normaliser `n: (D,)`,
  stabiliser `m: scalar`.
- No biases on Q / K / V (transformer convention); single bias per
  scalar gate.
- Constraint: `size >= 2`. `D = 1` would degenerate to a scalar cell
  and reduces to a (poorly conditioned) scalar LSTM variant.
- `num_weights = 3 * size * input_size + 3 * input_size + 3`.
- Forward in JAX is a `lax.scan` over time — no parallel form like
  MSR's, since the exponential-gated cumulative product through
  `(f_st, i_st)` doesn't factor cleanly. The recurrent-eval path
  (see `model_jax.forward_recurrent`) handles it the same way as
  any other scan-based layer.

## Suffix (`suffix.<length>`)

Sliding window: stacks the last `length` inputs into a single vector of
size `length * input_size`. No learned weights.

    out_t = concat(x_{t-length+1}, ..., x_t)

Useful as a lightweight n-gram feature before a downstream dense layer.
In training mode, `forward` emits `seq_len - length + 1` outputs (the
window requires `length - 1` padding tokens, supplied by the model's
`total_padding` machinery).

- State: `(length - 1, input_size)` — the recent history buffer.
- `length` attribute on the def is > 1; ModelDef enforces that two
  adjacent suffix-like layers are invalid.
- `num_weights = 0`.

## Norm (`norm`)

Elementwise L2 normalization along the feature axis:

    out = x / max(||x||_2, eps)

No learned weights. Stateless.

- Same `size` as `input_size`.
- Validity: first layer can't be norm, and you can't have two adjacent
  norms or norm-after-suffix.
- `num_weights = 0`.

## Latent (`latent.<size>.<reps>`)

Depth-recurrent dense layer — iterative refinement of a latent state.
Inspired by https://arxiv.org/abs/2502.05171.

    e = W_i @ normalize(x) + b
    s_0 = 0
    s_i = tanh(W_r @ normalize(s_{i-1}) + e)   for i = 1..reps
    out = s_reps

The input contribution `e` is re-injected at every iteration; this is
what makes the recurrence stable and gives the layer its "iterative
refinement" character.

- Normalization along the feature axis with `max(||·||, eps)`.
- No time recurrence — each position is independent.
- JAX: uses `lax.scan` over `reps` with fixed length.
- PyTorch: `reps` Python loop iterations.
- `num_weights = size * input_size + size + size * size`.
- Constraints: `size >= 2`, `reps >= 2`, both powers of 2.
- Note: the paper initializes `s_0` randomly to encourage
  path-independence. We zero-initialize for both backends until we see
  that the layer is commonly picked by search and a randomized variant
  is worth the RNG-plumbing.

## Lrnn (`lrnn.<size>.<reps>`)

Latent-recurrent RNN: combines time recurrence with latent reasoning.
At each timestep, runs `reps` refinement iterations using the previous
timestep's hidden state as additional context:

    e_t = W_i @ normalize(x_t) + W_h @ normalize(h_{t-1}) + b
    s_0 = 0
    s_i = tanh(W_r @ normalize(s_{i-1}) + e_t)
    h_t = s_reps

- JAX: outer `lax.scan` over time, nested `lax.scan` over reps.
- PyTorch: Python loops for both.
- `num_weights = size * input_size + size + size * size + size * size`.
- Constraints: same as Latent.

## Lmgu (`lmgu.<size>.<reps>`)

Latent-recurrent mGRU — combines mGRU's gating with `lrnn`-style
depth-recurrent iteration of the gate and candidate.

Per token, the gate `f` and candidate `hc` are jointly refined for
`reps` iterations starting from zeros; after the loop, the standard
mGRU blend updates the time-recurrent state:

    hc[0] = 0
    f[0]  = 0
    for i in range(reps):
        hc[i+1] = normalize(tanh(
            W_h_i x + W_h_h (f[i] * h) + W_h_s hc[i] + b_h))
        f[i+1]  = sigmoid(
            W_f_i x + W_f_h h + W_f_s hc[i+1] + b_f)
    h_new = (1 - f[reps]) * h + f[reps] * hc[reps]

- State per sample: `h: (size,)` — same as mgru. The `hc, f`
  iteration is local to a single token.
- Only `hc` is L2-normalized — this is the architecturally
  significant constraint (direction-only candidate). Raw `x` and `h`
  are fed to the W matmuls without normalization, matching standard
  mGRU.
- Unit-norm constraint on `hc` (the `normalize` after `tanh`) gives
  a clean factorisation: `hc` represents direction, `f` represents
  magnitude. Both `f` and `hc` are computed against unit-norm `hc`,
  so the final blend is consistent.
- Weights: 6 matrices (input / h / state projections, one set each
  for `hc` and `f`) plus 2 biases.
  `num_weights = 2·size·input_size + 4·size·size + 2·size`.
- `num_mults = num_weights · reps` (same coarse approximation as
  lrnn — the recurrent matmuls fire `reps` times per token).
- Constraints: `size > 1`, `reps >= 2`, both powers of 2.
- Forward in JAX hoists the input-only projections outside the
  outer time scan; the per-iteration matmuls stay in the inner
  `reps` scan.

Neighbor relations:
- `lmgu.X.R ↔ lrnn.X.R` (family swap at same size and reps).
- `lmgu.X.2 ↔ mgru.X` (the non-iterating cousin at `reps=2`).
- 2× mutations on size (clamped to `> 1`) and reps (clamped to `>= 2`).

## Skip (`skip.<X>.<add|cat>`) — residual connections

Pseudo-layer that marks the start of a residual connection spanning
`X` layers. It takes no weights and doesn't transform activations on
its own — it's a marker for `Model`/`ModelJax` to save the source
activation and merge it back `X` layers later.

- `.add` — elementwise add with soft size matching: the merged output
  has size `max(skip_src_size, merge_point_size)`. The first
  `min(skip_src_size, merge_point_size)` channels are summed; the
  remaining channels from the larger vector pass through unchanged.
- `.cat` — concatenation; output size is `skip_src_size + merge_point_size`.

Example: `bytes|skip.2.add-dense.32.tanh-dense.32.tanh-dense.64.gelu`.
The skip starts after the input (256 dims), skips over two `dense.32`
layers, and merges before the `dense.64`. At the merge point the main
path has 32 dims and the source has 256, so with `.add` the merged
activation is 256 dims (32 summed, 224 appended).

See [`skip.md`](skip.md) for the full design: validity rules, search
mutations, and how merges are handled in `Model`/`ModelJax`.

## Neighbor relations (search)

The search walks the architecture space by generating "neighbors" of a
given layer. Each layer type declares its own neighbors via
`LayerDef.neighbors()`. Summary:

- **dense ↔ rnn** — with the activation preserved.
- **dense.X.tanh ↔ latent.X.2** — promotes a dense-tanh to its
  depth-recurrent twin.
- **rnn.X.tanh ↔ lrnn.X.2** — likewise for recurrent.
- **rnn ↔ gru, mgru, mingru** — activation is dropped when moving to a
  gated cell.
- **gru, mgru, mingru, lstm** are mutual neighbors, with one exception:
  **lstm ↮ mingru** (mingru has no hidden-to-hidden state, so it's
  structurally too different).
- **latent.X.Y ↔ lrnn.X.Y** — swap between the two depth-recurrent
  variants at matching size/reps.
- Size `S` and reps `R` mutations: `×2`, `÷2` (keeping powers of 2 and
  minimum sizes).

See `layer.py:LayerDef.neighbors` for the full rules.
