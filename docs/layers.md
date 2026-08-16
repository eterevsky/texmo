# Layers

Every layer is a JAX implementation (`{Name}Jax`) plus a backend-
agnostic `{Name}Def` descriptor. (Until 2026-08 a subset also had
PyTorch `{Name}Module` implementations; those were removed with the
torch backend — see [`backends.md`](backends.md).) See
[`architecture.md`](architecture.md) for the abstractions.

Layer files live in `texmo/layers/`. In specs, layers are separated by `-`
and the full model is `<input>|<layer1>-<layer2>-...`.

## Notation

- `x` — input at a timestep, shape `(input_size,)`
- `h` — hidden state, shape `(size,)`
- `W_ih`, `W_hh`, `b` — weight matrices and biases
- `σ` — sigmoid
- Hidden sizes must be powers of 2.
- All new layer weights are Xavier-uniform initialized for matrices and
  zero for biases.

## Parameter-count convention

Several RNN libraries (PyTorch's `nn.RNN`, `nn.GRU`, `nn.LSTM` among
them) use **two bias vectors per gate** (`bias_ih + bias_hh`), a
historical quirk inherited from Lua Torch. Our implementations use
**one bias per gate**, which is more common in modern practice (GPT-2,
LLaMA, etc.), and `num_weights` on the `LayerDef` reports that
single-bias count. Worth knowing when comparing a `num_weights` here
against a published parameter count for the "same" cell.

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
- `num_weights = 2 * (size * (input_size + size) + size)`.

## minGRU (`mingru.<size>`)

Even more minimal — gate and candidate depend only on the input:

    z = σ(W_z x + b_z)
    h_new = W_h x + b_h
    h = (1 - z) * h_prev + z * h_new

No hidden-to-hidden recurrence — the only time dependence is the
elementwise mixing. From https://arxiv.org/abs/2410.01201.

- JAX layout: stacked input projections `w_ih: (2*size, input_size)`.
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

## RG-LRU (`rglru.<blocks>`)

The Real-Gated Linear Recurrent Unit from Griffin / Hawk (De et al.
2024), as shipped in RecurrentGemma — a diagonal *real* linear
recurrence with input-dependent gating:

    r = σ(W_a x + b_a)                    # recurrence gate
    i = σ(W_x x + b_x)                    # input gate
    a = exp(-c · r · softplus(Λ)),  c = 8  # per-channel decay
    h_new = a * h + sqrt(1 - a^2) * (i * x)

Neither gate depends on `h`, and the recurrence is elementwise, so the
whole layer is a linear scan. `Λ` is a learned per-channel parameter
(initialized to −4, putting the decay near 0.93 at the gate midpoint);
the `sqrt(1 - a^2)` factor holds the state variance steady as `a`
varies.

- Dimension-preserving: `size == input_size`, like `conv` and the
  normalizations. `blocks` is the **only** metaparameter.
- The two gate projections are **block-diagonal** with `blocks` blocks
  of width `block_width = size / blocks` — a parameter-efficiency
  choice copied from RecurrentGemma (where `blocks` is the attention
  head count) so pretrained weights load directly.
- Weights: `lam: (size,)`, and per gate `w: (blocks, bw, bw)` +
  `b: (blocks, bw)`. Each block is Xavier-uniform for fan-in `bw`.
  `num_weights = size + 2 · blocks · bw · (bw + 1)`, i.e.
  `size · (2·size/blocks + 3)`. At `blocks = size` the gates collapse
  to per-channel scale+bias pairs (`5·size` weights in total); at
  `blocks = 1` they are two full `size × size` matrices plus biases.
- State: `h`, accumulated in **float32** regardless of layer dtype,
  plus a `first` flag. Position 0 is a reset — input multiplier 1
  instead of `sqrt(1 - a^2)`, matching the reference at `t = 0` (it
  matters for long-memory channels, where that factor is tiny).
- Validity: `blocks` a power of 2 and `blocks` divides `input_size`.
  RecurrentGemma's `rglru.10` is deliberately *not* search-valid; that
  model is loaded outside the search.
- `projects_input` is False — `x` also enters elementwise through
  `i * x`, so a preceding bare dense survives (Griffin's linear
  front-end feeds conv + RG-LRU).
- Matches `transformers`' `RecurrentGemmaRglru` numerically.

## MSR (`msr.<dim>.<heads>`)

Multi-Scale Retention (RetNet, Sun et al. 2023) — linear attention
with a **fixed** per-head scalar decay, no softmax:

    S_h = γ_h · S_h_prev + RoPE(k_h)^T v_h
    o_h = RoPE(q_h) · S_h

with `γ_h = 1 − 2^(−5−h)` and `θ_j = 10000^(−2j/dim)` shared across
heads. The decay ladder is what makes it "multi-scale": head 0 forgets
fastest, the last head slowest.

- No biases on Q/K/V, no output projection, no GroupNorm — stack
  `norm` or `dense.<size>` after it as needed.
- RoPE here uses the **interleaved** pair convention `(x_2j, x_2j+1)`,
  unlike `attn`'s split-half. The decay and rotation tables stay in
  fp32: by `t ≈ 100` both the slow heads' `γ^t` and the smallest
  rotation angles have fallen below bf16 precision.
- Training uses the parallel (quadratic) form — a `(T, T)` decay matrix
  applied to `Q_rot K_rot^T` — while `step` runs the recurrence, so
  inference state is `O(heads · dim^2)` and independent of length.
- Weights: one stacked `w_qkv: (3 · heads · dim, input_size)`.
  `size = heads · dim`, `num_weights = 3 · heads · dim · input_size`.
  `num_mults = num_weights + 2 · heads · dim^2` (outer-product write
  plus the `q · S` read).
- State: `(S: (heads, dim, dim), pos)`.
- Validity: `dim` a power of 2 and `≥ 2` (RoPE needs one pair);
  `heads` a power of 2.
- The fixed γ is the suspected weakness: it forgets on a schedule
  rather than on content. The DeltaNet entry in
  [`roadmap.md`](roadmap.md#architectures-to-add) exists specifically
  to test that hypothesis, by swapping the scalar decay for a learned
  per-token delta-rule update.

## Attention (`attn.<size>.<heads>.<window>`)

Local (sliding-window) **multi-query** attention with partial rotary
embeddings — the attention block of Griffin / RecurrentGemma, with
`head_dim = size / heads`:

    q = W_q x       # (heads · head_dim,) -- one vector per query head
    k = W_k x       # (head_dim,) -- ONE shared head (MQA)
    v = W_v x       # (head_dim,) -- ONE shared head
    rope(q_i), rope(k)                            # split-half, first
                                                  # half of head_dim
    s_i = softmax(q_i · k_past / sqrt(head_dim))  # over the last
                                                  # `window` positions,
                                                  # current included
    out = W_o concat_i(s_i @ v_past) + b_o

- **MQA**: a single shared K/V head, so per-position KV state is
  `2 · head_dim` no matter how many query heads there are.
- **Partial rotary**: RoPE is applied to the first
  `head_dim / 2` dims of `q` and `k` (module constant
  `_ROTARY_FRACTION = 0.5`, base 10000); the rest passes through
  unrotated. Pairs are split-half `(x_j, x_{j+rd/2})`, **not** the
  interleaved pairs `msr` uses. Angles are computed in float32, as is
  the softmax; scores are scaled by `head_dim^-0.5`.
- Banded causal mask: position `t` attends to `(t − window, t]`.
  Position 0 attends to itself only — the shift-pad initial vector
  plays the BOS role. **No leading padding is consumed**, so
  `length = 1`: history lives in the step-mode state (a rolling K/V
  buffer), like an RNN's hidden state, rather than in the model's
  `total_padding`. This is what distinguishes it from `suffix` and
  `conv`, which do consume positions.
- Weights: `w_q: (size, input_size)`, `w_k`/`w_v:
  (head_dim, input_size)`, `w_o: (size, size)`, `b_o: (size,)`. Bias
  on the output projection only (Q/K/V bias-free, per the reference).
  `num_weights = size · input_size + 2 · head_dim · input_size
  + size^2 + size`.
  `num_mults = num_weights + 2 · window · size`.
- State: rolling K/V buffers of the last `window − 1` positions (keys
  stored post-rotation) plus a position counter.
- Validity: `size`, `heads` and `window` all powers of 2,
  `heads` divides `size`, `head_dim ≥ 4` (so the rotary half still has
  at least one rotatable pair), and `window ≥ 2`. RecurrentGemma's own
  `attn.2560.10.2048` is therefore not search-valid; like `rglru.10`
  it is loaded outside the search.
- Matches `transformers`' `RecurrentGemmaAttention` numerically, so
  pretrained weights load directly.

## Conv (`conv.<kernel>`)

Depthwise causal 1-D convolution along the time axis — per-channel
kernels, no cross-channel mixing:

    y[c, t] = sum_{k=0..L-1} w[k, c] * x[c, t + L-1 - k] + b[c]

- Depthwise, so `size == input_size`.
- Weights: `w: (kernel, input_size)`, `b: (input_size,)`.
  `num_weights = input_size * (kernel + 1)`. Initialized normal, scaled
  by `1/sqrt(L)` so the per-channel sum of `L` weighted inputs starts
  at unit variance.
- **Consuming**, like `suffix`: `length = kernel`, and `forward` emits
  a valid (un-padded) convolution of `seq_len - kernel + 1` outputs.
  The `kernel - 1` leading transient positions are supplied by the
  model's `total_padding` machinery. Step-mode inference instead warms
  the layer up with `kernel - 1` padding iterations, then emits one
  output per call.
- State: `(kernel - 1, input_size)` — the recent-input buffer.
- Validity: `kernel` a power of 2 and `≥ 2`. `conv.1` would collapse to
  a per-channel scale + bias.
- `projects_input` is False (no cross-channel matmul), so a preceding
  bare dense doesn't collapse into it.
- `conv` and `suffix` are mutual neighbors at the same span — the
  learned per-channel mix of the last `L` positions vs. the hard
  concat of them.

## Suffix (`suffix.<length>`)

Sliding window: stacks the last `length` inputs into a single vector of
size `length * input_size`. No learned weights.

    out_t = concat(x_{t-length+1}, ..., x_t)

Useful as a lightweight n-gram feature before a downstream dense layer.
In training mode, `forward` emits `seq_len - length + 1` outputs (the
window requires `length - 1` padding tokens, supplied by the model's
`total_padding` machinery).

- State: `(length - 1, input_size)` — the recent history buffer.
- `length` attribute on the def is > 1; `LayerSeqDef.is_valid` rejects
  two adjacent layers that both consume positions, and more broadly two
  adjacent **temporal-wrap** layers — `msr`, `attn`, `conv`, `suffix`
  in any combination all fold a span of previous positions into the
  current one, so stacking two with nothing in between is redundant.
- `num_weights = 0`.

## Norm (`norm`)

Parameter-free L2 normalization along the feature axis, in Tikhonov
form:

    out = x / sqrt(||x||_2^2 + eps)      eps = 1e-5

The `sqrt(||x||^2 + eps)` denominator (rather than `max(||x||, eps)`)
keeps the gradient well-defined at `x = 0`. No learned weights.
Stateless.

- Same `size` as `input_size`; requires `input_size > 1`.
- Validity: a normalization can't open the **top-level** chain — it
  would be normalizing the raw input encoding — but it *may* open a
  split branch, which is the pre-norm residual block
  `split.add(rmsnorm-..., pass)`. You also can't have two adjacent
  normalizations (either flavor) or a normalization directly after a
  `suffix`.
- `num_weights = 0`.

## RMSNorm (`rmsnorm`)

The learned-scale sibling of `norm`: Gemma / RecurrentGemma RMSNorm.

    out = x * rsqrt(mean(x^2) + eps) * (1 + γ)      eps = 1e-6

`γ` is a per-channel vector initialized to **zero**, so the layer
starts as pure RMS normalization (scale 1) and weight decay
regularises it toward identity. The reduction and the `(1 + γ)`
multiply run in float32 and the result is cast back to the layer
dtype, matching `transformers`' `RecurrentGemmaRMSNorm` so pretrained
weights load and run faithfully.

- Dimension-preserving and stateless; requires `input_size > 1`.
- `num_weights = size` (the learned `γ`) — the only substantive
  difference from `norm`, which has none.
- `projects_input` is False (elementwise rescale, no matmul).
- Shares every adjacency rule with `norm` above, and the two are
  mutual neighbors: `norm ↔ rmsnorm` is a single mutation.

## Latent (`latent.<size>.<reps>`)

Depth-recurrent dense layer — iterative refinement of a latent state.
Inspired by https://arxiv.org/abs/2502.05171.

    e = W_i @ x + b
    s_0 = 0
    s_i = tanh(W_r @ normalize(s_{i-1}) + e)   for i = 1..reps
    out = s_reps

The input contribution `e` is re-injected at every iteration; this is
what makes the recurrence stable and gives the layer its "iterative
refinement" character. Raw `x` is fed to `W_i` without normalization;
prepend a `norm` layer in the spec to recover the pre-2026-05 behaviour.

- Normalization uses Tikhonov form `x / sqrt(||x||^2 + eps)`, which
  has well-defined gradients at `x = 0` (the `max(||x||, eps)` variant
  propagates NaNs through autodiff there).
- No time recurrence — each position is independent.
- JAX: uses `lax.scan` over `reps` with fixed length.
- `num_weights = size * input_size + size + size * size`.
- Constraints: `size >= 2`, `reps >= 2`, both powers of 2.
- Note: the paper initializes `s_0` randomly to encourage
  path-independence. We zero-initialize until we see that the layer is
  commonly picked by search and a randomized variant is worth the
  RNG-plumbing.

## Lrnn (`lrnn.<size>.<reps>`)

Latent-recurrent RNN: combines time recurrence with latent reasoning.
At each timestep, runs `reps` refinement iterations using the previous
timestep's hidden state as additional context:

    e_t = W_i @ x_t + W_h @ normalize(h_{t-1}) + b
    s_0 = 0
    s_i = tanh(W_r @ normalize(s_{i-1}) + e_t)
    h_t = s_reps

- Raw `x_t` is fed to `W_i` without normalization; prepend `norm` in
  the spec to recover the pre-2026-05 behaviour. The recurrent
  hidden state `h` is still L2-normalized (it has size ≥ 2 so the
  unit-norm projection isn't degenerate).
- JAX: outer `lax.scan` over time, nested `lax.scan` over reps.
- `num_weights = size * input_size + size + size * size + size * size`.
- Constraints: same as Latent.

## Lmgu (`lmgu.<size>.<reps>`)

Latent-recurrent mGRU — combines mGRU's gating with `lrnn`-style
depth-recurrent iteration of the gate and candidate.

Per token, the gate `f` and candidate `hc` are jointly refined for
`reps` iterations starting from zeros; after the loop, the standard
mGRU blend updates the time-recurrent state:

    h_n   = normalize(h)
    hc[0] = 0
    f[0]  = 0
    for i in range(reps):
        hc[i+1] = normalize(tanh(
            W_h_i x + W_h_h (f[i] * h_n) + W_h_s hc[i] + b_h))
        f[i+1]  = sigmoid(
            W_f_i x + W_f_h h_n + W_f_s hc[i+1] + b_f)
    h_new = (1 - f[reps]) * h + f[reps] * hc[reps]

- State per sample: `h: (size,)` — same as mgru. The `hc, f`
  iteration is local to a single token.
- `hc` is L2-normalized inside the reps loop (the architecturally
  significant direction-only-candidate constraint) and the
  time-recurrent `h` is normalized before each use as `h_n`. Raw `x`
  is fed to `W_*_i` without normalization; prepend a `norm` layer
  in the spec to recover the pre-2026-05 behaviour.
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

## Split (`split.<op>(<branch>, <branch>)`) — fork-and-merge

The fork-and-merge node: both branches see the same input and their
outputs are combined channel-wise by `op` ∈ {`add`, `cat`, `mul`}.
Branches are layer-lists or the keyword `pass` (identity). Splits are
the recursive successor to the legacy `skip` pseudo-layer, and host both
residual connections and gating:

- `split.add(F, pass)` / `split.cat(F, pass)` — residual, the skip
  analog: `x + F(x)` or `concat(F(x), x)`.
- `split.mul(value, gate)` — gating (GeGLU / SwiGLU / self-gating):
  `value(x) ⊙ gate(x)`, e.g. `split.mul(dense.X.gelu, dense.X)` or the
  self-gate `split.mul(pass, dense.X.gelu)`.

No learned weights of its own (`num_weights` = sum over branches);
the runtime is `SplitJax`. Example:

    bytes|dense.32.gelu-split.add(dense.32.gelu-dense.32.gelu, pass)-dense.64.tanh

See [`split.md`](split.md) for the full design: merge semantics, the
residual/gate families, canonical form + validity, and the search
mutations.

### Skip (`skip.<X>.<add|cat>`) — retired residual

`skip.X.add` / `skip.X.cat` was the original residual marker — a
pseudo-layer spanning `X` layers, merged at the end (same `.add` /
`.cat` size semantics Split inherits). The syntax was retired in
2026-07 after the result DB was migrated to split form; a `skip.*`
spec now fails to parse like any unknown layer. See
[`skip.md`](skip.md) for the historical design.

## Neighbor relations (search)

The search walks the architecture space by generating "neighbors" of a
model. Each layer type declares its own per-layer neighbors via
`LayerDef.neighbors()`; the chain- and tree-level mutations (append /
remove a layer, insert / remove `suffix`·`norm`, wrap / unwrap residual
and gate Splits) live in `model2.py` — see [`split.md`](split.md).
Per-layer summary:

- **dense ↔ rnn** — with the activation preserved. A *bare* (
  activation-less) dense has no cross-type swap; it stays a dense.
- **dense activation cycle** — `dense.X` ↔ `dense.X.tanh` ↔
  `dense.X.gelu` at the same size, the bare form included (it's the
  tied-codec adapter and the gate/linear path, filtered by validity
  everywhere else). `rnn` cycles between `tanh` and `gelu` only; it has
  no bare form. A retired `relu` layer still proposes the survivors, so
  existing relu lineages migrate instead of dying.
- **dense.X.tanh ↔ latent.X.2** — promotes a dense-tanh to its
  depth-recurrent twin (needs `X > 1`).
- **rnn.X.tanh ↔ lrnn.X.2** — likewise for recurrent.
- **rnn ↔ gru, mgru, mingru** — activation is dropped when moving to a
  gated cell.
- **gru, mgru, mingru, lstm** are mutual neighbors, with one exception:
  **lstm ↮ mingru** (mingru has no hidden-to-hidden state, so it's
  structurally too different).
- **lstm, matlstm, slstm, mullstm** are all-to-all mutual neighbors
  within the LSTM family (matlstm is skipped at size 1); `lstm`
  additionally keeps its swaps with `gru` and `mgru`.
- **latent.X.Y ↔ lrnn.X.Y ↔ lmgu.X.Y** — swaps within the
  depth-recurrent family at matching size/reps (`latent ↮ lmgu`
  directly). At `reps == 2` each collapses to its non-iterating cousin:
  `latent.X.2 ↔ dense.X.tanh`, `lrnn.X.2 ↔ rnn.X.tanh`,
  `lmgu.X.2 ↔ mgru.X`.
- **norm ↔ rmsnorm** — the parameter-free L2 norm and the learned-scale
  RMS norm. Neither has any other metaparameter, so this is their only
  mutation.
- **suffix.L ↔ conv.L** — same time-axis span; the hard concat vs. the
  learned per-channel mix.
- **suffix.L ↔ attn.<input_size>.1.L** — the same span wrapped softly
  by a single attention head instead of concatenated. The reverse edge
  is guarded to the exact image of the forward one (`heads == 1` and
  `size == input_size`) so the relation stays symmetric.
- **conv.K ↔ attn.<input_size>.1.K** — the third side of the windowed
  triangle: unlike the two edges above, both of these are
  *length-preserving*, a learned static FIR against a
  content-dependent lookup over the same K positions. The span is
  carried across exactly (`kernel ↔ window`; both floors are 2, so
  every valid kernel is a valid window and back), and the reverse edge
  reuses the suffix edge's exact-image guard, which makes the round
  trip the identity. Without this edge conv reaches attn only through
  `suffix`, a two-hop path that downsamples the time axis midway.
- **msr.D.H ↔ attn.(H·D).H.16** — the attention-family swap, linear
  attention against softmax attention. The edge matches the two specs
  by **output width**, not field for field: `msr`'s first field is the
  *per-head* `dim` (output width `H·D`) while `attn`'s is the total
  `size` (head_dim `size/H`), so `msr.8.4` (width 32) swaps to
  `attn.32.4.16` (width 32, head_dim 8) and back. Only at `H = 1` do
  the two spellings coincide. The forward edge is guarded to `D ≥ 4`
  and the reverse to `head_dim ≥ 4`, since `attn` needs a rotary pair;
  below that there is no equal-shape twin and the swap is skipped.
  `attn` needs a window that `msr` doesn't have, so the swap lands on
  a modest 16 (`_ATTN_SWAP_WINDOW`) and lets the window mutate from
  there — the one field the round trip doesn't preserve.
- **msr.X.1 ↔ mgru.X** — single-head retention has the same vector
  state width as mgru (skipped at `X = 1`, where msr has no RoPE pair).
- **rglru.1 ↔ mgru.X, mingru.X** — the RG-LRU is size-preserving, so
  the swap only fires when the gated cell preserves size too
  (`size == input_size`), and it lands on the canonical single-block
  form.
- Size `S`, reps `R`, `heads`, `window`, `kernel` and `blocks`
  mutations: `×2`, `÷2` (keeping powers of 2 and minimum sizes;
  `is_valid` filters the boundary cases, e.g. `attn`'s `head_dim ≥ 4`
  or an `rglru` block count that no longer divides the width).

See `layer.py:LayerDef.neighbors` for the per-layer rules and
[`split.md`](split.md) (`model2.py`) for the chain- and Split-level
mutations.
