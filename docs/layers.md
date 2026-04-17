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
