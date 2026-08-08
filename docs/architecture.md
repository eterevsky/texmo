# Architecture

Texmo models are character-level language models that predict the next token
from a sequence of preceding tokens. A token can be a full byte or a sub-byte
chunk of bits, depending on the input encoding.

## Overview

A model is defined by a **spec string** like `bytes|dense.128.gelu` or
`bits.2.oh+bp|dense.64.tanh-gru.64`. The spec has two parts separated by `|`:

    <input_spec>|<layer1>-<layer2>-...

The pipeline is:

    tokens --> [Input Module] --> [Layer 1] --> ... --> [Layer N] --> [Output Dense] --> logits

The output is always a Dense layer projecting to the token vocabulary.

A layer can also be a `split.op(branch, branch)` **fork-and-merge** node
(residual connections and gating), so the chain is really a layer-DAG,
not a straight line — e.g. `bytes|split.add(dense.64.gelu, pass)`. See
[`split.md`](split.md).

See [`layers.md`](layers.md) for the full list of available layers.

## Backends

**JAX is the only backend**, end to end: layers, full model, and
training loop. The `--backend` flag and `create_manager(backend, …)`
survive as the seam for a future second one. See
[`backends.md`](backends.md) for the trade-offs that led here and the
history of the removed torch backend.

## Key abstractions

### Model2Def (`model2.py`) — the model representation

**Model2Def** is a lightweight descriptor (no weights): a recursive
**layer-DAG** where the hidden chain is a `LayerSeqDef` (a sequence of
`LayerDef`s) and a `SplitDef` hosts two `LayerSeqDef` branches for
fork-and-merge. This subsumes the retired flat representation's
`skip.D.op` pseudo-layers (residuals are `split.op(span, pass)`; the
skip syntax itself is retired) and adds gating (`split.mul`). See
[`split.md`](split.md). It holds:

- A codec (`OneHotCodecDef` or `EmbeddingCodecDef`) owning both model
  ends — input encoding and logit head (see [`io.md`](io.md))
- The `LayerSeqDef` hidden chain
- A `precision` (Precision enum; `precision.py`)

Key surface: `spec`, `num_weights`, `num_mults`, `num_layers`,
`is_valid`, `neighbors`, equality/hashing. Constructed only via
`spec_parser.parse_model2(spec, precision)`, which owns every
string→tree decision (the `|`-split, codec dispatch, and the
recursive layer grammar). `Model2Def.__init__` just stores the
already-parsed pieces.

`build_jax()` returns a **Model2Jax**: functional JAX implementations
with weights returned separately by `init_weights(rng)` and passed
explicitly to each call. It supports two modes:

- **Full-sequence forward** for training. The input is shifted right and
  padded at position 0 with the codec's initial vector (uniform
  distribution over tokens).
- **Single-timestep `step`** for inference. States are a list aligned
  with layers: `states[0]` is the input state, `states[1]` the layer-
  sequence states.

### LayerDef / LayerJax (`layer.py`, `layer_jax.py`)

Base classes for hidden layers.

- **LayerDef** — config/descriptor. Has `input_size`, `size`, `num_weights`,
  `is_valid()`, `neighbors()`. Factory method: `build_jax(dtype)` → JAX
  `LayerJax`. `length` attribute
  (default 1) is how many previous timesteps the layer depends on
  (suffix-like layers have `length > 1`).
- **LayerJax** — a lightweight wrapper (does **not** own weights). Weights
  are passed explicitly to every call as a pytree of `jax.Array`s. The
  interface: `init_weights(rng)` returns the weight pytree, `init_state()`
  returns an initial recurrent state, `step(weights, state, x)` and
  `forward(weights, inputs)` do the actual computation. Weight init uses
  Xavier uniform for matrices and zeros for biases (see
  `layer_jax.xavier_uniform`).

### Manager / ManagerJax (`manager.py`, `manager_jax.py`)

`Manager` is the base class defining the backend-agnostic interface:
`__init__(conf, system, dataset, …)`, `train(steps, time_limit)`,
`eval()`, `train_and_eval(…)`, `continue_prefix(…)`. `ManagerJax`
handles the backend-specific bits: building the model and optimizer,
training loop mechanics, eval, and text generation.

Use `create_manager(backend, **kwargs)` to build one — `'jax'` is the
only accepted value. The shared `train()` and `train_and_eval()` loops
live on `Manager`; the backend supplies `_get_batch()`,
`train_step(batch)`, and `eval()`.

### Precision (`precision.py`)

Enum with a `.jax_dtype` property (jnp.dtype) and `.neighbors` (used in
the search). FP64 is supported but not in the default set (Metal
doesn't support it on Mac).

## Input modules

Input modules convert integer token indices to float vectors. They live
in `layers/input_bits.py` and `layers/input_bytes.py`.

Spec format: `bits.<nbits>[.oh][+bp]`, or the alias `bytes`.

- **nbits**: 1, 2, 4, or 8. Number of bits per chunk.
- **.oh**: one-hot encoding (vector of size `2^nbits`). Without `.oh`,
  each bit becomes one float value.
- **+bp**: append binary position encoding — the chunk's position within
  the current byte, encoded as individual bits (3 bits for nbits=1, 2 for
  nbits=2, 1 for nbits=4).

Examples:

| Spec | ntokens | output_size | Description |
|------|---------|-------------|-------------|
| `bits.1` | 2 | 1 | Single bit as one float |
| `bits.1+bp` | 2 | 4 | 1 bit value + 3 position bits |
| `bits.2.oh` | 4 | 4 | 2-bit chunk as 4-dim one-hot |
| `bits.4.oh+bp` | 16 | 17 | 4-bit one-hot + 1 position bit |
| `bits.8` | 256 | 8 | Full byte as 8 raw bits |
| `bytes` | 256 | 256 | Alias for `bits.8.oh` |

For **non-one-hot** encodings, the initial padding vector uses 0.5 for
value bits (max entropy). For **one-hot**, it uses `1/ntokens` (uniform
distribution). Position bits in the initial vector use the wrap-around
position (last position in the byte cycle).

## Binary output for bits.1

When `ntokens <= 2` (bits.1), the output Dense layer produces a single
logit instead of 2. This logit is padded with 0 to form `[x, 0]`, making
`softmax([x, 0]) = [sigmoid(-x), sigmoid(x)]` — the single value acts as
log-odds of class 1 vs class 0.
