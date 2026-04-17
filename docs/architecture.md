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

See [`layers.md`](layers.md) for the full list of available layers.

## Two backends

Every layer has both a PyTorch implementation and a JAX implementation,
selectable via `--backend torch|jax`. Layer definitions (specs, weight
counts, neighbor rules) are shared. See [`backends.md`](backends.md) for
the trade-offs between the two.

## Key abstractions

### ModelDef / Model / ModelJax (`model.py`, `model_jax.py`)

**ModelDef** parses a spec string and builds a lightweight descriptor (no
weights). It holds:

- An input def (`InputBytesDef` or `InputBitsDef`)
- A list of `LayerDef`s for hidden layers
- An output `DenseDef` projecting to the token vocabulary
- A `precision` (Precision enum; `precision.py`)

It has two factory methods:
- `build_model()` — returns a PyTorch `Model` (`nn.Module`) that owns its
  weights.
- `build_jax()` — returns a `ModelJax` with the functional JAX layer
  implementations; weights are returned separately by `init_weights(rng)`
  and passed explicitly to each call.

Both support two modes:

- **Full-sequence forward** for training. The input is shifted right and
  padded at position 0 with the input module's initial vector (uniform
  distribution over tokens).
- **Single-timestep `step`** for inference. States are a list aligned with
  layers: `states[0]` is the input module's state, `states[1:]` are hidden
  layer states.

Factory: `build_model_def(spec, precision)` is cached by `(spec, precision)`.

### LayerDef / LayerModule / LayerJax (`layer.py`, `layer_jax.py`)

Base classes for hidden layers.

- **LayerDef** — config/descriptor. Has `input_size`, `size`, `num_weights`,
  `is_valid()`, `neighbors()`. Factory methods: `build_module()` → PyTorch
  `LayerModule`, `build_jax(dtype)` → JAX `LayerJax`. `length` attribute
  (default 1) is how many previous timesteps the layer depends on
  (suffix-like layers have `length > 1`).
- **LayerModule** — a PyTorch `nn.Module` that owns its weights.
  `step(state, input)` for single-timestep, `forward(inputs)` for batched
  sequences. `init_state(device, dtype)` creates a matching-dtype
  recurrent state (or `None` for stateless).
- **LayerJax** — a lightweight wrapper (does **not** own weights). Weights
  are passed explicitly to every call as a pytree of `jax.Array`s. The
  interface: `init_weights(rng)` returns the weight pytree, `init_state()`
  returns an initial recurrent state, `step(weights, state, x)` and
  `forward(weights, inputs)` do the actual computation. Weight init uses
  Xavier uniform for matrices and zeros for biases (see
  `layer_jax.xavier_uniform`).

### Manager / ManagerTorch / ManagerJax (`manager.py`, `manager_torch.py`, `manager_jax.py`)

`Manager` is the base class defining the backend-agnostic interface:
`__init__(conf, system, dataset, …)`, `train(steps, time_limit)`,
`eval()`, `train_and_eval(…)`, `continue_prefix(…)`. Subclasses
`ManagerTorch` and `ManagerJax` handle the backend-specific bits:
building the model and optimizer, training loop mechanics, eval, and
text generation.

Use `create_manager(backend, **kwargs)` to build one for the selected
backend. The shared `train()` and `train_and_eval()` loops live on
`Manager`; subclasses supply `_get_batch()`, `train_step(batch)`, and
`eval()`.

### Precision (`precision.py`)

Enum with `.dtype` property (torch.dtype), `.jax_dtype` (jnp.dtype), and
`.neighbors` (used in the search). FP64 is supported but not in the
default set (MPS doesn't support it on Mac).

## Input modules

Input modules convert integer token indices to float vectors. Both Torch
and JAX implementations live in `layers/input_bits.py` and `layers/input_bytes.py`.

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
