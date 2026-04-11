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

The output is always a Dense layer producing logits over the token vocabulary.


## Key abstractions

### ModelDef / Model (`model.py`)

**ModelDef** parses a spec string and builds a lightweight descriptor (no
weights). It holds:

- An input def (`InputBytesDef` or `InputBitsDef`)
- A list of `LayerDef`s for hidden layers
- An output `DenseDef` projecting to the token vocabulary
- A `precision` (Precision enum; `precision.py`)

**Model** is the runnable `nn.Module`. It supports two modes:

- `forward(batch)` — full-sequence forward for training. The input is shifted
  right and padded at position 0 with the input module's `initial_vector()`
  (uniform distribution over tokens).
- `step(states, token)` — single-timestep inference. `states[0]` is the input
  module's state; `states[1:]` are hidden layer states.

Factory: `build_model_def(spec, precision)` is cached by `(spec, precision)`
tuple.

### LayerDef / LayerModule (`layer.py`)

Base classes for hidden layers. Each layer type provides:

- **LayerDef** — config/descriptor. Has `input_size`, `size`, `num_weights`,
  `is_valid()`, `neighbors()`, and `build_module()`. `length` attribute
  (default 1) is how many previous timesteps the layer depends on
  (suffix layers are "length > 1").
- **LayerModule** — `nn.Module` with `step(state, input)` for single-timestep
  and `forward(inputs)` for batched sequence processing. `init_state(device,
  dtype)` creates a matching-dtype recurrent state (or None for stateless).

Implemented layer types (all in `texmo/layers/`):

- `dense.py` — dense feed-forward (DenseDef)
- `rnn.py` — Elman RNN (via nn.RNN for tanh/relu, custom for gelu)
- `gru.py` — standard GRU (nn.GRU), mGRU (single-gate variant), minGRU
  (input-only gates)
- `lstm.py` — standard LSTM (nn.LSTM)
- `suffix.py` — sliding window (stacks last N inputs)
- `norm.py` — L2 normalization
- `latent.py` — depth-recurrent dense (Latent) and RNN (Lrnn) layers
  from https://arxiv.org/abs/2502.05171

### Manager (`manager.py`)

Training and inference manager. Handles:

- Building the model and optimizer from a `Configuration`
- Training loop with gradient clipping and LR scheduling
- Evaluation on test data (in fp32 regardless of training dtype)
- Loss conversion from bits/token to bits/byte (b/B)
- Text generation via `continue_prefix()`

### Precision (`precision.py`)

Enum with `.dtype` property (torch.dtype) and `.neighbors` (search).
FP64 is supported but not in the default set (isn't supported on MPS).


## Input modules

Input modules convert integer token indices to float vectors. They all provide:

- `init_state()` — initial state (None if stateless)
- `step(state, token, device)` -> `(new_state, vector)` — single token
- `initial_vector(device)` — uninformative input for the first position
- `forward(tokens, padding=0)` — batched encoding with optional padding

### InputBytes (`layers/input_bytes.py`)

Spec: `bytes` (or empty). Encodes each byte (0-255) as a 256-dimensional
one-hot vector. Stateless. `ntokens = 256`, `output_size = 256`.

### InputBits (`layers/input_bits.py`)

Encodes bytes split into sub-byte chunks. A byte is split into `8/nbits`
chunks, each chunk being an `nbits`-bit value.

Spec format: `bits.<nbits>[.oh][+bp]`

- **nbits**: 1, 2, 4, or 8. Number of bits per chunk.
- **.oh**: one-hot encoding (vector of size `2^nbits`). Without `.oh`, each bit
  becomes one float value (vector of size `nbits`).
- **+bp**: append binary position encoding — the chunk's position within the
  current byte, encoded as individual bits (3 bits for nbits=1, 2 for nbits=2,
  1 for nbits=4).

Examples:

| Spec | ntokens | output_size | Description |
|------|---------|-------------|-------------|
| `bits.1` | 2 | 1 | Single bit as one float |
| `bits.1+bp` | 2 | 4 | 1 bit value + 3 position bits |
| `bits.2.oh` | 4 | 4 | 2-bit chunk as 4-dim one-hot |
| `bits.4.oh+bp` | 16 | 17 | 4-bit one-hot + 1 position bit |
| `bits.8` | 256 | 8 | Full byte as 8 raw bits |
| `bytes` | 256 | 256 | Alias for `bits.8.oh` |

For **non-one-hot** encodings, `initial_vector()` uses 0.5 for value bits (max
entropy). For **one-hot**, it uses `1/ntokens` (uniform distribution). Position
bits in the initial vector use the wrap-around position (last position in the
byte cycle).

### Binary output for bits.1

When `ntokens <= 2` (bits.1), the output Dense layer produces a single logit
instead of 2. This logit is padded with 0 to form `[x, 0]`, making
`softmax([x, 0]) = [sigmoid(-x), sigmoid(x)]` — the single value acts as
log-odds of class 1 vs class 0.
