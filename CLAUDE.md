# CLAUDE.md

## Project overview

Texmo is a project for experimenting with language models, with a focus on
older recurrent architectures. It includes implementations of various building
blocks and a search over meta-parameters and model architectures.

## Architecture

See `docs/architecture.md` for a detailed description of the model pipeline,
layer abstractions, and input encoding schemes.

## Current goal

Migrating model implementations from JAX to PyTorch. The JAX code remains as
reference.

### Already migrated

- `texmo/layer_torch.py` — `LayerDef`/`LayerModule` base classes
- `texmo/model_torch.py` — `ModelDef`/`Model`
- `texmo/manager_torch.py` — `ManagerTorch` training/inference manager
- `texmo/layers/dense_torch.py` — `DenseDef`/`DenseModule`
- `texmo/layers/input_bytes_torch.py` — `InputBytesDef`/`InputBytesModule`
- `texmo/layers/input_bits_torch.py` — `InputBitsDef`/`InputBitsModule`
- `texmo/layers/suffix_torch.py` — `SuffixDef`/`SuffixModule`

### Not yet migrated

JAX-only layers in `texmo/layers/`: GRU, LSTM, Attn, Input,
RecurrentBase, S4, LatentAttention.

## Conventions

### Tests

- Tests live next to the code, named `*_test.py` (e.g. `dense_torch_test.py`).
- Use **pytest** (plain functions, not unittest classes).
- No fixtures; use local helper functions like `_make_module()`.
- Run with: `uv run pytest path/to/test_file.py`

### PyTorch layer pattern

A new layer consists of:
1. `{Name}Def(LayerDef)` — layer definition / config (dataclass-like).
2. `{Name}Module(LayerModule)` — the `nn.Module` with `step(state, input)` and
   `forward(inputs)`.
3. Register in `model_torch.py` `_build_layer_def()`.

### Tools

- Package manager: `uv`
- Run commands via: `uv run <command>`
- Config defaults: `config.py`