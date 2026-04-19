# CLAUDE.md

## Project overview

Texmo is a project for experimenting with small language models, with a
focus on older recurrent architectures. It includes implementations of
various building blocks and a search over meta-parameters and model
architectures.

## Documentation

Detailed docs live in [`docs/`](docs/):

- [`architecture.md`](docs/architecture.md) — model pipeline, layer
  abstractions, input encoding, and how the two backends fit together.
- [`layers.md`](docs/layers.md) — the available layer types, their
  equations, parameter counts, and neighbor relations.
- [`skip.md`](docs/skip.md) — design reference for residual
  connections (`skip.X.add` / `skip.X.cat`).
- [`timing.md`](docs/timing.md) — training-step time prediction
  model (features, fit, and use).
- [`backends.md`](docs/backends.md) — PyTorch vs JAX trade-offs and why
  we support both.
- [`search.md`](docs/search.md) — distributed architecture search:
  server, client, result DB, neighbor generation.
- [`decay_and_checkpoints.md`](docs/decay_and_checkpoints.md) — design
  notes on LR decay interacting with intermediate checkpoints.
- [`roadmap.md`](docs/roadmap.md) — planned features.

## Conventions

### Tests

- Tests live next to the code, named `*_test.py`.
- Use **pytest** (plain functions, not unittest classes).
- No fixtures; use local helper functions like `_make_module()`.
- Run with: `uv run pytest path/to/test_file.py`.

### Layer pattern

Each layer module contains three classes:

1. `{Name}Def(LayerDef)` — backend-agnostic descriptor (spec parsing,
   `num_weights`, `is_valid`, `neighbors`).
2. `{Name}Module(LayerModule)` — PyTorch `nn.Module` implementation.
3. `{Name}Jax(LayerJax)` — JAX functional implementation (weights
   passed explicitly).

`{Name}Def` exposes both `build_module()` and `build_jax(dtype)`. New
layers are registered in `model.py:_build_layer_def()`.

### Tools

- Package manager: `uv`.
- Run commands via: `uv run <command>`.
- Config defaults: `config.py`.
