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
- [`split.md`](docs/split.md) — design reference for the `split.op(...)`
  fork-and-merge layer (residual + gate families, validity, skip→split
  translation, search mutations). The current `Model2` representation.
- [`skip.md`](docs/skip.md) — **legacy** residual connections
  (`skip.X.add` / `skip.X.cat`), superseded by `split`; kept for the
  merge semantics Split inherits.
- [`timing.md`](docs/timing.md) — training-step time prediction
  model (features, fit, and use).
- [`loss_prediction.md`](docs/loss_prediction.md) — RNN loss
  predictor (architecture, training, current accuracy, use in search).
- [`loss_rnn_experiments.md`](docs/loss_rnn_experiments.md) — RNN
  loss-predictor sweep results (best config, null results, open ideas).
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
