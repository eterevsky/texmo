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
  fork-and-merge layer (residual + gate families, validity, search
  mutations). The current `Model2` representation.
- [`skip.md`](docs/skip.md) — **retired** residual connections
  (`skip.X.add` / `skip.X.cat`), superseded by `split` and no longer
  parsed; kept for the merge semantics Split inherits.
- [`io.md`](docs/io.md) — how models turn token ids into logits: the
  model contract and the three IO kinds (one-hot bit chunks, embedded
  bit chunks, tokenized), what/how/why for each.
- [`tied_io.md`](docs/tied_io.md) — design record for the tied
  InputOutput migration: decision log, rejected alternatives, mutation
  consequences, Gemma fidelity mapping. Agreed, not yet implemented.
- [`timing.md`](docs/timing.md) — training-step time prediction
  model (features, fit, and use).
- [`loss_prediction.md`](docs/loss_prediction.md) — the
  structure-mirroring (tree) loss predictor: architecture, measured
  performance, ablation history, use in search, refit/persistence.
- [`loss_rnn_experiments.md`](docs/loss_rnn_experiments.md) — flat-
  RNN-era sweep results (best configs, null results, open ideas).
- [`backends.md`](docs/backends.md) — PyTorch vs JAX trade-offs and why
  JAX is the training runtime (the torch full-model backend is parked).
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
layers are registered in `spec_parser.py:_build_layer_def()`.

### Style

- No string type annotations: annotate with the real class name and
  import it for real. Reserve `TYPE_CHECKING` (and the accompanying
  quoted name) for genuine import cycles only — check that the cycle
  actually exists before reaching for it.
- Imports live in the file preamble, never inline in functions.

### Tools

- Package manager: `uv`.
- Run commands via: `uv run <command>`.
- Config defaults: `config.py`.
- When writing JSON files from Python, use `texmo.pjson`
  (`save_json` / `pprint`) — adaptive pretty-printing (short values
  inline, long ones expanded) matching the repo's JSON style. Don't
  hand-roll `json.dumps` formatting.
