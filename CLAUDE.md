# CLAUDE.md

## Project overview

Texmo is a project for experimenting with small language models, with a
focus on older recurrent architectures. It includes implementations of
various building blocks and a search over meta-parameters and model
architectures.

## Documentation

Detailed docs live in [`docs/`](docs/):

- [`architecture.md`](docs/architecture.md) — model pipeline, layer
  abstractions, input encoding, and how a spec becomes a runnable
  model.
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
  codec (`EmbeddingCodec`): decision log, rejected alternatives,
  mutation consequences, Gemma fidelity mapping. Implemented; kept
  for the reasoning, including the reversals.
- [`tokens.md`](docs/tokens.md) — tokenization machinery: the tokenset
  JSON format, the tokenizer implementations, text preprocessing, the
  DP tokenization algorithm, and the Rust generators. The tokenset
  *families* live in [`io.md`](docs/io.md).
- [`timing.md`](docs/timing.md) — training-step time prediction
  model (features, fit, and use).
- [`loss_prediction.md`](docs/loss_prediction.md) — the
  structure-mirroring (tree) loss predictor: architecture, measured
  performance, ablation history, use in search, refit/persistence.
- [`loss_rnn_experiments.md`](docs/loss_rnn_experiments.md) — flat-
  RNN-era sweep results (best configs, null results, open ideas).
- [`backends.md`](docs/backends.md) — PyTorch vs JAX trade-offs and why
  JAX is the training runtime (the torch backend was removed 2026-08).
- [`search.md`](docs/search.md) — distributed architecture search:
  server, client, result DB, neighbor generation.
- [`threads.md`](docs/threads.md) — the server's threading model:
  which thread owns what, the queue protocol between them, and the
  read/write DB split.
- [`best_models.md`](docs/best_models.md) — dated snapshot of the
  Pareto frontier by weight count, regenerated from the result DB by
  `scripts/make_best_models.py`.
- [`decay_and_checkpoints.md`](docs/decay_and_checkpoints.md) — design
  notes on LR decay interacting with intermediate checkpoints.
- [`roadmap.md`](docs/roadmap.md) — planned features; open work only.
- [`done.md`](docs/done.md) — dated log of completed roadmap items,
  newest first.
- [`findings.md`](docs/findings.md) — important empirical results
  that keep informing decisions; the knowledge, not the work log.
- [`links.md`](docs/links.md) — bare list of reference papers and
  repos.

`docs/bench.txt` is a hand-maintained scratch table of per-machine
training benchmarks, not prose.

## Conventions

### Tests

- Tests live next to the code, named `*_test.py`.
- Use **pytest** (plain functions, not unittest classes).
- No fixtures; use local helper functions like `_make_module()`.
- Run with: `uv run pytest path/to/test_file.py`.

### Layer pattern

Each layer module contains two classes:

1. `{Name}Def(LayerDef)` — backend-agnostic descriptor (spec parsing,
   `num_weights`, `is_valid`, `neighbors`).
2. `{Name}Jax(LayerJax)` — JAX functional implementation (weights
   passed explicitly).

`{Name}Def` exposes `build_jax(dtype)`. New layers are registered in
`spec_parser.py:_build_layer_def()`.

### Style

- No string type annotations: annotate with the real class name and
  import it for real. Reserve `TYPE_CHECKING` (and the accompanying
  quoted name) for genuine import cycles only — check that the cycle
  actually exists before reaching for it.
- Imports live in the file preamble, never inline in functions.

### Scratch space

Temporary scripts and working data (analysis probes, fetched logs,
one-off experiments) go in `scratch/` at the repo root (untracked) —
not in the system temp directory, so they survive the session and
stay inspectable. Scripts worth keeping graduate to `scripts/` (with
its `sys.path` shim).

### Tools

- Package manager: `uv`.
- Run commands via: `uv run <command>`.
- Config defaults: `config.py`.
- When writing JSON files from Python, use `texmo.pjson`
  (`save_json` / `pprint`) — adaptive pretty-printing (short values
  inline, long ones expanded) matching the repo's JSON style. Don't
  hand-roll `json.dumps` formatting.
