# Project structure

What lives where in the repository, and what does not belong in each
place. Features described as *(planned)* are recorded here so new
files land in the right place; the **In flux** section at the end
lists the parts of the tree that have not caught up with this layout
yet.

## The two heads

The repo has two command-line entry points with a strict dependency
direction between them:

- **`texmo.py`** — everything Texmo-native: architecture search,
  training, evaluation, generation, chat, servers. Each command is
  implemented in `texmo/cli/`.
- **`rig.py`** *(planned)* — operations that involve third-party
  models (llama.cpp / HF LLMs) on top of Texmo models: the
  scripted-examiner chat eval, LLM corpus processing, dialog
  recording, dataset preparation. `./rig.sh` is the shortcut for
  `uv run --group rig rig.py …` — the rig's extra dependencies
  (`pyarrow`, …) form the `rig` dependency group in `pyproject.toml`,
  which `uv sync`/`uv run` skip by default so search worker machines
  never install them.

`rig` imports `texmo` (it needs to run Texmo models); `texmo` never
imports `rig`. Texmo-only benchmarking belongs to `texmo.py bench`,
not the rig.

## Code

- **`texmo/`** — one Python package: the engine (layers, models,
  tokenizers, training), the architecture search (server, client,
  result DB access, predictors) and everything related to running and
  evaluating Texmo models. It does **not** contain operations with
  third-party models that run on other engines — those go in `rig/`.
- **`rig/`** *(planned)* — the rig's package. Commands:
  - `chat-eval` — the scripted-examiner eval (generate / judge /
    phrases; see `docs/findings.md` for what it measures).
  - `dialogs` — record dialogs between two LLMs.
  - `process` — run a dataset through an LLM, generically: the input
    is split into chunks at a separator (`\n\n` for a turn,
    `\n\n\n` for a dialog), every chunk goes through one LLM with one
    prompt, and the outputs are logged resumably, one record per
    chunk. The whole setup is a JSON file — which LLM
    (`llms/<name>`), the prompt (inline or a `.txt` next to it), the
    separator, output format, parallelism — so a pass is reproducible
    from its setup file and a new kind of pass is a new setup, not
    new code.
  - `render` — write dataset variants out of a process log (LLM-free
    substitutions: simplification levels, speech acts, case style).
  - `soda-txt`, `eval-seeds` — dataset preparation.
- **`src/`**, **`Cargo.toml`** — the Rust code: today the tokenset
  generators (`docs/tokens.md`); anything else that benefits from
  being written in Rust goes here too.
- **`benchmarks/`** — training benchmarks: `suite.json` (a fixed
  suite generated from the result DB), per-machine result files, and
  standalone reference benchmarks (`bench_jax.py`, `bench_torch.py` —
  hand-written GRU/reference models outside the Texmo layer stack).
- **`web/`** *(planned)* — the GitHub Pages tree: a static
  in-browser chat page (self-contained HTML, inference in JS),
  published together with `models/` and `tokens/`.

## Models and tokensets

- **`models/`** — only Texmo-format JSON models (`{"spec",
  "precision", "weights"}`), ready to run on the engine. No GGUFs, no
  checkpoints, no third-party formats. *(Planned)* a `"format"` field
  in the manifest names what the model expects — named formats, not
  parametrized ones (hardcoding a format is easier than customizing
  it):
  - `"raw"` — a plain text model; continues whatever it is given.
  - `"chat-simple"` — `User: …` / `Bot: …` turns separated by blank
    lines, the model replying as `Bot` (the format of the SODA-derived
    corpora, implemented by `chat`, `chat-server` and the web page). A
    new dialog convention gets a new name, not options.
- **`tokens/`** — the tokensets models refer to. Corpus-agnostic sets
  live at the top level (`tokens.32.hexbpe.json`); a set trained on a
  particular corpus lives in a subdirectory named after it and is
  referred to with the directory in the model spec:
  `tokens/s5/tokens.64.hexbpe.json` ↔ a spec starting
  `s5/tokens.64.hexbpe.oh|…` *(planned; the spec parser does not
  accept the `dir/` prefix yet)*. The `asoif/` and `pride/`
  subdirectories are the same idea for earlier corpora.
- **`llms/`** — third-party models, one directory per model
  (`llms/gemma-4/`, `llms/qwen3-8/`, `llms/recurrentgemma-2b/`).
  Weight files are gitignored; each directory carries a committed
  `meta.json` *(planned)* recording source repo/URL, file name,
  quantization, sha256, size, and what the model is used for, so
  commands can say `--examiner=llms/gemma-4` and a fresh machine
  knows what to download. Server settings (context size, slots,
  thinking flags) belong to the command that uses a model, not to the
  model. ELIZA is an honorary LLM: the vendored DOCTOR port lives at
  `llms/eliza/` with its code committed — the code *is* the model —
  and is addressed like the others (`--student=llms/eliza`).

`models/` and `tokens/` are meant to be served as-is via GitHub
Pages: a web client picks a model, loads its JSON, reads the tokenset
name out of the spec, loads that too, and runs. Served models must be
committed; today only `recurrentgemma-2b.json` is tracked and the
locally trained models are not — which subset to publish is a
curation decision.

## Data and outputs

- **`data/`** — datasets. **Convention:** every dataset we actively
  train or evaluate on is one whole file directly in `data/`
  (`data/<name>.txt`) — never sharded, never a directory of pieces; a
  derived variant is another whole file under its own name. Today:
  `books3.txt`, `pride.txt`, `soda_{train,valid,test}.txt` (the
  User/Bot rendering of SODA), `soda_s0.txt` … `soda_s5.txt`
  (simplified training variants), `eval_seeds.jsonl` (the eval's raw
  input: the SODA validation dialogs the scripted examiner follows).
  Preprocessed copies (case folding and the like) are not kept —
  processing happens on the fly. Subdirectories hold raw downloads we
  are not training on directly (`data/soda/` — the parquet source of
  the SODA files; `data/lmsys-chat-1m/`, `data/openhermes-2.5/`,
  `data/daily_dialog*/`) and sources that datasets are rendered from
  (`data/simplified/` — LLM process logs; `data/dialogs/` — recorded
  LLM–LLM dialogs). A dataset that becomes active is rendered to a
  whole file directly in `data/`. Everything is gitignored except the
  committed toy dataset `data/pride.txt`.
- **`evals/`** — everything an eval run produces or is configured by:
  generated dialogs, grades, reports, server logs, and the eval
  settings — as files for now; results and settings may migrate to a
  DB later. This is the rig's directory: `texmo.py eval` prints
  numbers and stores nothing. Gitignored.
- **`chat-logs/`** — transcripts of actual conversations with models:
  the `chat` REPL, one-off tries, and any conversations a web page
  may later collect. Kept apart from `evals/` deliberately — eval
  output is machine-generated, large and regenerable; transcripts are
  human, small, irreplaceable, and may contain personal data.
  Gitignored.
- **`results/`** — the search result DB (`db.sqlite`), the persisted
  predictors, and various run outputs. Per machine, gitignored;
  backup rules live in the untracked `CLAUDE.local.md` on the search
  machine. Trained language models do not go here — they go to
  `models/`.

## Everything else

- **`docs/`** — design and reference docs, indexed from `CLAUDE.md`.
- **`papers/`** — downloaded papers (`docs/links.md` records which
  ones matter and why).
- **`scratch/`** — untracked working space for any temporary stuff:
  probes, one-off experiments, working data (see `CLAUDE.md`,
  "Scratch space"). Also currently home to `dialog_prefix.txt`, the
  prefix used with `train --prefix-file` to eyeball a freshly trained
  chat model.
- **`config.py`** / `config_sample.py`, **`named_regexes.json`** /
  `named_regexes_sample.json` — per-host configuration; the `_sample`
  files are the committed templates.

## In flux (2026-08-31)

- **`scripts/`** is being dissolved and will disappear: the chat
  eval, dialog harness, corpus processing and dataset preparation
  move to `rig/`; the DB/report helpers (`make_best_models.py`,
  `export_top_confs.py`, `coverage.py`, `pjson_format.py`) and the
  benchmark suite tools become `texmo.py` commands (`best-models`,
  `export-top-confs`, `coverage`, `pjson-format`, `bench`,
  `bench-suite`); the retired one-off tokenset generators
  (`make_shift32/64.py`, `make_fold32.py`, `make_bucket.py`) go to
  `scratch/`; `migrate_norm_rename.py` and `fold_corpus.py` are
  deleted. The vendored `scripts/eliza/` moves to `llms/eliza/`.
- `rig.py`, `rig/` and the `llms/` `meta.json` registry do not exist
  yet (`rig.sh` and the `rig` dependency group do); until the
  migration, the rig-to-be commands are still invoked as
  `uv run python scripts/<name>.py …`.
- `README.md` predates this layout and needs a revision; `setup.txt`
  is per-machine setup notes and stays uncommitted.
