# Search

Texmo searches over model architecture and training hyperparameters to find
configurations with the best loss within a compute budget. The search is
distributed: a server holds the template and database, and clients pull
configurations to train and post results back.

## Key pieces

- **Configuration** (`configuration.py`) — an immutable tuple of
  `(model, lr, length, batch, steps, decay)`. The `precision` property
  delegates to `model.precision`.
- **Template** (`configuration.py`) — an immutable descriptor of the region
  of configuration space being explored. Has bounds on each hyperparameter,
  plus a model spec (either exact or regex).
- **DbReader / DbWriter** (`db/`) — SQLite database of confs and runs,
  split by read/write boundary.
- **Search** (`search.py`) — the `select_conf(system)` strategy that
  decides what to run next.
- **Server** (`server.py`) — Flask app that hosts search, serves `/select`
  and `/add` to clients, and `/` (index) to humans.
- **Client** (`client.py`) — worker loop that pulls from `/select`, runs
  the training with `Manager`, posts back via `/add`.

## Weighted template sets (sub-searches)

The server can split its select budget between several sub-searches
that share **every** template bound and differ **only** in their spec
filter — e.g. 60% unrestricted, 20% transformer-shaped, 20% a newly
implemented layer. Without this a fresh layer competes from cold
against a mature general population and is starved before it can be
fairly measured.

Configured with `--templates <path.json>`, or row by row in the
sub-search table on the index page (same thing, two spellings):

```json
[
  {"name": "main", "share": 60},
  {"name": "attn", "share": 20, "regex": ".*\\|split\\.add\\(.*attn.*",
   "default_spec": "bytes|split.add(norm-attn.8.2.8, pass)"}
]
```

- `share` values are normalized (they need not sum to 100).
- `regex` uses the same syntax as `--spec`; omitted = unrestricted.
- `default_spec` seeds the sub-space — semantically the *smallest*
  conf that agrees with the entry. Omitted, it is derived by
  `default_from_template` against the entry's template, which fails
  loudly at configuration time if nothing matches.

`select_conf` draws one entry per call, weighted by share, right after
the (template-blind) pick_me and warmup steps. Everything below the
draw runs inside that entry: the coverage walk, the layer cap, the
`max_weights` ceiling, the strategy, and both fallbacks. Consequences
worth knowing:

- **Per-entry frontier.** The coverage walk pulls `top_confs_global`
  for the *entry's* template, so a conf that is Pareto-optimal for a
  sub-template but dominated globally still gets its cross-system
  re-runs. Its sticky flag and progress stats are keyed by
  `(system, entry)`.
- **Per-entry weight ceiling.** `_select_max_weights` derives from the
  best conf *within* the entry, so each sub-space gets a weight range
  matched to its own population.
- **Per-entry layer caps.** `_LAYER_CAP_PROBS` drops every cap below
  the entry's minimum layer count (from its default conf) and
  renormalizes. A transformer block is ~9 layers, so without this 40%
  of its selects would query an empty intersection and fall through.
- **Per-entry default fallback.** The last-resort fallback returns the
  drawn entry's default conf. This is what bootstraps an empty
  sub-space; without it a new entry returns None and its budget
  quietly drains into the general search.

With no `--templates` there is a single entry (`main`, carrying
`--spec` if given), and the draw is a short-circuit that consumes no
randomness — selection is byte-for-byte what it was before template
sets existed. Model specs live **only** in the entries: the base
template they share has no spec of its own, so nothing filters twice
and the index form has one place, not two, where specs are set.

The index page shows each entry's nominal share next to its **realized**
share and the raw select count behind it (an in-memory counter of confs
served per entry since server start, no schema change; realized share
means little below a few hundred selects). The top-confs and compare
views can be filtered to one entry, which just applies its regex at
query time. Nothing about which entry produced a run is recorded in
the DB.

## Search strategies

`Search.select_conf(system)` first picks a time budget `t` (log-uniform
within the configured range) and a weight budget `max_weights`
(log-uniform between the template's min weights and
`min(8 × top_conf.weights, template max weights)`), then tries the
strategies below in order. Each gated strategy succeeds with a fixed
probability **conditional on having reached it**; on miss it falls
through to the next.

For every selected configuration we record which strategy picked it
(stored in `run.strategy`) and whether the resulting run changed the
winning conf at `(system, train_time, num_weights)` (stored in
`run.changed_winner`). `uv run texmo.py strategy-stats` reports
runs-per-strategy and the % that changed the winner — the live signal
for whether each strategy is earning its slot.

### 1. `predicted_2nd_neighbor`

Gated by `_PREDICTED_2ND_NEIGHBOR_PROB`. Requires both the loss
model and the timing model to be fitted.

Pipeline (shared with `predicted_3rd_neighbor`; see
`Search._select_predicted_best_impl`):

1. **Seed pick.** Take the top conf (lowest median loss) for
   `(system, max_weights, max_time ≤ t)` — `t` is the per-select
   sampled time budget. If none, bail.
2. **BFS over unnormalized neighbors** to `bfs_depth` (2 for this
   strategy). The set is deduped on the raw `Configuration` — every
   step / batch / lr / arch mutation produced by `conf_neighbors`
   stays distinct in the walk. Over-budget intermediates are kept;
   they're just bridges to other candidates at depth N.
3. **Weight filter.** Drop confs whose `num_weights > max_weights`.
4. **Per-conf step cap.** For each survivor, ask the timing model
   for the predicted total time at the conf's current `steps`. If
   it already fits `max_t = train_time[1]`, keep the conf
   unchanged. Otherwise replace `steps` with the largest pow2 that
   does fit; drop the conf if even the smallest run won't fit.
   **The cap is to `max_t`, not to the sampled `t`** — `t` is for
   seed selection only, so step counts are scored at the largest
   budget the user has authorized. Multiple over-budget neighbors
   can collapse to the same capped form; dedupe on the result.
5. **Score and walk.** Predict log-loss for each surviving conf,
   compute a **compound** median of `(predicted_log_loss,
   observed_log_losses...)`, sort ascending. Walk the top 9 across
   run-limit sequences `[1] / [2,1,1] / [3,2,2,1×6]`, returning
   the first candidate whose total run count is below the
   position's limit.

The "BFS doesn't normalize, post-BFS does" split is what preserves
(W, T) variety end-to-end: under-budget step counts from the
neighbor walk survive into the scoring pool, instead of collapsing
into a single max-fitting T per W.

### 2. `predicted_3rd_neighbor`

Gated by `_PREDICTED_3RD_NEIGHBOR_PROB`. Same pipeline as the
2nd-neighbor strategy but with `bfs_depth = 3` (~1000 candidates).
Larger jump from the seed into less-explored territory; predictor
noise is correspondingly higher.

### 3. `time_budget`

Gated by `_TIME_BUDGET_PROB`. Score-ordered scan of confs with
`time_estimate ≤ t AND weights ≤ max_weights` (using either median
or predicted time estimates). For each iteration of an expanding
run-limit sequence (`[1] / [2,1,1] / [3,2,2,1×6] / ...`), pick the
first position whose total run count is below the limit at that
position, or whose `system_runs == 0`.

### 4. `neighbor`

The always-available fallback. Iterate top confs and their direct
neighbors looking for a candidate that needs more runs, with
expansion depth and per-position run thresholds defined by an
expanding sequence. For each neighbor, pick it if:

- `system_runs == 0` (needs a run on this system for cross-system
  visibility), OR
- `total_runs < min_neighbor` (confidence building).

### 5. `default`

Final fallback: return the default configuration of the sub-template
drawn for this select (the whole template's default when no template
set is configured). Skipped if that conf doesn't match its own template
or has already been explored enough.

## Loss prediction model

The two `predicted_*` strategies depend on a tiny RNN that predicts
log-loss from the configuration's globals + per-layer feature
sequence. See [`loss_prediction.md`](loss_prediction.md) for
architecture, training, and current accuracy (~3.6% typical error).

## Time prediction model

Per-step training time is predicted by an additive linear-in-features
model fit per `(system, precision)` pair. See
[`timing.md`](timing.md). The estimate per conf+system is stored in
`conf_time_estimate` (either as a `'median'` of observed runs or as
a `'predicted'` value).

## Neighbor generation

Two layers:

- **`LayerDef.neighbors()`** — size/length 2x mutations and type swaps
  between related layer types (dense↔rnn, rnn↔gru/mgru/mingru, etc.). See
  `layer.py:LayerDef.neighbors` for the exact mapping.
- **`Model2Def.neighbors()`** — precision changes, input-layer mutations,
  single-layer mutations (via `LayerDef.neighbors`, recursing into split
  branches), append/remove last layer, insert/remove suffix.2,
  insert/remove norm, prepend/remove a dense lead-in, and the Split
  mutations (wrap/unwrap residual and gate Splits, `add↔cat` op-swap,
  grow/shrink a residual span, append a self-gate). Each mutation is a
  spec string re-parsed via `parse_model2` and `is_valid`-filtered. See
  [`split.md`](split.md).

Neighbors are generated **at runtime** (in-memory). No neighbor table in
the DB. `DbReader.get_conf_id()` is cached (Configuration → id) so
neighbor lookups are cheap.

Validity constraints (`Model2Def.is_valid`, recursing through
`LayerSeqDef` / `SplitDef`):
- Input must be valid.
- No two adjacent "suffix-like" layers (length > 1).
- Norm/rmsnorm can't open the top-level chain (a split branch may —
  that's the pre-norm residual pattern).
- No two adjacent norms.
- Norm can't follow a suffix.
- A bare (activation-less) dense is legal iff its consumer doesn't
  start with a full linear projection (`LayerDef.projects_input`):
  before conv/rglru/norms, as a branch terminal — not before
  dense/rnn/gru-family or at the chain end.
- Split rules: exactly 2 branches, canonical `pass` position, `mul`
  branches equal-sized, no suffix-terminal branch. See [`split.md`](split.md).

## Cross-system behavior

Multiple machines can work on the same database concurrently. Each machine
explores independently. See `docs/decay_and_checkpoints.md` for the
reasoning behind this.

The `system_runs == 0` rule in `_select_top_neighbor` is the current
mechanism for bootstrapping new systems: a top neighbor that's never been
run locally gets picked once, even if its total run count is already high.

### Why a conf's first run reads optimistically

A conf's first recorded loss is biased low, by a measured **0.3-0.6%**
across every machine in the fleet (2026-08-08). Nothing is broken —
this is regression to the mean, and it is the direct consequence of
the search re-running confs *because* they scored well: the first run
is the one that selected the conf, so every later run drifts back
toward the truth.

Two existing rules already absorb it: top confs get re-run precisely
to correct the selection, and `top_confs_*` takes `min_num_runs=2` so
a single lucky run cannot reach a leaderboard. The magnitude is worth
knowing mainly as a calibration — 0.3-0.6% sits well under the 1-2%
differences the search adjudicates, so the two-run floor is
adequately sized rather than merely pointed the right way. Re-check
it against this number if anything ever promotes confs on fewer runs.

## Database access (`texmo/db/`)

- **`DbWriter`** — opens the SQLite file read-write. SearchThread and
  ModelThread each hold their own writer (both `check_same_thread=False`),
  serialized via SQLite's WAL + `BEGIN IMMEDIATE` + a 30 s `busy_timeout`.
  Step 6 of the threading refactor will collapse them to a single
  writer on a dedicated DBWriter thread.
- **`DbReader`** — opens the SQLite file with `mode=ro` (URI form).
  Flask request handlers open a fresh `DbReader(self.path)` per
  request; Search holds a persistent reader for its `select` loop.
- **Score computation** — `median_score` is the median loss across all
  runs of a conf (across all systems, since runs are equivalent by dtype).
  Per-system time estimates live in `conf_time_estimate` with
  `source IN ('median', 'predicted')`; the median estimate is written
  on every `add_run` and supersedes any predicted value for the same
  `(conf, system)`.
- **Persisted models** — the fitted timing model and loss model are
  pickled next to the DB file as `timing_model.pickle` /
  `loss_model.pickle` (`predict/persist.py`), so the server doesn't
  re-fit on restart. Both are refit on the model thread; clients only
  ever train configurations.

CLI commands:
- `db-update` — recompute all scores.
- `db-clear-system <name>` — delete all runs from a given system and
  recompute scores.
- `db-bootstrap-estimates` — backfill medians, fit per-`(system, precision)`
  timing models, and write predicted estimates for confs without runs.
- `strategy-stats` — per-strategy run count and `% of runs that
  changed the (t, w)-winner`.

## Server UI

The `/` page has a template form (editable), a system filter dropdown,
a graph (score vs. weights, Pareto frontier), and a table of top configs.

- **Template form** — posting to `/update` creates a new Template via
  `Template.from_form()` and updates `search_thread.search.template`
  (atomic swap, not mutation). It holds the bounds shared by every
  entry (weights, layers, length, batch, steps, lr, time, precisions,
  decay types) and **no spec field**.
- **Sub-search rows** — one row per entry: name, the two model specs
  (regex and default, stacked full-width in one column since real ones
  run 40-80 characters), and share, with Add / Remove buttons and a
  live-computed normalized percentage next to each share. There is
  always at least one row; with no sub-searches configured it is a
  single `main` row whose blank regex means unrestricted. The four
  fields post as parallel repeated fields (`entry_name`, `entry_regex`,
  `entry_default_spec`, `entry_share`) and are zipped back by position;
  rows left entirely blank are dropped. The set is rebuilt against the
  new base template on every update; anything malformed (a nameless or
  shareless row, a non-numeric or negative share, duplicate names, a
  bad regex, a `default_spec` outside its own entry) is rejected with
  the error banner, leaves the running search alone, and comes back
  with the submitted values still in the fields.
- **System dropdown** — filters the graph and table to configs that have
  at least one run on the selected system.
- **Sub-search dropdown** — with a template set configured, filters the
  graph and table to one entry's frontier.
- **Copy link** — each row has a "copy" link that copies a
  `uv run texmo.py train ...` command to the clipboard.

