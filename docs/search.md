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
- **ResultDB** (`resultdb.py`) — SQLite database of confs and runs.
- **Search** (`search.py`) — the `select_conf(system)` strategy that
  decides what to run next.
- **Server** (`server.py`) — Flask app that hosts search, serves `/select`
  and `/add` to clients, and `/` (index) to humans.
- **Client** (`client.py`) — worker loop that pulls from `/select`, runs
  the training with `Manager`, posts back via `/add`.

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

Picked ~15% of the time (tunable). Requires both the loss model and
the timing model to be fitted.

Take the top conf (lowest median loss) for `(system, max_weights,
max_time≤t)` as a seed. BFS to depth 2 (~100 candidates), filter by `weights ≤
max_weights`, predict per-step time and adjust `steps` to the budget,
predict log-loss for each adjusted candidate, then score each by the
**compound** median of `(predicted_log_loss, observed_log_losses...)`.
Walk the top 9 across run-limit sequences `[1] / [2,1,1] / [3,2,2,1×6]`,
returning the first candidate whose total run count is below the
position's limit.

### 2. `predicted_3rd_neighbor`

Picked ~20% of the remaining 85% (tunable). Same machinery as the
2nd-neighbor strategy but BFS depth 3 (~1000 candidates). Larger jump
from the seed into less-explored territory; predictor noise is
correspondingly higher.

### 3. `time_budget`

Picked ~30% of the time, conditional on the two predictor strategies
not firing (tunable). Score-ordered scan of confs with `time_estimate
≤ t AND weights ≤ max_weights` (using either median or predicted time
estimates). For each iteration of an expanding run-limit sequence
(`[1] / [2,1,1] / [3,2,2,1×6] / ...`), pick the first position whose
total run count is below the limit at that position, or whose
`system_runs == 0`.

### 4. `neighbor`

The always-available fallback. Iterate top confs and their direct
neighbors looking for a candidate that needs more runs, with
expansion depth and per-position run thresholds defined by an
expanding sequence. For each neighbor, pick it if:

- `system_runs == 0` (needs a run on this system for cross-system
  visibility), OR
- `total_runs < min_neighbor` (confidence building).

### 5. `default`

Final fallback: return the initial/default configuration from the
template (skipped if the init conf doesn't match the current template
or has already been explored enough).

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
- **`ModelDef.neighbors()`** — precision changes, input-layer mutations,
  single-layer mutations (via `LayerDef.neighbors`), append/remove last
  layer, insert/remove suffix.2, insert/remove norm.

Neighbors are generated **at runtime** (in-memory). No neighbor table in
the DB. `ResultDB.get_conf_id()` is cached (Configuration → id) so
neighbor lookups are cheap.

Validity constraints (`ModelDef.is_valid`):
- Input must be valid.
- No two adjacent "suffix-like" layers (length > 1).
- Norm can't be the first layer.
- No two adjacent norms.
- Norm can't follow a suffix.

## Cross-system behavior

Multiple machines can work on the same database concurrently. Each machine
explores independently. See `docs/decay_and_checkpoints.md` for the
reasoning behind this.

The `system_runs == 0` rule in `_select_top_neighbor` is the current
mechanism for bootstrapping new systems: a top neighbor that's never been
run locally gets picked once, even if its total run count is already high.

## ResultDB

- **Main writer connections** — SearchThread and ModelTrainingThread
  each hold their own connection (both `check_same_thread=False`),
  serialized via SQLite's WAL + `BEGIN IMMEDIATE` + a 30 s
  `busy_timeout`.
- **Read-only connections** — `db.open_readonly()` returns a new URI
  `file:...?mode=ro` connection used by Flask request handlers. Context
  manager: `with self.db.open_readonly() as ro_db: ...`
- **Score computation** — `median_score` is the median loss across all
  runs of a conf (across all systems, since runs are equivalent by dtype).
  Per-system time estimates live in `conf_time_estimate` with
  `source IN ('median', 'predicted')`; the median estimate is written
  on every `add_run` and supersedes any predicted value for the same
  `(conf, system)`.
- **Persisted models** — the fitted timing model and loss model are
  pickled into the `model` table under keys `'timing'` and `'loss'`,
  so the server doesn't re-fit on restart.

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
  (atomic swap, not mutation).
- **System dropdown** — filters the graph and table to configs that have
  at least one run on the selected system.
- **Copy link** — each row has a "copy" link that copies a
  `uv run texmo.py train ...` command to the clipboard.

