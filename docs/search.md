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
- **Search** (`search2.py`) — the `select_conf(system)` strategy that
  decides what to run next.
- **Server** (`server.py`) — Flask app that hosts search, serves `/select`
  and `/add` to clients, and `/` (index) to humans.
- **Client** (`client.py`) — worker loop that pulls from `/select`, runs
  the training with `Manager`, posts back via `/add`.

## Search strategy

`Search.select_conf(system)` tries, in order:

1. **`_select_top_neighbor`** — iterate top configs (from
   `top_confs_for_system(system, ...)`) and their neighbors, looking for
   a candidate that needs more runs. Expansion depth and required run counts
   come from `_generate_limits()`. For each neighbor, we pick it if:
   - `system_runs == 0` (needs a run on this system for cross-system
     visibility), OR
   - `total_runs < min_neighbor` (confidence building)
2. **Fallback** — return the initial/default configuration from the template.

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

- **Main writer connection** — SearchThread only, `check_same_thread=False`.
- **Read-only connections** — `db.open_readonly()` returns a new URI
  `file:...?mode=ro` connection used by Flask request handlers. Context
  manager: `with self.db.open_readonly() as ro_db: ...`
- **Score computation** — `median_score` is the median loss across all
  runs of a conf (across all systems, since runs are equivalent by dtype).
  `median_time` is per-system, stored in the `conf_time` table.

CLI commands:
- `db-update` — recompute all scores.
- `db-clear-system <name>` — delete all runs from a given system and
  recompute scores.

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

## Future directions

The current search strategy is a greedy walk — it looks at top confs
and their neighbors, picking the one with the fewest runs. This is a
reasonable baseline but doesn't use any information about what's likely
to improve.

Once we have a loss prediction model (predicting expected loss from
architecture + hyperparameters), we can add smarter strategies: skip
neighbor mutations that are unlikely to improve on the current best,
prioritize mutations in high-uncertainty regions, or run simulated
annealing / evolutionary search over the model space.
