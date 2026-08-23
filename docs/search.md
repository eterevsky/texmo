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
sub-search table on the index page. The two are the same object, but
not the same spelling: the JSON carries patterns inline, while a UI
row names a **preset** and the server resolves it (see [Named regex
presets](#named-regex-presets-named_regexesjson)).

```json
[
  {"name": "main", "share": 60},
  {"name": "attn", "share": 20, "regex": ".*\\|split\\.add\\(.*attn.*",
   "default_spec": "bytes|split.add(norm-attn.8.2.8, pass)",
   "seed": true}
]
```

- `share` values are normalized (they need not sum to 100).
- `regex` uses the same syntax as `--spec`; omitted = unrestricted.
- `default_spec` seeds the sub-space — semantically the *smallest*
  conf that agrees with the entry. Omitted, it is derived by
  `default_from_template` against the entry's template, which fails
  loudly at configuration time if nothing matches.
- `seed` (default false) turns on [frontier
  seeding](#frontier-seeding) for the entry.

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

### Frontier seeding

A share of the budget is not enough on its own for a brand-new family.
`bits.4.pair.*` landed with its own sub-search and was still visited
three times in its first day, because the only way *into* it is a
handful of bridge edges (`bits.4.oh+bp ↔ .16`, `tokens.32.hexbpe.oh
↔`) that the neighbor walk has to stumble onto at random. Seeding
walks those edges deliberately instead.

Per entry, a **Seed** checkbox (`"seed": true` in `--templates`). While
it is on, `select_conf` serves that entry from a queue instead of
running its strategy lottery:

1. Take the Pareto frontier of the **unrestricted** template — the
   same `top_confs_global` list the index page shows, within the
   server's weight bounds. Not the entry's own frontier: the point is
   to carry what the general search has found *into* the sub-space.
2. `conf_neighbors(source, entry.template)` for each frontier conf,
   keeping what the entry's full `match` admits (`conf_neighbors`
   re-checks the spec on model mutations only, so a steps or lr
   neighbor of an off-regex source is still off-regex), that is valid,
   and that is not a retired input.
3. Drop anything with a recorded run anywhere. Zero runs is what makes
   this **run-once**; anything already measured is somebody else's job.
4. Shuffle, cap at `_SEED_QUEUE_CAP` (200), cache.

A seeded candidate keeps its **source's metaparams** — a spec mutation
inherits `(lr, batch, length, steps)` — so a new family's head arrives
pre-tuned rather than at the sub-space's cold default. Before it goes
out, `_cap_seed_steps` checks the timing model for the *requesting*
system and reduces `steps` to the largest power of two that fits the
time limit; with no fit for that `(system, precision)` the conf goes
unchanged (the first run on a new pair is what seeds the predictor),
and an unfittable candidate is floored rather than dropped, matching
`_cap_steps`.

Seeding sits **first inside the entry**, ahead of the coverage walk
and the lottery — deliberately, since the coverage walk's sticky flag
can fire at 100%. It yields as soon as the queue is empty, and the
switch stays on: no extra budget knob, because the entry's share
already governs the drain rate and run-once bounds the total spend.
The picked conf is recorded as strategy `frontier_seed`.

**Cache and invalidation.** The sweep is one `conf_neighbors` call per
frontier conf (~200 of them), so it must not run per select. Each
entry's queue is cached and rebuilt when:

- **the frontier moves** — `DbWriter.add_run` already samples the
  winning conf at `(system, train_time, num_weights)` before and after
  each insert (the `changed_winner` column), and the writer thread
  publishes each flip by bumping a shared `FrontierVersion` counter
  the search thread reads (see [`threads.md`](threads.md));
- **the queue drains** — one rebuild to see whether the sweep still
  has anything. A queue that was built *empty* is not stale, so a
  sub-space with no live bridge edges doesn't pay for the sweep on
  every select;
- **the entry changes** — `set_templates` drops every cached queue,
  since an `/update` rebuilds all entries and may have moved the
  shared bounds under them.

Each candidate's zero-run status is re-checked at pop time: the queue
can be minutes old, another worker may have run it since, and run-once
means once.

The row's live stats show how many candidates are left. Leave Seed on
only while a sub-search is still new — the queue refills whenever the
frontier moves, so on a broad entry it never stops.

## Search strategies

Two strategies run before any of this, inside the drawn sub-search
entry: [frontier seeding](#frontier-seeding) when its queue has
anything left, then the coverage walk.

`Search.select_conf(system)` then picks a time budget `t` (log-uniform
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
  re-fit on restart. The timing model is always refit on the model
  thread; the loss model is refit either there or on a worker, per
  `config.LOSS_REFIT` (see
  [`loss_prediction.md`](loss_prediction.md)).

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
- **Sub-search table** — one row per entry, editing and monitoring in
  the same place: a **preset** dropdown (the row's identity *and* its
  spec filter), an editable **share** with a live-computed normalized
  percentage, a **Seed** checkbox ([frontier
  seeding](#frontier-seeding)), the row's **regex and default**
  rendered read-only (stacked full-width in one column, since real
  ones run 40-80 characters), its **live stats** (nominal vs realized
  share, select count, seed queue remaining, default conf, linking to
  that entry's frontier), and Remove. Add appends a row on
  `Unrestricted` with an empty share.

  Two fields post positionally — `entry_preset` and `entry_share`, as
  parallel repeated fields zipped back by position. `entry_seed`
  cannot join them: a browser posts nothing at all for an unchecked
  box, so an unticked row wouldn't hold a slot and every later row's
  flag would shift up one. Each box carries **its own row index** as
  its value instead (the page's `updateShares`, which already walks
  every row on load / Add / Remove, keeps those current), so the
  posted list simply names the ticked rows. **The regex is not
  submittable**: `/update` resolves each preset name against
  `named_regexes.json` as of that request and snapshots the result
  into the entry. There is always at least one row; with no
  sub-searches configured it is a single `Unrestricted` row, which is
  the `main` entry (see below). Rows still on `Unrestricted` with a
  blank share are dropped as untouched. The set is rebuilt against the
  new base template on every update; anything malformed (a shareless
  row, a non-numeric or negative share, the same preset on two rows, a
  preset that is no longer in the file, or a `default_spec` outside
  its own entry) is rejected with the error banner, leaves the running
  search alone, and comes back with the submitted rows still selected.
- **Spec filter on `/compare`** — one control per column: the same
  preset dropdown, posting `t1_preset` / `t2_preset` as the column's
  whole spec filter. (It replaced three overlapping controls — a
  free-text regex box, a dropdown that filled it, and a sub-search
  select that overrode it — once sub-search entries became presets and
  the three could disagree.) Names resolve fresh per render, so a
  bookmarked compare URL naming a preset that has since been edited
  out renders the form with an error line rather than failing, and
  keeps the name selected so the link works again once the file does.
  Compare uses the preset's regex only; `default_spec` seeds a
  sub-search and means nothing to a query.
- **System dropdown** — filters the graph and table to configs that have
  at least one run on the selected system.
- **Sub-search dropdown** — with a template set configured, filters the
  index graph and table to one entry's frontier by name (`?entry=`);
  entry names are preset names now, except the unrestricted `main`.
- **Copy link** — each row has a "copy" link that copies a
  `uv run texmo.py train ...` command to the clipboard.

### Named regex presets (`named_regexes.json`)

Sub-search regexes worth keeping are long and fiddly, and the live
template state is rebuilt on every `/update` — so without somewhere to
put them, a pattern that took real effort to get right lives only as
long as the browser tab. Presets are that place.

The store is **hand-edited**: a JSON file named `named_regexes.json`
in the repo top directory, next to `config.py` (per-host personal
state, so it is gitignored — there is no UI for editing it, and the
server never writes to it). `named_regexes_sample.json` is a
committed starter — copy it over, like `config_sample.py` — carrying
a real transformer-block pattern with the JSON escaping already
right. Name → pattern, `default_spec` optional:

```json
{
  "transformers": {
    "regex": ".*\\|split\\.add\\(.*attn.*",
    "default_spec": "bytes|split.add(norm-attn.8.2.8, pass)"
  },
  "recurrent": {"regex": ".*\\|.*(gru|lstm|rnn)\\..*"},
  "tied codec": {"regex": ".*emb.*"}
}
```

Remember that JSON needs its backslashes doubled: a regex
`.*\|attn.*` is written `".*\\|attn.*"`.

Presets are the **only** way to name a spec filter in the UI: both the
index form and the compare columns post preset names, never patterns.
Iterating on a regex therefore means editing this file and reloading —
which is also what makes the pattern reviewable and reusable instead
of living in one browser tab.

Behaviour worth knowing:

- **Read fresh on every page render and on every Update**, never
  cached. Edit the file, refresh, the dropdown is current — no
  restart. (The file is tiny; this costs nothing.)
- **Resolved at Update, then snapshotted.** The entry keeps the regex
  and default it was built with, so editing the file cannot quietly
  change what a running search is doing. Until the next Update the two
  can differ, and the row says so: a ⚠ marker whose tooltip explains
  whether the preset has changed or disappeared. That is information,
  never an error — nothing is rejected until you press Update.
- **Entry names are preset names.** They key the select counters and
  the per-(system, entry) coverage flags, so renaming a preset in the
  file and re-applying it starts that entry's coverage bookkeeping
  fresh. The `Unrestricted` row is the exception: it maps to the
  historical `main` entry name, so the unrestricted search keeps its
  identity and counters across the switch to presets.
- **A name that no longer resolves** (deleted from the file, or an
  entry that arrived via `--templates` / `--spec` and never was a
  preset) still renders as a selectable option, so the form
  round-trips and the row can be seen and fixed; pressing Update with
  it still selected is what reports the error. When a `--spec` pattern
  is textually equal to a preset's, the row starts on that preset
  instead.
- **Lenient loading.** A hand-edited file must never take the page
  down, so anything unusable is skipped with a `logging.warning`
  naming the entry, and the rest still render. Skipped: entries whose
  value isn't an object, entries with no `regex` (or a blank one), a
  `regex` that doesn't compile (the DB's `REGEXP` bridge uses the same
  `re` module, so it could never match anyway), blank names, and
  duplicates. A file that doesn't parse as JSON, or isn't a
  JSON object, means no presets at all — again with a warning. A
  missing file is normal and silent.
- **`Unrestricted` is reserved.** Every dropdown offers it first as
  the virtual "no spec filter" choice (picking it clears the row), so
  a stored entry by that name could never be selected and is skipped,
  case-insensitively.
- **`default_spec` is not validated at load time.** A typo there
  surfaces on `/update`, where the existing entry validation reports
  it in the error banner with the surrounding context — see
  `TemplateEntry`.

