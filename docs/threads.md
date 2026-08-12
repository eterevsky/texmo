# Threading model

How the search server splits work across threads: all DB writes land on
a single dedicated thread, and Search is stateless and read-only.

## Rule

> Threads share only **read-only** objects. Anything that needs to be
> mutated by another thread is sent through a **Queue**.

The few legitimately shared-mutable objects (`TrainTimingModel`,
`LossModelHolder`, `latency` measures) are encapsulated behind narrow
interfaces with clearly identified writer threads.

## Message format

Each queue that carries more than one kind of payload defines a small
set of per-command dataclasses (one class per command) and a union
type alias for the queue. Consumers dispatch with `match` class
patterns, which do `isinstance` + named-field destructure in one
step. Fields are flat and typed; the consumer doesn't pay for
tuple/dict-unpacking and the type checker can catch shape mismatches.

```python
@dataclass
class RunAdded:
    system: str
    precision: Precision

@dataclass
class LossRefit:
    pass

@dataclass
class Stop:
    pass

TrainMessage = RunAdded | LossRefit | BootstrapTiming | Stop


# In the consumer:
match m:
    case RunAdded(system=s, precision=p):
        ...
    case LossRefit():
        ...
    case Stop():
        break
    case _:
        assert False, f"unknown: {m!r}"
```

RPC-style messages add a reply-queue field on just the classes that
need it (`latency._Report(reply)`), not on every message.

The simple object-only queues (`confs_by_system[system]`, which just
carries a `SearchResult | None`) are exempt — they don't need the
wrapper.

## Threads

- **Werkzeug request handlers** (internal :5000, external :5001)
  - Count: many (one per request).
  - Owns: nothing persistent.
  - Reads from: a per-request `DbReader`, only for the report-style
    routes (`/`, `/compare`, `/throughput`, `/fastest`, and `/update`
    when it rejects a form and re-renders the index instead of
    redirecting). The hot paths
    (`/select`, `/add`) never touch the DB directly, and `/latency`
    doesn't either.
  - Sends to:
    - `WriterThread` via `write_queue` to record incoming runs
      (`/add`), through the `SearchServer`-owned `DbWriterProxy`.
    - `SearchThread` via `requests_queue`: one `Select` per `/select`
      request (plus one extra the first time a system is seen, which
      primes that system's pipeline by one), and a `SetTemplate` from
      `/update`.
    - `ModelThread` via `train_queue` to notify it of newly added runs
      so it can update its own counters and decide on refits.
  - No request handler fits or publishes a model; every refit happens
    on ModelThread.

- **WriterThread** (`texmo/db/writer.py`)
  - Count: 1.
  - Owns: the only post-init `DbWriter` (read-write connection) and
    all write-transaction state. Opens it inside `run()`, so sqlite's
    default `check_same_thread=True` holds.
  - Reads from: its `write_queue`.
  - Sends to: nothing.

- **SearchThread** (`texmo/search.py`)
  - Count: 1 (created in `SearchServer.__init__`).
  - Owns: its own persistent `DbReader` and the `Search` object, both
    built inside `run()`.
  - Reads from:
    - `DbReader` for query work.
    - The shared `TrainTimingModel` and `LossModelHolder` instances
      (writer is the Model thread; see "Shared mutable objects").
    - `requests_queue` for incoming commands.
  - Sends to: the matching per-system queue in `confs_by_system`,
    dropping each produced `SearchResult` (or `None`) onto it.
  - Writes nothing. A `Select` whose `enqueued_at` is older than
    `_SELECT_STALE_S` is answered with `None` without doing the work —
    its client has certainly timed out. A `select_conf` exception is
    caught and logged rather than killing the thread, since a dead
    search thread would leave every `/select` blocked forever.

- **ModelThread** (`texmo/predict/model_thread.py`)
  - Count: 1.
  - Owns (mutates): the shared `TrainTimingModel` (per-(system,
    precision) weight inserts; one shared instance is reused across
    threads), and the underlying `LossModel` slot inside the shared
    `LossModelHolder` (replaced wholesale on refit).
  - Owns (private): per-(system, precision) run counters and refit
    counters (the latter drive the incremental-vs-full estimate
    refresh cadence), plus the global labeled-run counter behind the
    loss-refit cadence (`_LOSS_REFIT_EVERY`).
  - Fits both predictors: per-pair timing fits (~30 s each) and the
    full loss refit (~2 min, all labeled runs). Both are synchronous
    on this thread, so everything else on `train_queue` waits;
    that is the accepted cost of keeping fitting off the request
    path and out of the clients.
  - Reads from: its own `DbReader` for fitting runs, plus
    `train_queue` for incoming notifications.
  - Sends to: `WriterThread` via `write_queue` (through a
    `DbWriterProxy`) to persist predicted-time estimates. Fitted
    predictors go to disk directly via `save_predictor`, not through
    the queue.

- **Latency** (dedicated thread inside `texmo/latency.py`; see
  "Shared mutable objects")
  - Count: 1, started lazily as a daemon at module load.
  - Owns: the `_measures` dict.
  - Reads from: its internal `_queue` (`_Record` / `_Report`).

- **Latency dump loop** (`SearchServer._dump_latency_loop`)
  - Count: 1 daemon, started by `serve()`. Appends a timestamped
    latency snapshot plus queue depths to `latency.log` every
    `_LATENCY_DUMP_INTERVAL` seconds. Touches no DB and no search
    state, so it keeps reporting even while `/select` is stalled.

The two werkzeug listener threads (`internal_srv.serve_forever` and
`external_srv.serve_forever`) are infrastructure, not application
threads — they dispatch into the request-handler thread pool.

## Queues

- `write_queue`
  - Producers: request handlers (`AddRun` on the `/add` path, via
    `SearchServer._writer`) and ModelThread (predicted-time bulk
    upserts). Both go through `DbWriterProxy`, which has the same
    method names as `DbWriter` and turns each call into a message.
  - Consumer: `WriterThread`.
  - Messages (`WriteMessage = AddRun | UpsertPredictedTimeEstimates
    | UpdateAllScores | Stop`):
    - `AddRun(conf, run, strategy, track_winner_change)`
    - `UpsertPredictedTimeEstimates(rows)` — the proxy splits a
      refresh into `_UPSERT_CHUNK` (20k) row batches so `AddRun`
      messages interleave between the writer's transactions instead
      of waiting behind one giant upsert.
    - `UpdateAllScores()` (used by the median backfill in
      `bootstrap()`)
    - `Stop()` — exit the loop.
  - SearchThread does **not** write. Anything bookkeeping-adjacent
    that needs to land in the DB is either posted by the `/add`
    handler or routed via ModelThread.

- `requests_queue`
  - Producers: request handlers only. `/select` posts one `Select`
    per request; the first request for a previously unseen system
    posts an extra one while holding `confs_by_system_lock`, which
    leaves that system's pipeline one conf ahead. `/update` posts
    `SetTemplate`.
  - Consumer: the SearchThread.
  - Messages (`SearchMessage = Select | SetTemplate | Stop`):
    - `Select(system, enqueued_at)`
    - `SetTemplate(template, init_conf, train_time, templates)` — the
      group moves together so a partial update never lands mid-loop.
    - `Stop()`

- `confs_by_system: dict[str, Queue[SearchResult | None]]`
  - Producer: SearchThread, one item per `Select` processed.
  - Consumer: `/select` handlers, which just `get()` from the
    target system's queue.
  - Lock: `confs_by_system_lock` protects creation of the per-system
    queue + its one-off extra seed, and every `put` / `qsize` on the
    dict's queues.

- `train_queue`
  - Producers:
    - Request handlers (`/add`), posting `RunAdded` after the `AddRun`
      write has been enqueued from the same thread — SQLite WAL then
      guarantees a later ModelThread read sees the row. `/add` drops
      runs without a loss before either enqueue, so `RunAdded` is one
      per *labeled* run.
    - `SearchServer.__init__` posts a one-shot `BootstrapTiming`
      and/or `LossRefit` at startup for whichever persisted predictor
      is missing or fails its compatibility probe.
  - Consumer: ModelThread.
  - Messages (`TrainMessage = RunAdded | LossRefit | BootstrapTiming
    | Stop`):
    - `RunAdded(system, precision)` — Model owns the per-pair
      counters and the global labeled-run counter, and decides when
      to fit the timing model and when to refit the loss model.
    - `LossRefit()` — force a loss-model refit now (startup only;
      the cadence itself lives in the `RunAdded` handler).
    - `BootstrapTiming()` — run the full timing bootstrap (median
      backfill, per-pair refits, predictions for every conf without a
      median) on the Model thread.
    - `Stop()` — exit the loop.
  - WriterThread does **not** know about training. The producer that
    posted the write is the same one that notifies the Model
    thread; the business logic stays out of the writer.

- `latency._queue`
  - Producers:
    - `Timer.__exit__` posts a `_Record(name, elapsed_ns)` from any
      thread.
    - `get_report()` posts a `_Report(reply)` and blocks on the reply
      queue, which the latency thread fills with the report string.
      FIFO ordering guarantees every record posted before the call is
      folded into the response.
  - Consumer: the Latency thread.
  - The queue is private to `texmo/latency.py`; the public surface
    stays `Timer` / `timer(name)` / `get_report()` / `report()`.

## Shutdown

`/stop` (internal port only) spawns a one-shot thread that calls
`shutdown()` on both werkzeug servers — doing it from inside the
request thread would deadlock. `serve()` then falls through to
`SearchServer.join()`, which stops the producers before the consumer:
`Stop` to `requests_queue` and join SearchThread, then `train_queue`
and ModelThread, then `write_queue` and WriterThread, so nothing is
still posting when the writer drains.

## Shared mutable objects

A small number of objects are necessarily shared across threads. Each
is encapsulated behind a narrow interface.

- **`TrainTimingModel`** (single shared instance)
  - Already keyed by `(system, precision) -> Weights` internally; each
    refit just does a dict insert for one key. Dict inserts are atomic
    in CPython, so readers never see a partially-written entry.
  - Writer (Model thread only): `fit(...)` / `_refit_pair(...)`.
  - Readers (Search): `predict`, `predict_batch`, `predict_step_time`,
    `predict_max_steps`, `has_weights` — each returns `None` (or
    False) if the requested `(system, precision)` pair has no weights
    yet.
  - Created in `server.py` at startup and passed by reference to the
    Model thread and the Search thread. If a compatible persisted
    model loads from disk, *that* instance becomes the shared one —
    swapped in before any thread starts.
  - `timing.featurize` is a module-level `@functools.cache`, so it is
    shared by every thread that predicts or fits. `Configuration` is
    immutable and the cache is a plain dict, so a race at worst
    duplicates work.

- **`LossModelHolder`** (lives next to `LossModel` in
  `predict/loss_rnn.py`)
  - Atomically-swappable holder for a `LossModel`. Refits replace the
    whole model wholesale, so the holder just owns one pointer and
    exposes `predict` that delegates to the current model.
  - Writer: the Model thread only (`_refit_loss`, reached from the
    `RunAdded` cadence or a startup `LossRefit`). A second writer
    existed while refits ran on clients — the `/submit_loss_model`
    handler, arbitrating by `run_count` under a lock — and went away
    with that path.
  - Readers (Search): `is_ready()` for the gate, `predict(confs)` for
    the actual call.
  - Implementation: one attribute write swaps the pointer; atomic in
    CPython, no lock needed. Readers may briefly see a stale-but-
    consistent snapshot.

- **Latency** (the dedicated thread above)
  - `_measures` is owned by the thread; never touched directly.
  - The module exposes a `Timer` context manager (posts a `_Record`
    on `__exit__`) and `get_report()` / `report()` (post a `_Report`
    with an ephemeral reply queue and wait on the reply).

- **`SearchServer.template` / `default` / `templates` / `train_time`**
  are rebound wholesale by `/update` under no lock, then handed to the
  Search thread as a `SetTemplate` message. The objects themselves are
  never mutated in place, so a concurrent reader sees either the old
  or the new one; the server-side copies exist only for form rendering
  and `/compare`, and the search thread's copies may lag by one queue
  step.

## Database access (`texmo/db/`)

`ResultDB` is split into two classes with disjoint method sets so the
type system enforces the read/write boundary. Code layout:

- `texmo/db/common.py` — shared helpers: regex bridge, ndarray
  packing, template->SQL conditions, the one shared SQL constant
  (`FIND_CONF`), and the connection bootstrap (`open_connection`).
- `texmo/db/reader.py` — `DbReader` class, plus `ConfScore` /
  `ConfWithRuns` (only produced by reads) and read-only SQL
  constants.
- `texmo/db/writer.py` — `DbWriter`, `DbWriterProxy`, `WriterThread`
  and the write-side SQL constants.
- `texmo/db/schema.sql` — the schema (used to bootstrap the writer
  connection on a fresh DB).

Every connection is opened on the thread that uses it and inherits
sqlite's default `check_same_thread=True`. File-backed writer
connections enable WAL and a 30 s `busy_timeout`.

- **`DbReader`**
  - Constructor opens the SQLite file with `mode=ro` (URI form) and
    asserts a real path — sqlite refuses `mode=ro` on `:memory:`.
  - Each thread that needs DB reads instantiates one — per-request
    for report handlers, persistent for the Search and Model threads.
  - Methods (read-only): `get_conf_id`, `get_run_counts`,
    `get_systems`, `has_covering_run`, `top_confs_global`,
    `pick_me_conf`, `fastest_near_best_segments`,
    `fastest_near_best_segments_any_system`, `top_confs_for_system`,
    `best_conf_for_spec_on_system`, `confs_under_time`, `total_runs`,
    `get_confs_runs`, `get_runs_for_timing`, `iter_labeled_runs`,
    `iter_labeled_runs_raw`, `iter_confs_by_precision`,
    `iter_confs_missing_estimate`, `get_conf_ids_with_median_time`,
    `get_losses_by_conf_ids`, `get_time_estimate`.
  - Internal state: positive-only `_conf_id_cache` (Configuration ->
    conf_id), populated lazily on disk hits.

- **`DbWriter`**
  - Constructor opens the SQLite file read-write, applying the schema
    if the DB is fresh. In the running server `WriterThread` owns the
    only post-init instance; `SearchServer.__init__` opens a
    throwaway one on the main thread purely to apply that schema
    before the first read. Single-threaded callers (the `cli/db.py`
    admin commands, tests) use it directly.
  - Methods (write-only, but may read as part of a write):
    `add_run` (still runs `track_winner_change` inline — it owns the
    transaction, and just doesn't return the bool to the caller),
    `find_or_add_conf`, `add_pick_me_conf`, `upsert_time_estimates`,
    `upsert_predicted_time_estimates`, `update_all_scores`,
    `backfill_num_layers`, `clear_system`.
  - `upsert_predicted_time_estimates` uses an
    `ON CONFLICT ... WHERE source = 'predicted'` guard so a `median`
    row written concurrently by `_add_run` is never clobbered.
  - Internal state: `_conf_id_cache` (Configuration -> conf_id,
    populated by `find_or_add_conf`) and all write-transaction logic
    (`BEGIN IMMEDIATE` / `COMMIT`).

The fitted predictors (timing, loss) are **not** in the DB. They are
pickles next to the DB file, written atomically with rotation by
`texmo/predict/persist.py` (`save_predictor` / `load_predictor`).

## Outstanding

- **No I/O-error panic path on WriterThread.** An unrecoverable write
  error propagates out of `run()`, closes the connection and kills the
  thread silently; producers keep posting into a queue nobody drains.
  The intent is to log the failure and trigger the same clean-shutdown
  path as the `/stop` button so producers fail fast.
- **A single Search thread.** The design supports N (`requests_queue`
  is a shared work queue and any consumer can serve any message; each
  would need its own persistent `DbReader` plus references to the
  shared `TrainTimingModel` / `LossModelHolder`), but only one is
  created today, in `SearchServer.__init__` rather than in
  `server.main()`.
- **`/select` doesn't prefetch.** Each request posts its own `Select`
  and then blocks on the response queue, so the one extra seed per
  system is the whole pipeline depth. An earlier design had Search
  self-replenish after every production; that is not what the code
  does.

## History

This file began as a plan for a refactor that moved every DB write
onto a dedicated thread and made Search stateless. The refactor landed
in 2026-07 and the document was rewritten as a description of the
result. Along the way:

1. Search and the Model thread came to share one `TrainTimingModel`
   and one `LossModelHolder` instead of pushing weights through the
   search queue.
2. The refit counters moved off Search onto `ModelThread`, and
   `timing_queue` became `train_queue` carrying `RunAdded` /
   `LossRefit` / `Stop`.
3. `model._cache` and `Template._conf_neighbors_cache` were removed
   once measurement showed `conf_neighbors` averages ~1 ms/call
   without them.
4. Every multi-payload queue was standardised on per-command
   dataclasses with `match` dispatch.
5. `ResultDB` was split into `DbReader` / `DbWriter` under
   `texmo/db/`, with the boundary type-enforced.
6. `WriterThread` + `DbWriterProxy` took over all writes; SearchThread
   became read-only and the `/add` handler enqueues `AddRun` +
   `RunAdded` itself.
7. `latency._measures` moved onto its own daemon thread behind an
   unchanged public surface.

`track_winner_change` stayed inside the writer so the (before, after)
sampling is atomic with the run insert. The result is recorded in the
`changed_winner` column; the caller doesn't see it.
