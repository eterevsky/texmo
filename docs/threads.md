# Threading model (planned refactor)

Working document for the in-flight refactor that moves all DB writes
onto a single dedicated thread and lets search become stateless.
This file describes the *target* shape; the code is mid-transition.

## Rule

> Threads share only **read-only** objects. Anything that needs to be
> mutated by another thread is sent through a **Queue**.

The few legitimately shared-mutable objects (`TrainTimingModel`,
`LossModelHolder`, `latency` measures) are encapsulated behind narrow
interfaces with clearly identified writer threads.

## Message format

Every queue that carries more than one kind of payload uses a single
`Message` dataclass so consumers can `match m.command:` without
worrying about tuple unpacking shapes:

```python
@dataclass
class Message:
    command: str
    response_queue: Optional[Queue] = None  # set for RPC-style replies
    # everything else: additional named fields per command
```

The simple object-only queues (`confs_by_system[system]`, which just
carries a `SearchResult | None`) are exempt — they don't need the
wrapper.

## Threads

- **Werkzeug request handlers** (internal :5000, external :5001)
  - Count: many (one per request).
  - Owns: nothing persistent.
  - Reads from: a per-request `DbReader`, only for the report-style
    routes (`/`, `/compare`, `/throughput`, `/latency`). The hot
    paths (`/select`, `/add`) just enqueue messages and never touch
    the DB directly.
  - Sends to:
    - `DBWriter` via `write_queue` to record incoming runs (`/add`).
    - `Search` via `select_queue`, but only the first time a new
      system is seen — to seed N initial `select` messages. After
      that the handler just drains `confs_by_system[system]`;
      Search itself keeps the queue primed (see below).
    - `Model` via `train_queue` to notify it of newly added runs
      so it can update its own counters and decide on refits.

- **DBWriter** (new)
  - Count: 1.
  - Owns: the single `DbWriter` instance (read-write connection) and
    all write-transaction state.
  - Reads from: its `write_queue`.
  - Sends to: nothing on the happy path. On unrecoverable I/O error
    it logs the failure and triggers the same clean-shutdown path as
    the `/stop` button (drain queues, refuse further messages, join
    threads) so other producers fail fast rather than wedge forever.

- **Search**
  - Count: N (≥ 1; 1 for now, multiple later).
  - Owns: its own persistent `DbReader`.
  - Reads from:
    - `DbReader` for query work.
    - The shared `TrainTimingModel` and `LossModelHolder` instances
      (writer is the Model thread; see "Shared mutable objects").
    - `select_queue` for incoming requests.
  - Sends to:
    - The matching per-system queue in `confs_by_system`, dropping
      each produced `SearchResult` onto it.
    - `select_queue`, putting back one `select` for the same system
      after each one it consumes. This is what keeps the prefetch
      pipeline primed — Search never needs to know about the set of
      known systems; it just refills the one it just produced for.

- **Model**
  - Count: 1.
  - Owns (mutates): the shared `TrainTimingModel` (per-(system,
    precision) weight inserts; one shared instance is reused across
    threads), and the underlying `LossModel` slot inside the shared
    `LossModelHolder` (replaced wholesale on refit).
  - Owns (private): per-(system, precision) refit counters, the
    loss-refit counter, and the `featurize()` `@functools.cache`
    (purely local to this thread by construction).
  - Reads from: its own `DbReader` for fitting runs, plus
    `train_queue` for incoming "run added" notifications.
  - Sends to: `DBWriter` via `write_queue` to persist predicted-time
    estimates and model snapshots.

- **Latency** (new dedicated thread; see "Shared mutable objects")
  - Count: 1.
  - Owns: the `_measures` dict.
  - Reads from: its `latency_queue`.

The two werkzeug listener threads (`internal_srv.serve_forever` and
`external_srv.serve_forever`) are infrastructure, not application
threads — they dispatch into the request-handler thread pool.

## Queues

- `write_queue`
  - Producers: request handlers (`/add`), Model (predicted-time
    bulk upserts, snapshot persistence).
  - Consumer: DBWriter.
  - Commands:
    - `add_run` (fields: `conf`, `run`, `strategy`)
    - `upsert_predicted_estimates` (fields: `system`, `rows`)
    - `save_model` (fields: `name`, `blob`)
  - Search does **not** write. Anything bookkeeping-adjacent that
    needs to land in the DB is either piggy-backed on `/add` by the
    handler or routed via Model.

- `select_queue`
  - Producers:
    - Request handlers (`/select`), **only** the first time a new
      system is seen, to seed N initial `select` messages.
    - Search threads themselves — after producing a `SearchResult`
      for system `S`, the Search thread puts one more
      `select(system=S)` back on the queue. This is what keeps the
      per-system prefetch primed without anyone tracking the set of
      known systems.
  - Consumers: the N Search threads (any one picks up a message).
  - Commands:
    - `select` (fields: `system`)

- `confs_by_system: dict[str, Queue[SearchResult | None]]`
  - Producer: Search threads, one item per `select` command processed.
  - Consumer: `/select` handlers, which just `get()` from the
    target system's queue. They do **not** enqueue refills —
    Search self-replenishes after every production.
  - Lock: `confs_by_system_lock` protects creation of the per-system
    queue + initial N-seed (the first time a new system appears).

- `train_queue`
  - Producers: request handlers (`/add`), after they enqueue the
    write.
  - Consumer: Model.
  - Commands:
    - `run_added` (fields: `system`, `precision`) — Model owns its
      own counters and decides when to fit.
  - DBWriter does **not** know about training. The handler that
    posted the write is the same one that notifies the Model
    thread; the business logic stays out of the writer.

- `latency_queue`
  - Producers: every `Timer.__exit__`, from any thread.
  - Consumer: the Latency thread.
  - Commands:
    - `record` (fields: `name`, `elapsed_seconds`)
    - `report` (fields: none; uses `response_queue` to reply with
      aggregated percentiles/averages)

## Shared mutable objects

A small number of objects are necessarily shared across threads. Each
is encapsulated behind a narrow interface with one designated writer.

- **`TrainTimingModel`** (single shared instance)
  - Already keyed by `(system, precision) -> Weights` internally; each
    refit just does a dict insert for one key. Dict inserts are atomic
    in CPython, so readers never see a partially-written entry.
  - Writer (Model thread only): `fit(...)` / `_refit_pair(...)`.
  - Readers (Search): `predict`, `predict_batch`, `predict_step_time`,
    `predict_max_steps` — each returns `None` if the requested
    `(system, precision)` pair has no weights yet.
  - The instance is created in `server.py` at startup and passed by
    reference to the Model thread and to every Search thread.

- **`LossModelHolder`** (lives next to `LossModel` in `predict/loss_rnn.py`)
  - Atomically-swappable holder for a `LossModel`. Refits replace the
    whole model wholesale, so the holder just owns one pointer and
    exposes `predict` that delegates to the current model.
  - Writer (Model thread only): `set_model(loss_model)`.
  - Readers (Search): `is_ready()` for the gate, `predict(confs)` for
    the actual call.
  - Implementation: one attribute write swaps the pointer; atomic in
    CPython, no lock needed. Readers may briefly see a stale-but-
    consistent snapshot.

- **Latency** (the dedicated thread above)
  - `_measures` is owned by the thread; never touched directly.
  - The module exposes a `Timer` context manager (puts on the
    queue) and a `get_report()` function (sends a `report` message
    with an ephemeral response queue and waits on the reply).

The `SearchServer.template` / `default` / `train_time` objects are
built once at frontend startup and are immutable thereafter, so they
are plain read-only references handed to the Search threads — not
shared mutable state.

## ResultDB split

`ResultDB` becomes two classes with disjoint method sets so the type
system enforces the read/write boundary. Code layout:

- `texmo/db_common.py` — shared helpers: SQL constant strings, row
  factories, connection plumbing, `Configuration.from_dict`-style
  serialisation, the small `_train_time_from_row` family.
- `texmo/db_reader.py` — `DbReader` class.
- `texmo/db_writer.py` — `DbWriter` class.

- **`DbReader`**
  - Constructor opens the SQLite file with `mode=ro` (URI form).
  - Each thread that needs DB reads instantiates one — per-request
    for report handlers, persistent for Search and Model threads.
  - Methods (read-only):
    - `get_systems`, `get_conf_id`, `get_run_counts`,
      `top_confs_global`, `top_confs_for_system`,
      `fastest_near_best_segments`, `confs_under_time`,
      `best_conf_for_spec_on_system`, `iter_confs_by_precision`,
      `get_conf_ids_with_median_time`, `get_losses_by_conf_ids`,
      `get_time_estimate`, `get_runs_for_timing`,
      `get_confs_runs`, `iter_labeled_runs`, `get_run_count`,
      `load_model`.

- **`DbWriter`**
  - Constructor opens the SQLite file read-write. Exactly one
    instance per process, lives on the DBWriter thread.
  - Methods (write-only, but may read as part of a write):
    - `add_run` (still runs `track_winner_change` inline — owns the
      transaction; just doesn't return the bool to the caller),
    - `find_or_add_conf`,
    - `upsert_predicted_time_estimates`,
    - `upsert_time_estimates`,
    - `save_model`, `clear_system`, `update_all_scores`.
  - Internal state:
    - `_conf_id_cache` (Configuration -> conf_id, populated by
      `find_or_add_conf`).
    - All write-transaction logic (`BEGIN IMMEDIATE` / `COMMIT`).

## Migration plan

Land in this order so each step is independently verifiable:

1. ~~**Introduce shared model instances.**~~ **done**: Search and the
   Model thread now share one `TrainTimingModel` (mutated in place via
   atomic per-key dict inserts) and one `LossModelHolder` (a tiny
   pointer-swap wrapper around `LossModel`). The Model thread no
   longer pushes `loss_weights` / `timing_weights` messages through
   the search queue.
2. **Move the refit counters off Search.** `_run_counter` and
   `_total_run_counter` move into the Model thread. Add a
   `train_queue` and let request handlers notify the Model thread
   on each `/add`. Model decides on its own when to refit.
3. ~~Cache decisions~~ **done**: `model._cache` and
   `Template._conf_neighbors_cache` were removed once measurement
   showed `conf_neighbors` averages ~1 ms/call without them.
4. **Standardise queue messages on the `Message` dataclass.** Touch
   every existing producer/consumer pair. Mechanical refactor.
5. **Split `ResultDB` into `DbReader` / `DbWriter`** under the
   three new modules. Nothing changes semantically, but it pins
   down the read/write boundary before the next step.
6. **Introduce DBWriter thread and `write_queue`.** The `DbWriter`
   instance moves onto that thread. All other threads queue write
   requests. Drop the SQL-level `_UPSERT_PREDICTED_ESTIMATE`
   ON CONFLICT guard once writes are serialized by construction.
   Wire the I/O-error panic path to the existing `/stop` shutdown.
7. **Move `latency._measures` onto its own thread** with the
   request/response queue protocol described above.
8. **Optional: spawn N Search threads.** Created explicitly in
   `server.main()`, each with its own persistent `DbReader` and
   references to the shared `TrainTimingModel` / `LossModelHolder` /
   `select_queue` / `confs_by_system`.

## Open questions

- The `Message` dataclass uses `command` as the field name; it's
  what the current `command, args = queue.get()` loops use. If a
  better name surfaces during the refactor, swap it then — it's
  one find-and-replace.
- `track_winner_change` stays inside DBWriter so the (before, after)
  sampling is atomic with the run insert. The result is recorded
  in the `changed_winner` column; the caller doesn't see it. No
  user-facing change.
