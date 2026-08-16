"""Read-write access to the results DB.

`DbWriter` opens the SQLite file read-write and owns all
write-transaction state. Reads happen inside the writer only as
part of a write transaction (e.g. `_top_conf_at` for the
`changed_winner` sample, or `_find_or_add_conf` to dedupe before
inserting); the public surface is write-only.

In the running server `WriterThread` (defined below) is the sole
owner of the post-init `DbWriter`. It opens the connection on its
own thread (so `check_same_thread=True` works) and consumes write
requests from a `Queue` populated by SearchThread, ModelThread, and
SearchServer's startup bootstrap. Background producers post via
`DbWriterProxy`, a tiny wrapper that has the same write methods as
`DbWriter` but turns each call into a `WriteMessage` on the queue.

Single-process / single-thread callers (the `cli/db.py` admin
commands, the bootstrap-on-init path on the main thread, tests)
construct `DbWriter` directly; they're already single-threaded so
they don't need the queue indirection.
"""

import logging
import math
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from statistics import StatisticsError, median
from typing import Optional

from .. import latency
from ..common import INF
from ..configuration import Configuration
from ..precision import Precision
from ..run import Run
from ..spec_parser import parse_model2
from .common import (
    FIND_CONF,
    _pack_ndarray,
    open_connection,
)


# --- SQL constants (write-only side) ----------------------------------------


INSERT_CONF = """
INSERT INTO conf (spec, weights, lr, length, batch, steps, precision,
                  decay, cosine, num_layers)
VALUES (:spec, :weights, :lr, :length, :batch, :steps, :precision,
        :decay, :cosine, :num_layers)
"""

INSERT_RUN = """
INSERT INTO run(conf_id, system, train_time, timestamp, loss,
                loss_model_v, loss_model, strategy, changed_winner)
VALUES (:conf_id, :system, :train_time, :timestamp, :loss,
        :loss_model_v, :loss_model, :strategy, :changed_winner)
"""

# Best (lowest-median_score) conf under (system, max_weights, max_time).
# Used by add_run to tell if a new run changed the winner at that point.
#
# EXISTS form: walk conf by `conf_median_score` index in ASC order
# and check the cte UNIQUE(conf_id, system) for each. The previous
# JOIN form forced SQLite to either SCAN the 3M-row cte (when no
# cte system index existed) or filter-then-sort the cte slice for
# this system. EXISTS lets the planner stop at the first qualifying
# conf -- microseconds vs 165-650 ms on the live DB.
# INDEXED BY conf_median_score forces the planner to walk confs in
# median_score order and stop at the first match (LIMIT 1), instead of
# its default "scan every conf under the weight cap -> temp-B-tree sort",
# which touches a large fraction of the 400k-conf table every call (~1.5s
# on the prod DB). The walk early-exits in ms-to-100ms on an established
# active system. Always called with the worker's (active) system, where
# the best-scoring conf usually has an estimate -> a shallow walk.
TOP_CONF_AT = """
SELECT conf.id AS conf_id
FROM conf INDEXED BY conf_median_score
WHERE conf.median_score IS NOT NULL
  AND conf.weights <= :max_weights
  AND EXISTS (
      SELECT 1 FROM conf_time_estimate cte
      WHERE cte.conf_id = conf.id
        AND cte.system = :system
        AND cte.time_s <= :max_time
  )
ORDER BY conf.median_score ASC
LIMIT 1
"""

GET_TRAIN_TIMES = """
SELECT train_time
FROM run
WHERE conf_id = :conf_id
  AND system = :system
"""

UPSERT_TIME_ESTIMATE = """
INSERT OR REPLACE INTO conf_time_estimate(conf_id, system, time_s, source)
VALUES (:conf_id, :system, :time_s, :source)
"""

# Bulk-write 'predicted' estimates without clobbering existing 'median'
# rows. A 'median' is the source of truth when it exists; an
# unguarded REPLACE racing against a concurrent _update_median_time
# write would erase it, leaving the conf timeless on the UI.
UPSERT_PREDICTED_ESTIMATE = """
INSERT INTO conf_time_estimate(conf_id, system, time_s, source)
VALUES (:conf_id, :system, :time_s, 'predicted')
ON CONFLICT(conf_id, system) DO UPDATE
  SET time_s = excluded.time_s
  WHERE conf_time_estimate.source = 'predicted'
"""

class DbWriter(object):
    """Read-write handle to the results DB. Owns write transactions."""

    @staticmethod
    def from_args(db: Optional[str]) -> 'DbWriter':
        return DbWriter(db)

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = ':memory:'
        self._path = path
        self._db = open_connection(path, readonly=False)

    @property
    def path(self) -> str:
        return self._path

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # --- conf upsert --------------------------------------------------------

    def _find_or_add_conf(
        self, cur: sqlite3.Cursor, conf: Configuration
    ) -> int:
        conf_dict = conf.to_dict()
        cur.execute(FIND_CONF, conf_dict)
        row = cur.fetchone()
        if row is not None:
            return row[0]
        conf_dict['weights'] = conf.num_weights
        conf_dict['num_layers'] = conf.model.num_layers
        cur.execute(INSERT_CONF, conf_dict)
        return cur.lastrowid

    def find_or_add_conf(self, conf: Configuration) -> int:
        """Find the conf in the db (or insert), returning the conf_id."""
        with latency.timer('DbWriter.find_or_add_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN IMMEDIATE')
            conf_id = self._find_or_add_conf(cur, conf)
            cur.execute('COMMIT')
            return conf_id

    def add_pick_me_conf(
        self, conf: Configuration,
    ) -> tuple[int, bool]:
        """Insert `conf` with `pick_me = 1` if it isn't already in the
        DB; otherwise leave the existing row's pick_me flag unchanged
        (the conf has its own measurement history and we don't want
        the migration to re-flag a conf we've already characterized).

        Returns `(conf_id, was_inserted)`.
        """
        with latency.timer('DbWriter.add_pick_me_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN IMMEDIATE')
            conf_dict = conf.to_dict()
            cur.execute(FIND_CONF, conf_dict)
            row = cur.fetchone()
            if row is not None:
                cur.execute('COMMIT')
                return row[0], False
            conf_dict['weights'] = conf.model.num_weights
            conf_dict['num_layers'] = conf.model.num_layers
            cur.execute(
                'INSERT INTO conf '
                '(spec, weights, lr, length, batch, steps, precision,'
                ' decay, cosine, num_layers, pick_me) '
                'VALUES '
                '(:spec, :weights, :lr, :length, :batch, :steps,'
                ' :precision, :decay, :cosine, :num_layers, 1)',
                conf_dict,
            )
            conf_id = cur.lastrowid
            cur.execute('COMMIT')
            return conf_id, True

    # --- run insertion ------------------------------------------------------

    def _add_run_execute(self, cur: sqlite3.Cursor, run_dict: dict):
        with latency.timer('DbWriter._add_run_execute'):
            cur.execute(INSERT_RUN, run_dict)

    def _update_median_score(self, cur: sqlite3.Cursor, conf_id: int):
        with latency.timer('DbWriter._update_median_score'):
            cur.execute('SELECT loss FROM run WHERE conf_id = :conf_id',
                        {'conf_id': conf_id})
            try:
                median_score = median(row[0] for row in cur)
            except StatisticsError:
                median_score = None
            cur.execute('UPDATE conf SET median_score = :median_score WHERE id = :conf_id',
                        {'median_score': median_score, 'conf_id': conf_id})

    def _update_median_time(
        self, cur: sqlite3.Cursor, conf_id: int, system: str
    ):
        with latency.timer('DbWriter._update_median_time'):
            cur.execute(GET_TRAIN_TIMES, {'conf_id': conf_id, 'system': system})
            try:
                median_time = median(row[0] for row in cur if row[0] > 0.0001)
            except StatisticsError:
                median_time = None
            if median_time is not None and median_time > 0.0001:
                # The median-based estimate supersedes any prediction
                # previously stored for this (conf, system).
                assert median_time > 0, (
                    f"_update_median_time about to write non-positive "
                    f"time_s={median_time} for conf={conf_id} on {system}"
                )
                cur.execute(UPSERT_TIME_ESTIMATE, {
                    'conf_id': conf_id,
                    'system': system,
                    'time_s': median_time,
                    'source': 'median',
                })
                # Same-transaction read-back: if we just wrote a 'median'
                # row, it must be visible here. A read failure here means
                # an earlier statement on this cursor has been silently
                # rolled back, or someone else's REPLACE clobbered it
                # mid-transaction (which shouldn't be possible).
                cur.execute(
                    "SELECT source FROM conf_time_estimate "
                    "WHERE conf_id = :conf_id AND system = :system",
                    {'conf_id': conf_id, 'system': system},
                )
                row = cur.fetchone()
                assert row is not None and row[0] == 'median', (
                    f"_update_median_time wrote 'median' for conf={conf_id} "
                    f"on {system} but read-back returned {row!r}"
                )

    def _update_scores(self, cur: sqlite3.Cursor, conf_id: int, system: str):
        self._update_median_score(cur, conf_id)
        self._update_median_time(cur, conf_id, system)

    def _top_conf_at(
        self, cur: sqlite3.Cursor,
        system: str, max_weights: int, max_time: float,
    ) -> Optional[int]:
        cur.execute(TOP_CONF_AT, {
            'system': system,
            'max_weights': max_weights,
            'max_time': max_time,
        })
        row = cur.fetchone()
        return row[0] if row is not None else None

    def _add_run(
        self,
        conf: Configuration,
        run: Run,
        conf_id: Optional[int],
        timestamp: Optional[datetime],
        strategy: Optional[str],
        track_winner_change: bool,
    ) -> Optional[bool]:
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')

        if conf_id is None:
            conf_id = self._find_or_add_conf(cur, conf)

        if run.loss_trend is None:
            loss_model_v = None
            loss_model = None
        else:
            loss_model_v = run.loss_trend.version
            loss_model = _pack_ndarray(run.loss_trend.params())

        timestamp = timestamp.isoformat() if timestamp else None
        loss = INF if math.isnan(run.loss) or run.loss is None else run.loss

        # Sample the winner at (system, run.train_time, conf.weights)
        # before and after writing the run, using the same (T, W) both
        # times. A flip means this run changed the winner for some
        # (t, w). Diverged runs (train_time None) carry no usable
        # time, so skip tracking for those.
        do_track = (
            track_winner_change
            and run.train_time is not None
            and run.train_time > 0.0001
        )
        before: Optional[int] = None
        if do_track:
            before = self._top_conf_at(
                cur, run.system, conf.num_weights, run.train_time)

        run_dict = {
            'conf_id': conf_id,
            'system': run.system,
            # 0 is the schema's "no usable time" sentinel; reads should
            # convert back to None. See schema.sql for details.
            'train_time': run.train_time or 0,
            'timestamp': timestamp,
            'loss': loss,
            # The per-step loss history is deliberately NOT stored: it
            # dominated the DB size and went unused for months. The
            # fitted trend params (loss_model, 3 floats) reconstruct
            # the LossTrend without it.
            'loss_model_v': loss_model_v,
            'loss_model': loss_model,
            'strategy': strategy,
            'changed_winner': None,  # filled in below if tracked
        }

        self._add_run_execute(cur, run_dict)
        run_id = cur.lastrowid
        self._update_scores(cur, conf_id, run.system)

        changed_winner: Optional[int] = None
        if do_track:
            after = self._top_conf_at(
                cur, run.system, conf.num_weights, run.train_time)
            changed_winner = 1 if before != after else 0
            cur.execute(
                'UPDATE run SET changed_winner = :cw WHERE id = :id',
                {'cw': changed_winner, 'id': run_id},
            )

        cur.execute('COMMIT')
        return None if changed_winner is None else bool(changed_winner)

    def add_run(
        self,
        conf: Configuration,
        run: Run,
        conf_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        strategy: Optional[str] = None,
        track_winner_change: bool = False,
    ) -> Optional[bool]:
        """Insert a run, optionally flagging whether it changed the winner.

        `strategy` is the label of the search strategy that picked the
        conf. With `track_winner_change`, the run's `changed_winner`
        column is set to 1/0 by sampling the winning conf at
        (system, train_time, conf.num_weights) before and after the
        insert, inside the same transaction. Returns the resulting
        bool (or None if not tracked / train_time unusable).
        """
        # Diverged training produces `run.train_time = None`; we
        # translate that to the schema's 0 sentinel below. Anything
        # else must be a real, positive measurement -- catching
        # accidental near-zero values that would otherwise pollute
        # the medians.
        assert run.train_time is None or run.train_time > 0.0001, (
            f"add_run got run.train_time={run.train_time!r} "
            f"(system={run.system!r}, conf={conf})"
        )
        with latency.timer('DbWriter.add_run'):
            return self._add_run(
                conf, run, conf_id, timestamp,
                strategy, track_winner_change,
            )

    # --- bulk maintenance ---------------------------------------------------

    def update_all_scores(self):
        """Recompute median_score and median_time for every (conf, system).

        Used at bootstrap to backfill rows that pre-date per-run
        maintenance in `add_run`.
        """
        with latency.timer('DbWriter.update_all_scores'):
            cur = self._db.cursor()
            cur.execute(
                'SELECT DISTINCT conf.id AS conf_id, system FROM conf, run '
                'WHERE conf.id = run.conf_id'
            )
            for row in cur.fetchall():
                self._update_scores(cur, row['conf_id'], row['system'])
            self._db.commit()

    def backfill_num_layers(self, batch_size: int = 1000) -> int:
        """Populate `num_layers` for every conf row where it's NULL.

        Used to migrate existing databases after the `num_layers`
        column was added. Returns the number of rows updated. Logs
        progress every batch.
        """
        cur = self._db.cursor()
        cur.execute("SELECT COUNT(*) FROM conf WHERE num_layers IS NULL")
        total = cur.fetchone()[0]
        if total == 0:
            return 0
        logging.info(
            f"Backfilling num_layers for {total} confs...")

        updated = 0
        while True:
            cur.execute(
                "SELECT id, spec FROM conf "
                "WHERE num_layers IS NULL LIMIT :n",
                {'n': batch_size},
            )
            rows = cur.fetchall()
            if not rows:
                break
            updates = []
            for row in rows:
                try:
                    model = parse_model2(
                        row['spec'], precision=Precision.FP32)
                    updates.append((model.num_layers, row['id']))
                except Exception as exc:
                    logging.warning(
                        f"skipping conf id={row['id']} "
                        f"spec={row['spec']!r}: {exc}")
            cur.execute('BEGIN IMMEDIATE')
            cur.executemany(
                "UPDATE conf SET num_layers = ? WHERE id = ?", updates)
            cur.execute('COMMIT')
            updated += len(updates)
            logging.info(f"  {updated}/{total} done")
        return updated

    def clear_system(self, system: str) -> int:
        """Delete all runs and time estimates for a given system.

        Recomputes median_score for any affected configurations.
        Returns the number of runs that were deleted.
        """
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')

        # Find affected confs before deleting the runs.
        cur.execute(
            'SELECT DISTINCT conf_id FROM run WHERE system = :system',
            {'system': system},
        )
        affected_conf_ids = [row[0] for row in cur.fetchall()]

        cur.execute(
            'DELETE FROM run WHERE system = :system',
            {'system': system},
        )
        deleted = cur.rowcount

        cur.execute(
            'DELETE FROM conf_time_estimate WHERE system = :system',
            {'system': system},
        )

        # Recompute median_score for affected confs (estimates for this
        # system are already gone).
        for conf_id in affected_conf_ids:
            self._update_median_score(cur, conf_id)

        cur.execute('COMMIT')
        return deleted

    # --- bulk upserts -------------------------------------------------------

    def upsert_time_estimates(
        self, rows: Iterable[tuple[int, str, float, str]]
    ):
        """Bulk insert/replace estimates. `rows` is (conf_id, system, time_s, source)."""
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')
        cur.executemany(UPSERT_TIME_ESTIMATE, [
            {'conf_id': cid, 'system': s, 'time_s': t, 'source': src}
            for (cid, s, t, src) in rows
        ])
        cur.execute('COMMIT')

    def upsert_predicted_time_estimates(
        self, rows: Iterable[tuple[int, str, float]]
    ):
        """Bulk-write 'predicted' estimates without overwriting 'median' rows.

        `rows` is (conf_id, system, time_s). Use this from the Model
        thread; an unguarded `upsert_time_estimates` would race with
        `_add_run`'s median write and erase the truth.
        """
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')
        cur.executemany(UPSERT_PREDICTED_ESTIMATE, [
            {'conf_id': cid, 'system': s, 'time_s': t}
            for (cid, s, t) in rows
        ])
        cur.execute('COMMIT')

# --- WriterThread + queued writes -------------------------------------------


@dataclass
class AddRun:
    conf: Configuration
    run: Run
    strategy: Optional[str] = None
    track_winner_change: bool = False


@dataclass
class PredictedTimeRow:
    conf_id: int
    system: str
    time_s: float


@dataclass
class UpsertPredictedTimeEstimates:
    rows: list[PredictedTimeRow]


@dataclass
class UpdateAllScores:
    pass


@dataclass
class Stop:
    pass


WriteMessage = (
    AddRun
    | UpsertPredictedTimeEstimates
    | UpdateAllScores
    | Stop
)


class DbWriterProxy:
    """Drop-in for `DbWriter` that posts each call as a `WriteMessage`.

    Background producers (Search/Model threads) hold one of these and
    call the same write methods they used to call on `DbWriter`; the
    actual SQL runs on `WriterThread`. Writes are fire-and-forget —
    `add_run`'s `changed_winner` return value is dropped.
    """

    def __init__(self, queue: Queue):
        self._queue = queue

    def add_run(
        self,
        conf: Configuration,
        run: Run,
        strategy: Optional[str] = None,
        track_winner_change: bool = False,
    ) -> None:
        self._queue.put(AddRun(
            conf=conf, run=run,
            strategy=strategy,
            track_winner_change=track_winner_change,
        ))

    # A full estimate refresh posts ~350k rows; split into chunks so
    # AddRun messages interleave between the writer's transactions
    # instead of waiting behind one giant upsert.
    _UPSERT_CHUNK = 20_000

    def upsert_predicted_time_estimates(
        self, rows: Iterable[tuple[int, str, float]]
    ) -> None:
        # Materialize before queueing — the producer's iterator may
        # close before the WriterThread gets to it.
        rows = [
            PredictedTimeRow(conf_id=cid, system=s, time_s=t)
            for (cid, s, t) in rows
        ]
        for i in range(0, len(rows), self._UPSERT_CHUNK):
            self._queue.put(UpsertPredictedTimeEstimates(
                rows[i:i + self._UPSERT_CHUNK]))

    def update_all_scores(self) -> None:
        self._queue.put(UpdateAllScores())


class WriterThread(threading.Thread):
    """Owns the only post-init writable connection in the server.

    Opens its `DbWriter` on this thread (so `check_same_thread=True`
    holds) and serializes all writes from background producers.

    A write that raises is **fatal**: there is no retry tier, because
    the connection's 30 s `busy_timeout` already absorbs transient
    contention, so what reaches us here (disk I/O error, disk full, a
    lock that outlived the timeout) is unrecoverable. Dying quietly
    is the worst outcome -- the server would keep answering reads and
    acknowledging `/add` while every write piled up in a queue nobody
    drains -- so we log CRITICAL and call `on_fatal`, whose job is to
    bring the server down and make producers fail loudly. Without a
    callback (bare constructions in tests, admin tools) the CRITICAL
    log is the whole signal and the thread still exits.
    """

    def __init__(
        self,
        db_path: Optional[str],
        queue: Queue,
        on_fatal: Optional[Callable[[], None]] = None,
    ):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        self._on_fatal = on_fatal

    def run(self):
        writer = DbWriter(self._db_path)
        logging.info("Started writer thread")
        try:
            while True:
                m = self._queue.get()
                try:
                    match m:
                        case AddRun(
                            conf=conf, run=run, strategy=strategy,
                            track_winner_change=track,
                        ):
                            writer.add_run(
                                conf, run,
                                strategy=strategy,
                                track_winner_change=track,
                            )
                        case UpsertPredictedTimeEstimates(rows=rows):
                            writer.upsert_predicted_time_estimates(
                                (r.conf_id, r.system, r.time_s)
                                for r in rows)
                        case UpdateAllScores():
                            writer.update_all_scores()
                        case Stop():
                            logging.info("Stopping writer thread")
                            break
                        case _:
                            assert False, f"Unknown write message: {m!r}"
                except Exception as e:
                    # Includes the unknown-message assert: reaching it
                    # means a producer is posting something no one can
                    # write, which is just as unrecoverable.
                    logging.critical(
                        f"Writer thread died on {type(m).__name__}: "
                        f"{e!r}. Writes are no longer being drained; "
                        f"shutting down.", exc_info=True)
                    self._panic()
                    break
        finally:
            writer.close()

    def _panic(self):
        """Hand the failure to the owner (normally SearchServer).

        Runs ON this thread, and the shutdown sequence it triggers
        joins this thread -- so the callback must not block on that
        join. `SearchServer` satisfies this by spawning the actual
        shutdown on a one-shot thread; keep any other implementation
        non-blocking too. A callback that itself raises must not
        strand the connection, hence the guard: `run`'s `finally`
        still closes the writer either way."""
        if self._on_fatal is None:
            return
        try:
            self._on_fatal()
        except Exception:
            logging.critical(
                "Writer thread's fatal callback raised", exc_info=True)
