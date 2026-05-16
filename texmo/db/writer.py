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
import pickle
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from statistics import StatisticsError, median
from typing import Optional

from .. import latency
from ..common import INF
from ..configuration import Configuration
from ..run import Run
from .common import (
    FIND_CONF,
    _pack_ndarray,
    open_connection,
)


# --- SQL constants (write-only side) ----------------------------------------


INSERT_CONF = """
INSERT INTO conf (spec, weights, lr, length, batch, steps, precision,
                  decay, cosine)
VALUES (:spec, :weights, :lr, :length, :batch, :steps, :precision,
        :decay, :cosine)
"""

INSERT_RUN = """
INSERT INTO run(conf_id, system, train_time, timestamp, loss, step_loss,
                loss_model_v, loss_model, strategy, changed_winner)
VALUES (:conf_id, :system, :train_time, :timestamp, :loss, :step_loss,
        :loss_model_v, :loss_model, :strategy, :changed_winner)
"""

# Best (lowest-median_score) conf under (system, max_weights, max_time).
# Used by add_run to tell if a new run changed the winner at that point.
TOP_CONF_AT = """
SELECT conf.id AS conf_id
FROM conf
JOIN conf_time_estimate cte
    ON cte.conf_id = conf.id AND cte.system = :system
WHERE conf.median_score IS NOT NULL
  AND conf.weights <= :max_weights
  AND cte.time_s <= :max_time
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

UPSERT_MODEL = """
INSERT OR REPLACE INTO model(name, data, updated_at)
VALUES (:name, :data, :updated_at)
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
        # Always single-thread: WriterThread opens its instance on its
        # own thread, the CLI / bootstrap / tests are main-thread only.
        # No caller crosses a thread boundary with this connection.
        self._db = open_connection(path, readonly=False, same_thread=True)
        # Configuration -> conf_id, populated by `find_or_add_conf`.
        # Positive lookups only (None never cached).
        self._conf_id_cache: dict[Configuration, int] = {}

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
        conf_id = self._conf_id_cache.get(conf)
        if conf_id is not None:
            return conf_id
        conf_dict = conf.to_dict()
        cur.execute(FIND_CONF, conf_dict)
        row = cur.fetchone()
        if row is not None:
            conf_id = row[0]
            self._conf_id_cache[conf] = conf_id
            return conf_id
        conf_dict['weights'] = conf.num_weights
        cur.execute(INSERT_CONF, conf_dict)
        conf_id = cur.lastrowid
        self._conf_id_cache[conf] = conf_id
        return conf_id

    def find_or_add_conf(self, conf: Configuration) -> int:
        """Find the conf in the db (or insert), returning the conf_id."""
        with latency.timer('DbWriter.find_or_add_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN IMMEDIATE')
            conf_id = self._find_or_add_conf(cur, conf)
            cur.execute('COMMIT')
            return conf_id

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
            'step_loss': _pack_ndarray(run.step_loss),
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

    def save_model(self, name: str, obj) -> None:
        """Pickle `obj` and upsert into the model table under `name`."""
        data = pickle.dumps(obj)
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')
        cur.execute(UPSERT_MODEL, {
            'name': name,
            'data': data,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        })
        cur.execute('COMMIT')
        logging.info(f"Saved model '{name}' ({len(data)} bytes)")


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
class SaveModel:
    name: str
    obj: object


@dataclass
class UpdateAllScores:
    pass


@dataclass
class Stop:
    pass


WriteMessage = (
    AddRun
    | UpsertPredictedTimeEstimates
    | SaveModel
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

    def upsert_predicted_time_estimates(
        self, rows: Iterable[tuple[int, str, float]]
    ) -> None:
        # Materialize before queueing — the producer's iterator may
        # close before the WriterThread gets to it.
        self._queue.put(UpsertPredictedTimeEstimates([
            PredictedTimeRow(conf_id=cid, system=s, time_s=t)
            for (cid, s, t) in rows
        ]))

    def save_model(self, name: str, obj) -> None:
        self._queue.put(SaveModel(name=name, obj=obj))

    def update_all_scores(self) -> None:
        self._queue.put(UpdateAllScores())


class WriterThread(threading.Thread):
    """Owns the only post-init writable connection in the server.

    Opens its `DbWriter` on this thread (so `check_same_thread=True`
    holds) and serializes all writes from background producers.
    """

    def __init__(self, db_path: Optional[str], queue: Queue):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue

    def run(self):
        writer = DbWriter(self._db_path)
        logging.info("Started writer thread")
        try:
            while True:
                m = self._queue.get()
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
                            (r.conf_id, r.system, r.time_s) for r in rows)
                    case SaveModel(name=name, obj=obj):
                        writer.save_model(name, obj)
                    case UpdateAllScores():
                        writer.update_all_scores()
                    case Stop():
                        logging.info("Stopping writer thread")
                        break
                    case _:
                        assert False, f"Unknown write message: {m!r}"
        finally:
            writer.close()
