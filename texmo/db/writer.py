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
import time
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
from .conf_index import (
    CONF_CHANGE_KEEP,
    SHADOW_EVERY,
    ConfIndex,
    ConfIndexUnavailable,
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
# The `conf.id` tie-break is the index walk's own order (entries are
# (median_score, rowid)), written out so ties are deterministic and
# `ConfIndex` reproduces them exactly; the plan is unchanged.
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
ORDER BY conf.median_score ASC, conf.id ASC
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

@dataclass
class PickMeStatus:
    """Outcome of `DbWriter.add_pick_me_conf`.

    `target` is the conf's stored `pick_me` value after the call (the
    run count the search keeps picking it towards;
    `DbReader.pick_me_candidates` floors it at its `min_runs`),
    `num_runs` how many runs it has now.
    """

    conf_id: int
    inserted: bool
    num_runs: int
    target: int


class DbWriter(object):
    """Read-write handle to the results DB. Owns write transactions.

    With `conf_index=True`, `_top_conf_at` (the `changed_winner`
    sample `add_run` takes twice per tracked run) is answered from a
    lean in-memory `ConfIndex` owned by this writer, synced from this
    connection inside the write transaction -- so it sees the
    transaction's own uncommitted changes, exactly as the SQL did.
    Every `shadow_every`-th call also runs the SQL and compares.
    """

    @staticmethod
    def from_args(db: Optional[str]) -> 'DbWriter':
        return DbWriter(db)

    def __init__(
        self,
        path: Optional[str] = None,
        conf_index: bool = False,
        shadow_every: int = SHADOW_EVERY,
    ):
        if path is None:
            path = ':memory:'
        self._path = path
        self._db = open_connection(path, readonly=False)
        self._conf_index: Optional[ConfIndex] = None
        self._shadow_every = shadow_every
        self._top_conf_calls = 0
        self._shadow_checks = 0
        self._shadow_mismatches = 0
        if conf_index:
            self.enable_conf_index()

    def enable_conf_index(self) -> bool:
        """Attach and build the writer's `ConfIndex` in a read
        transaction of its own (a full load takes seconds; doing it
        here keeps it out of the first `add_run`'s write lock)."""
        index = ConfIndex(full=False)
        self._db.execute('BEGIN')
        try:
            index.sync(self._db)
        except ConfIndexUnavailable as e:
            logging.warning(f'Writer ConfIndex unavailable: {e}')
            return False
        finally:
            self._db.commit()
        self._conf_index = index
        return True

    @property
    def conf_index(self) -> Optional[ConfIndex]:
        return self._conf_index

    @property
    def shadow_stats(self) -> tuple[int, int]:
        """(shadow checks run, mismatches seen)."""
        return self._shadow_checks, self._shadow_mismatches

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
        self, conf: Configuration, runs: int = 1,
    ) -> PickMeStatus:
        """Mark `conf` as a priority pick until it has `runs` runs.

        `pick_me` holds the conf's own target run count (see
        `DbReader.pick_me_candidates`). A conf not yet in the DB is inserted
        with `pick_me = runs`; an existing one has its target *raised*
        to `runs` — never lowered, so re-asking for fewer runs than an
        earlier request can't cancel it, and a conf already flagged
        keeps whatever target it had.

        Returns the resulting `PickMeStatus`.
        """
        assert runs >= 1, f'pick_me target must be >= 1, got {runs}'
        with latency.timer('DbWriter.add_pick_me_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN IMMEDIATE')
            conf_dict = conf.to_dict()
            cur.execute(FIND_CONF, conf_dict)
            row = cur.fetchone()
            if row is not None:
                conf_id = row[0]
                cur.execute(
                    'UPDATE conf SET pick_me = MAX(pick_me, :runs) '
                    'WHERE id = :id',
                    {'runs': runs, 'id': conf_id})
                inserted = False
            else:
                conf_dict['weights'] = conf.model.num_weights
                conf_dict['num_layers'] = conf.model.num_layers
                conf_dict['pick_me'] = runs
                cur.execute(
                    'INSERT INTO conf '
                    '(spec, weights, lr, length, batch, steps, precision,'
                    ' decay, cosine, num_layers, pick_me) '
                    'VALUES '
                    '(:spec, :weights, :lr, :length, :batch, :steps,'
                    ' :precision, :decay, :cosine, :num_layers, :pick_me)',
                    conf_dict,
                )
                conf_id = cur.lastrowid
                inserted = True
            target = cur.execute(
                'SELECT pick_me FROM conf WHERE id = ?',
                (conf_id,)).fetchone()[0]
            num_runs = cur.execute(
                'SELECT COUNT(*) FROM run WHERE conf_id = ?',
                (conf_id,)).fetchone()[0]
            cur.execute('COMMIT')
            return PickMeStatus(
                conf_id=conf_id, inserted=inserted, num_runs=num_runs,
                target=target)

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

    def _top_conf_at_sql(
        self, cur: sqlite3.Cursor,
        system: str, max_weights: int, max_time: float,
    ) -> Optional[int]:
        """SQL implementation of `_top_conf_at`: the fallback without
        an index, and the shadow/test reference."""
        cur.execute(TOP_CONF_AT, {
            'system': system,
            'max_weights': max_weights,
            'max_time': max_time,
        })
        row = cur.fetchone()
        return row[0] if row is not None else None

    def _top_conf_at(
        self, cur: sqlite3.Cursor,
        system: str, max_weights: int, max_time: float,
    ) -> Optional[int]:
        """Best-scoring conf (ties by id) with weights <= max_weights
        and an estimate of any source on `system` with time_s <=
        max_time; None if there is none. Must run inside the write
        transaction (as `add_run` calls it)."""
        index = self._conf_index
        if index is None:
            return self._top_conf_at_sql(cur, system, max_weights, max_time)
        with latency.timer('DbWriter._top_conf_at.index'):
            try:
                index.sync(self._db)
            except ConfIndexUnavailable as e:
                logging.error(
                    f'Writer ConfIndex disabled, back to SQL: {e}')
                self._conf_index = None
                return self._top_conf_at_sql(
                    cur, system, max_weights, max_time)
            if max_weights is None or max_time is None:
                got = None  # `<= NULL` matches nothing in SQL either
            else:
                ids = index.ordered_ids(
                    None, system, weights_max=max_weights,
                    time_max=max_time)
                got = int(ids[0]) if len(ids) else None
            problem = None if got is None else self._top_conf_drift(
                cur, index, got, system, max_weights, max_time)
        if problem is not None:
            logging.error(
                f'Writer ConfIndex drift in _top_conf_at(system='
                f'{system!r}, max_weights={max_weights!r}, max_time='
                f'{max_time!r}): conf {got}: {problem}. Rebuilding; '
                f'answering from SQL.')
            index.invalidate()
            return self._top_conf_at_sql(cur, system, max_weights, max_time)

        due = (self._shadow_every > 0
               and self._top_conf_calls % self._shadow_every == 0)
        self._top_conf_calls += 1
        if not due:
            return got
        t0 = time.perf_counter()
        with latency.timer('DbWriter._top_conf_at.shadow_sql'):
            want = self._top_conf_at_sql(cur, system, max_weights, max_time)
        sql_ms = (time.perf_counter() - t0) * 1e3
        self._shadow_checks += 1
        args = (f'_top_conf_at(system={system!r}, max_weights='
                f'{max_weights!r}, max_time={max_time!r})')
        if got == want:
            logging.info(
                f'Writer ConfIndex shadow ok #{self._shadow_checks}: '
                f'{args}: conf {got}; sql {sql_ms:.1f} ms')
            return got
        self._shadow_mismatches += 1
        logging.error(
            f'Writer ConfIndex shadow MISMATCH #{self._shadow_checks} '
            f'({self._shadow_mismatches} so far): {args}: index conf '
            f'{got}, sql conf {want}. Rebuilding; answering from SQL.')
        index.invalidate()
        return want

    def _top_conf_drift(
        self, cur: sqlite3.Cursor, index: ConfIndex, conf_id: int,
        system: str, max_weights: int, max_time: float,
    ) -> Optional[str]:
        """Cheap always-on tripwire: does the conf the index picked
        satisfy the query in SQL, with the score the index ranked it
        by? Returns a description of the disagreement, or None."""
        row = cur.execute(
            'SELECT median_score, weights, '
            '(SELECT time_s FROM conf_time_estimate '
            ' WHERE conf_id = conf.id AND system = ?) '
            'FROM conf WHERE id = ?', (system, conf_id)).fetchone()
        if row is None:
            return 'conf row missing'
        score, weights, time_s = row[0], row[1], row[2]
        if score != index.score_of(conf_id):
            return f'median_score {score!r} != index {index.score_of(conf_id)!r}'
        if not weights <= max_weights:
            return f'weights {weights} > {max_weights}'
        if not (time_s is not None and time_s <= max_time):
            return f'time_s {time_s!r} > {max_time}'
        return None

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
            try:
                return self._add_run(
                    conf, run, conf_id, timestamp,
                    strategy, track_winner_change,
                )
            except BaseException:
                # The index may have synced this transaction's own
                # uncommitted rows (and seqs AUTOINCREMENT will hand
                # out again after the rollback), so it no longer
                # describes any snapshot: rebuild on next use.
                if self._conf_index is not None:
                    self._conf_index.invalidate()
                if self._db.in_transaction:
                    self._db.rollback()
                raise

    def prune_conf_change(self, keep: int = CONF_CHANGE_KEEP) -> int:
        """Drop all but the newest `keep` changelog rows, raising
        `conf_change_floor.floor` in the same transaction so any
        consumer still behind it detects the gap and rebuilds rather
        than missing changes. Returns the number of rows deleted."""
        cur = self._db.cursor()
        cur.execute('BEGIN IMMEDIATE')
        row = cur.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'conf_change'"
        ).fetchone()
        hw = row[0] if row is not None else 0
        (floor,) = cur.execute(
            'SELECT floor FROM conf_change_floor WHERE id = 0').fetchone()
        new_floor = hw - keep
        if new_floor <= floor:
            cur.execute('COMMIT')
            return 0
        cur.execute(
            'DELETE FROM conf_change WHERE seq <= ?', (new_floor,))
        deleted = cur.rowcount
        cur.execute(
            'UPDATE conf_change_floor SET floor = ? WHERE id = 0',
            (new_floor,))
        cur.execute('COMMIT')
        return deleted

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


class FrontierVersion:
    """Monotonic count of Pareto-winner flips, shared writer -> search.

    `_add_run` already samples the winning conf at (system,
    train_time, num_weights) before and after every insert -- the
    `changed_winner` column. That flip is the cheapest signal the
    server has that the frontier moved, and the frontier-seeding
    strategy (`Search._select_frontier_seed`) needs exactly that to
    know when its cached candidate queue went stale.

    Deliberately NOT a queue message, even though the house rule
    prefers those: the value coalesces, so a burst of flips costs one
    cache rebuild rather than one message each, and a reader that
    wasn't looking misses nothing. Lives here rather than in
    `search.py` because `search` imports `db`, not the other way
    around.

    Writer: the writer thread only (see `WriterThread.run`), which is
    the single thread that can observe a flip. Readers: the search
    thread, which compares `value` against the one its cache was built
    at. One writer plus a plain int attribute assignment (atomic in
    CPython) is the whole synchronization story -- the same shape as
    `LossModelHolder`'s pointer swap.
    """

    def __init__(self):
        self._n = 0

    @property
    def value(self) -> int:
        return self._n

    def bump(self) -> None:
        self._n += 1


@dataclass
class AddRun:
    conf: Configuration
    run: Run
    strategy: Optional[str] = None
    track_winner_change: bool = False


@dataclass
class AddPickMe:
    """Flag `conf` as a priority pick until it has `runs` runs.

    RPC-style: when `reply` is given, the writer thread puts the
    resulting `PickMeStatus` on it (the `/pick_me` handler blocks on
    that, since the caller wants the conf id and run count back). A
    write that raises takes the server down through the panic path
    and nothing is ever put, so every waiter uses a timeout.
    """

    conf: Configuration
    runs: int = 1
    reply: Optional[Queue] = None


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
    | AddPickMe
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

    def add_pick_me_conf(
        self,
        conf: Configuration,
        runs: int = 1,
        reply: Optional[Queue] = None,
    ) -> None:
        self._queue.put(AddPickMe(conf=conf, runs=runs, reply=reply))

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


# WriterThread prunes `conf_change` at startup and then once per this
# many queue messages.
_PRUNE_EVERY = 1000


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

    With a `frontier_version`, this thread is also the publisher of
    "the Pareto frontier moved": `add_run` computes that flip inside
    its own transaction and no other thread can see it. See
    `FrontierVersion`.
    """

    def __init__(
        self,
        db_path: Optional[str],
        queue: Queue,
        on_fatal: Optional[Callable[[], None]] = None,
        frontier_version: Optional[FrontierVersion] = None,
        conf_index: bool = False,
        shadow_every: int = SHADOW_EVERY,
    ):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        self._on_fatal = on_fatal
        self._frontier_version = frontier_version
        self._conf_index = conf_index
        self._shadow_every = shadow_every

    def _prune(self, writer: DbWriter) -> None:
        with latency.timer('WriterThread.prune_conf_change'):
            deleted = writer.prune_conf_change()
        if deleted:
            logging.info(f'Pruned {deleted} conf_change rows')

    def run(self):
        # The writer's own ConfIndex (lean: weights/score/estimates)
        # lives and dies with this connection, on this thread; the
        # SearchThread has a separate one. See docs/threads.md.
        writer = DbWriter(
            self._db_path, conf_index=self._conf_index,
            shadow_every=self._shadow_every)
        logging.info("Started writer thread")
        processed = 0
        try:
            while True:
                if processed % _PRUNE_EVERY == 0:
                    # Startup, then every _PRUNE_EVERY messages: keeps
                    # conf_change bounded (a full estimate refresh
                    # logs ~1M rows) between server restarts.
                    try:
                        self._prune(writer)
                    except Exception as e:
                        logging.critical(
                            f"Writer thread died pruning conf_change: "
                            f"{e!r}; shutting down.", exc_info=True)
                        self._panic()
                        break
                m = self._queue.get()
                processed += 1
                try:
                    match m:
                        case AddRun(
                            conf=conf, run=run, strategy=strategy,
                            track_winner_change=track,
                        ):
                            changed = writer.add_run(
                                conf, run,
                                strategy=strategy,
                                track_winner_change=track,
                            )
                            # The only place the flip is observable:
                            # the proxy is fire-and-forget, so the
                            # bool dies here unless we publish it.
                            if changed and self._frontier_version is not None:
                                self._frontier_version.bump()
                        case AddPickMe(
                            conf=conf, runs=runs, reply=reply,
                        ):
                            status = writer.add_pick_me_conf(
                                conf, runs=runs)
                            if reply is not None:
                                reply.put(status)
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
