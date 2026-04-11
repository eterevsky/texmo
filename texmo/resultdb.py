import logging
import math
import os
import re
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from statistics import StatisticsError, median
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from . import latency
from .common import INF, console
from .configuration import Configuration, Precision, Template
from .model import build_model_def
from .precision import Precision
from .run import Run


def _pack_ndarray(step_loss):
    step_loss = np.array(step_loss, dtype=np.float32)
    assert len(step_loss.shape) == 1
    return step_loss.tobytes()


def _unpack_ndarray(blob):
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _build_loss_trend(step_loss, model_version, params):
    from .predict import build_loss_trend

    return build_loss_trend(step_loss, model_version, params)


def _make_template_conditions(template: Template) -> tuple[list[str], dict]:
    conditions = []
    params = {}
    if template.spec is not None:
        conditions.append('spec = :spec')
        params['spec'] = template.spec
    if template.regex is not None:
        conditions.append('spec REGEXP :regex')
        params['regex'] = template.regex.pattern
    if template.lr.min > 0:
        conditions.append('lr >= :lr_min')
        params['lr_min'] = template.lr.min
    if template.lr.max < INF:
        conditions.append('lr <= :lr_max')
        params['lr_max'] = template.lr.min
    if template.length.min > 1:
        conditions.append('length >= :length_min')
        params['length_min'] = template.length.min
    if template.length.max < INF:
        conditions.append('length <= :length_max')
        params['length_max'] = template.length.max
    if template.batch.min > 1:
        conditions.append('batch >= :batch_min')
        params['batch_min'] = template.batch.min
    if template.batch.max < INF:
        conditions.append('batch <= :batch_max')
        params['batch_max'] = template.batch.max
    if template.steps.min >= 2:
        conditions.append('steps >= :steps_min')
        params['steps_min'] = template.steps.min
    if template.steps.max < INF:
        conditions.append('steps <= :steps_max')
        params['steps_max'] = template.steps.max
    if template.max_weights.max < INF:
        conditions.append('weights <= :weights_max')
        params['weights_max'] = template.max_weights.max
    if len(template.precision) != len(Precision):
        precisions_list = ', '.join(f'"{p}"' for p in template.precision)
        conditions.append(f'precision IN ({precisions_list})')
    if template.decay.min > 0:
        conditions.append('decay >= :decay_min')
        params['decay_min'] = template.decay.min
    if template.decay.max < 1:
        conditions.append('decay <= :decay_max')
        params['decay_max'] = template.decay.max
    return conditions, params


class ConfScore(object):
    def __init__(
        self,
        conf_id: int,
        conf: Configuration,
        median_score: Optional[float],
        system: str,
        median_time: Optional[float],
        num_runs: int,
    ):
        self.conf_id: int = conf_id
        self.conf: Configuration = conf
        self.median_score: Optional[float] = median_score
        self.system: str = system
        self.median_time: Optional[float] = median_time
        self.num_runs: int = num_runs  # Number of runs on all systems

    @staticmethod
    def _from_row(row: sqlite3.Row, system: Optional[str] = None) -> ConfScore:
        precision = Precision(row['precision'])
        model = build_model_def(row['spec'], precision=precision)

        if system is None:
            system = row['system']

        conf = Configuration(
            model, lr=row['lr'], length=row['length'],
            batch=row['batch'], steps=row['steps'],
            decay=row['decay']
        )
        return ConfScore(
            row['conf_id'],
            conf,
            median_score=row['median_score'],
            system=system,
            median_time=row['median_time'],
            num_runs=row['num_runs'],
        )


def _regexp(pattern, input_string):
    if input_string is None or pattern is None:
        return False
    return re.fullmatch(pattern, input_string) is not None


_FIND_CONF = """
SELECT id
FROM conf
WHERE spec = :spec
    AND lr = :lr
    AND length = :length
    AND batch = :batch
    AND steps = :steps
    AND precision = :precision
    AND decay = :decay
"""

_INSERT_CONF = """
INSERT INTO conf (spec, weights, lr, length, batch, steps, precision, decay)
VALUES (:spec, :weights, :lr, :length, :batch, :steps, :precision, :decay)
"""

_INSERT_RUN = """
INSERT INTO run(conf_id, system, train_time, timestamp, loss, step_loss, loss_model_v, loss_model)
VALUES (:conf_id, :system, :train_time, :timestamp, :loss, :step_loss, :loss_model_v, :loss_model)
"""

_GET_TRAIN_TIMES = """
SELECT train_time
FROM run
WHERE conf_id = :conf_id
  AND system = :system
"""

_INSERT_MEDIAN_TIME = """
INSERT OR REPLACE INTO conf_time(conf_id, system, median_time)
VALUES (:conf_id, :system, :median_time)
"""

_GET_CONFS_RUNS = """
SELECT conf.id AS conf_id,
        spec,
        precision,
        lr,
        length,
        batch,
        steps,
        decay,
        run.id AS run_id,
        system,
        train_time,
        timestamp,
        loss,
        step_loss,
        loss_model_v,
        loss_model
FROM conf, run
WHERE conf.id = run.conf_id
"""



class ResultDB(object):
    @staticmethod
    def from_args(db: Optional[str]) -> 'ResultDB':
        return ResultDB(db)

    def __init__(self, path: Optional[str] = None, readonly: bool = False):
        if path is None:
            path = ':memory:'
        self._path = path
        self._readonly = readonly
        exists = path != ':memory:' and os.path.exists(path)
        if path != ':memory:':
            logging.info(f'Connecting to results DB {path}')

        if readonly:
            assert path != ':memory:', \
                "Can't open :memory: database read-only"
            # Read-only instances are single-threaded by construction —
            # created, used, and closed within a single scope (typically
            # a Flask request handler).
            self._db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        else:
            # The main (writer) instance is created in the main thread
            # but used by SearchThread; check_same_thread=False disables
            # SQLite's safety check. By convention only SearchThread
            # touches this connection after construction.
            self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.create_function('REGEXP', 2, _regexp)

        if not exists:
            assert not readonly
            schema_path = os.path.join(
                os.path.dirname(__file__), 'db.sql'
            )
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()

        # Cache: Configuration -> conf_id (populated lazily)
        self._conf_id_cache: dict[Configuration, int | None] = {}

    def open_readonly(self) -> 'ResultDB':
        """Open a new read-only ResultDB on the same database.

        Use as a context manager from Flask request handlers to avoid
        contending with the writer thread holding the main connection.
        """
        return ResultDB(self._path, readonly=True)

    def close(self):
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def commit(self):
        self._db.commit()

    def get_conf_id(self, conf: Configuration) -> int | None:
        """Return conf_id if this configuration exists in the DB, else None.

        Uses an in-memory cache for positive lookups only.
        """
        conf_id = self._conf_id_cache.get(conf)
        if conf_id is not None:
            return conf_id

        conf_dict = conf.to_dict()
        cur = self._db.execute(_FIND_CONF, conf_dict)
        row = cur.fetchone()
        if row is None:
            return None
        conf_id = row[0]
        self._conf_id_cache[conf] = conf_id
        return conf_id

    def _find_or_add_conf(
        self, cur: sqlite3.Cursor, conf: Configuration
    ) -> int:
        conf_id = self.get_conf_id(conf)
        if conf_id is not None:
            return conf_id
        conf_dict = conf.to_dict()
        conf_dict['weights'] = conf.num_weights
        cur.execute(_INSERT_CONF, conf_dict)
        conf_id = cur.lastrowid
        self._conf_id_cache[conf] = conf_id
        return conf_id

    def find_or_add_conf(self, conf: Configuration) -> int:
        """Finds the conf in the db and returns the configuration id."""
        with latency.timer('ResultDB.find_or_add_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN TRANSACTION')
            conf_id = self._find_or_add_conf(cur, conf)
            cur.execute('COMMIT')
            return conf_id

    def get_run_counts(
        self, conf_ids: list[int], system: Optional[str] = None,
    ) -> dict[int, tuple[int, int]]:
        """Return {conf_id: (total_runs, system_runs)} for the given conf_ids.

        `total_runs` is the number of runs across all systems.
        `system_runs` is the number of runs on the given system (0 if
        `system` is None).
        """
        if not conf_ids:
            return {}
        placeholders = ','.join('?' for _ in conf_ids)
        if system is None:
            cur = self._db.execute(
                f'SELECT conf_id, COUNT(*), 0 FROM run '
                f'WHERE conf_id IN ({placeholders}) GROUP BY conf_id',
                conf_ids,
            )
        else:
            cur = self._db.execute(
                f'SELECT conf_id, COUNT(*), SUM(system = ?) FROM run '
                f'WHERE conf_id IN ({placeholders}) GROUP BY conf_id',
                [system] + conf_ids,
            )
        return {row[0]: (row[1], row[2]) for row in cur}

    def _add_run_execute(self, cur: sqlite3.Cursor, run_dict: dict):
        with latency.timer('ResultDB._add_run_execute'):
            cur.execute(_INSERT_RUN, run_dict)

    def _update_median_score(self, cur: sqlite3.Cursor, conf_id: int):
        with latency.timer('ResultDB._update_median_score'):
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
        with latency.timer('ResultDB._update_median_time'):
            cur.execute(_GET_TRAIN_TIMES, {'conf_id': conf_id, 'system': system})
            try:
                median_time = median(row[0] for row in cur if row[0] > 0.0001)
            except StatisticsError:
                median_time = None
            if median_time is not None and median_time > 0.0001:
                cur.execute(_INSERT_MEDIAN_TIME,
                            {
                                'median_time': median_time,
                                'conf_id': conf_id,
                                'system': system,
                            })

    def _update_scores(self, cur: sqlite3.Cursor, conf_id: int, system: str):
        self._update_median_score(cur, conf_id)
        self._update_median_time(cur, conf_id, system)

    def update_all_scores(self):
        with latency.timer('ResultDB.update_scores'):
            cur = self._db.cursor()

            cur.execute(
                'SELECT DISTINCT conf.id AS conf_id, system FROM conf, run WHERE conf.id = run.conf_id'
            )

            for row in cur.fetchall():
                conf_id = row['conf_id']
                system = row['system']
                logging.info(f'{conf_id} {system}')
                self._update_scores(cur, conf_id, system)
            self.commit()

    def clear_system(self, system: str) -> int:
        """Delete all runs and conf_time entries for a given system.

        Recomputes median_score for any affected configurations.
        Returns the number of runs that were deleted.
        """
        cur = self._db.cursor()
        cur.execute('BEGIN TRANSACTION')

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
            'DELETE FROM conf_time WHERE system = :system',
            {'system': system},
        )

        # Recompute median_score for affected confs (median_time for this
        # system is already gone).
        for conf_id in affected_conf_ids:
            self._update_median_score(cur, conf_id)

        cur.execute('COMMIT')
        return deleted

    def _add_run(
        self,
        conf: Configuration,
        run: Run,
        conf_id: Optional[int],
        timestamp: Optional[datetime],
    ):
        cur = self._db.cursor()
        cur.execute('BEGIN TRANSACTION')

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

        run_dict = {
            'conf_id': conf_id,
            'system': run.system,
            'train_time': run.train_time or 0,
            'timestamp': timestamp,
            'loss': loss,
            'step_loss': _pack_ndarray(run.step_loss),
            'loss_model_v': loss_model_v,
            'loss_model': loss_model,
        }

        self._add_run_execute(cur, run_dict)
        self._update_scores(cur, conf_id, run.system)
        cur.execute('COMMIT')

    def add_run(
        self,
        conf: Configuration,
        run: Run,
        conf_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ):
        assert run.train_time is None or run.train_time > 0.0001
        with latency.timer('ResultDB.add_run'):
            self._add_run(conf, run, conf_id, timestamp)

    def get_systems(self) -> list[str]:
        """Return a sorted list of all systems that have runs in the DB."""
        cur = self._db.execute(
            'SELECT DISTINCT system FROM run ORDER BY system')
        return [row[0] for row in cur]

    def top_confs_global(
        self, template: Template, system: Optional[str] = None
    ):
        """Yield top confs ordered by weights, one per weight count.

        Args:
            template: filter for the configurations.
            system: if provided, only include configurations that have been
                run on this system, and report median_time for this system
                only. Score is still the global median across all systems.
        """
        assert type(template) is Template

        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')
        conditions.append('num_runs > 1')

        if system is not None:
            conditions.append(
                'EXISTS (SELECT 1 FROM run '
                'WHERE run.conf_id = conf.id AND run.system = :system)')
            params['system'] = system

        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'length', 'batch', 'steps'])

        where = 'WHERE ' + ' AND '.join(conditions)

        if system is None:
            # Best median_time across all systems, report the winning system.
            time_select = (
                "(SELECT system FROM conf_time WHERE conf_id=conf.id "
                " ORDER BY median_time LIMIT 1) AS system, "
                "(SELECT MIN(median_time) FROM conf_time "
                " WHERE conf_id=conf.id) AS median_time"
            )
        else:
            # Median_time on the selected system specifically.
            time_select = (
                ":system AS system, "
                "(SELECT median_time FROM conf_time "
                " WHERE conf_id=conf.id AND system=:system) AS median_time"
            )

        query = f"""
            WITH conf_with_runs AS (
                SELECT id,
                       {conf_fields},
                       weights,
                       median_score,
                       (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                       {time_select}
                FROM conf
                {where}
            ),
            ranked_confs AS (SELECT id AS conf_id,
                   {conf_fields},
                   weights,
                   median_score,
                   num_runs,
                   system,
                   median_time,
                   ROW_NUMBER() OVER (PARTITION BY weights ORDER BY median_score) AS rn
            FROM conf_with_runs
            )
            SELECT conf_id,
                   {conf_fields},
                   weights,
                   median_score,
                   num_runs,
                   system,
                   median_time
            FROM ranked_confs
            WHERE rn = 1
        """

        cur = self._db.execute(query, params)

        best_score = INF

        for row in cur:
            conf_score = ConfScore._from_row(row)
            if conf_score.median_score < best_score:
                best_score = conf_score.median_score
                yield conf_score


    def fastest_near_best_segments(
        self,
        template: Template,
        system: str,
        pareto: list[ConfScore],
        tolerance: float = 0.01,
    ) -> list[tuple[int, Optional[int], ConfScore]]:
        """Build a piecewise-constant "fastest near-best" curve.

        Args:
            template: filter for the candidate configs.
            system: only configs with runs on this system are considered.
            pareto: the Pareto frontier — a list of ConfScores sorted by
                `weights` ascending with strictly decreasing median_score.
                Typically the result of `top_confs_global(..., system=...)`.
            tolerance: a conf is "near-best" at budget W if its score is
                within `(1 + tolerance)` of the best Pareto loss achievable
                at weight budget W.

        For each interval between consecutive Pareto points `[W_i, W_{i+1})`,
        query all confs with `median_score <= L_i * (1 + tolerance) AND
        weights < W_{i+1}`, ordered by median_time on `system` ASC. Walk
        the results, assigning each qualifying config to a sub-range of
        the interval such that for any weight budget `w` in the sub-range,
        the returned conf is the one with the smallest median_time.

        The last interval has no `W_{i+1}` boundary and extends to
        infinity — the last segment's `w_high` is None.

        Returns a list of `(w_low, w_high, ConfScore)` segments sorted by
        `w_low` ascending. The union covers `[W_0, +infty)`.
        """
        if not pareto:
            return []

        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'length', 'batch', 'steps'])

        segments: list[tuple[int, Optional[int], ConfScore]] = []

        n = len(pareto)
        ws = [cs.conf.model.num_weights for cs in pareto]
        ls = [cs.median_score for cs in pareto]

        for i in range(n):
            w_left = ws[i]
            w_right: Optional[int] = ws[i + 1] if i + 1 < n else None
            threshold = ls[i] * (1.0 + tolerance)

            # Build the candidate query: all confs with score within
            # tolerance of L_i, optionally bounded by w_right. Ordered
            # by time ASC so we can walk greedily.
            conditions, params = _make_template_conditions(template)
            conditions.append('median_score IS NOT NULL')
            conditions.append('median_score <= :threshold')
            params['threshold'] = threshold
            params['system'] = system
            if w_right is not None:
                conditions.append('weights < :w_right')
                params['w_right'] = w_right
            where = 'WHERE ' + ' AND '.join(conditions)

            query = f"""
                SELECT conf.id AS conf_id, {conf_fields},
                       weights,
                       median_score,
                       (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                       :system AS system,
                       ct.median_time AS median_time
                FROM conf
                JOIN conf_time AS ct ON ct.conf_id = conf.id AND ct.system = :system
                {where}
                ORDER BY ct.median_time ASC
            """
            cur = self._db.execute(query, params)

            # Walk results, collecting segments from the right boundary
            # downward. `uncovered_right` is the exclusive right end of
            # the still-uncovered portion of [w_left, w_right).
            #
            # The Pareto point `(W_i, L_i) = pareto[i]` itself is
            # guaranteed to satisfy this query (score exactly L_i, weight
            # W_i, has runs on the system), so we're always guaranteed
            # to hit a `w <= w_left` row and terminate with the interval
            # fully covered.
            uncovered_right: Optional[int] = w_right
            done = False
            for row in cur:
                w = row['weights']
                if uncovered_right is not None and w >= uncovered_right:
                    # A faster conf already covers this weight range.
                    continue
                # A conf with w <= w_left covers everything still
                # uncovered in this interval. Clamp its segment to
                # [w_left, uncovered_right) and stop.
                seg_low = max(w, w_left)
                segments.append(
                    (seg_low, uncovered_right, ConfScore._from_row(row)))
                uncovered_right = seg_low
                if w <= w_left:
                    done = True
                    break
            assert done, (
                f"interval [{w_left}, {w_right}) not fully covered — "
                "the Pareto point itself should have been in the results")

        segments.sort(key=lambda seg: seg[0])
        return segments

    def top_confs_for_system(
        self,
        system: str,
        template: Template,
        max_weights: Optional[float] = None,
        max_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterable[ConfScore]:
        assert type(system) is str
        assert type(template) is Template

        conditions, params = _make_template_conditions(template)

        conditions.append('median_score IS NOT NULL')
        params['system'] = system

        if max_weights:
            conditions.append('weights <= :max_weight')
            params['max_weight'] =  max_weights

        if max_time:
            conditions.append('median_time <= :max_time')
            params['max_time'] = max_time

        where = 'WHERE ' + ' AND '.join(conditions)

        if limit is None:
            limit = ''
        else:
            params['limit'] = limit
            limit = 'LIMIT :limit'

        query = f"""
            SELECT conf.id AS conf_id, spec, precision, lr, length, batch,
                decay, steps, median_score,
                (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                (SELECT median_time FROM conf_time
                 WHERE conf_id = conf.id AND system = :system
                 ORDER BY median_time LIMIT 1) AS median_time
            FROM conf {where} ORDER BY median_score ASC {limit}
            """

        cur = self._db.execute(query, params)

        for row in cur:
            yield ConfScore._from_row(row, system)

    def total_runs(self) -> int:
        cur = self._db.execute('SELECT COUNT(*) AS count FROM run')
        total = cur.fetchone()['count']
        assert isinstance(total, int)
        return total

    def get_confs_runs(
        self, with_timestamps: bool = False
    ) -> Iterable[tuple[int, Configuration, Run]]:
        cur = self._db.cursor()
        cur.execute(_GET_CONFS_RUNS)

        for row in cur:
            with latency.timer('ResultDB.get_confs_runs-row'):
                conf_id = row['conf_id']

                precision = Precision(row['precision'])
                model = build_model_def(row['spec'], precision=precision)
                conf = Configuration(model=model,
                                      lr=row['lr'], length=row['length'],
                                      batch=row['batch'], steps=row['steps'],
                                      decay=row['decay'])

                step_loss = _unpack_ndarray(row['step_loss'])
                loss_trend = _build_loss_trend(
                    step_loss,
                    row['loss_model_v'],
                    _unpack_ndarray(row['loss_model']),
                )

                run = Run(
                    id=row['run_id'],
                    step_loss=step_loss,
                    loss=row['loss'],
                    loss_trend=loss_trend,
                    train_time=row['train_time'],
                    system=row['system'],
                )

                if with_timestamps:
                    if row['timestamp'] is None:
                        timestamp = None
                    else:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                    yield conf_id, conf, run, timestamp
                else:
                    yield conf_id, conf, run

    def check_run_exists(
        self, conf: Configuration, run: Run, timestamp: datetime
    ) -> bool:
        """Check if a run with the same configuration and timestamp exists."""
        conf_id = self.find_or_add_conf(conf)
        timestamp = timestamp.isoformat()
        cur = self._db.cursor()
        cur.execute(
            """
            SELECT 1
            FROM run
            WHERE conf_id = :conf_id
            AND timestamp = :timestamp
            AND system = :system
            """,
            {'conf_id': conf_id, 'timestamp': timestamp, 'system': run.system},
        )
        return cur.fetchone() is not None
