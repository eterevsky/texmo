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
from .configuration2 import Configuration2, Optimizer, Precision, Template
from .model3 import build_model
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
    if len(template.optimizer) != len(Optimizer):
        optimizers_list = ', '.join(f'"{o}"' for o in template.optimizer)
        conditions.append(f'optimizer IN ({optimizers_list})')
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
        conf: Configuration2,
        median_score: Optional[float],
        system: str,
        median_time: Optional[float],
        num_runs: int,
    ):
        self.conf_id: int = conf_id
        self.conf: Configuration2 = conf
        self.median_score: Optional[float] = median_score
        self.system: str = system
        self.median_time: Optional[float] = median_time
        self.num_runs: int = num_runs  # Number of runs on all systems

    @staticmethod
    def _from_row(row: sqlite3.Row, system: Optional[str] = None) -> ConfScore:
        model = build_model(row['spec'])

        if system is None:
            system = row['system']

        conf = Configuration2(
            model, precision=row['precision'], lr=row['lr'], length=row['length'],
            batch=row['batch'], steps=row['steps'], optimizer=row['optimizer'],
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


_INSERT_NEIGHBORS = """
INSERT OR IGNORE INTO neighbor (conf1_id, conf2_id)
VALUES (:conf1_id, :conf2_id)
"""

_SET_HAS_NEIGHBORS = 'UPDATE conf SET has_neighbors = 1 WHERE id = :conf_id'

_FIND_CONF = """
SELECT id, has_neighbors
FROM conf
WHERE spec = :spec
    AND lr = :lr
    AND length = :length
    AND batch = :batch
    AND steps = :steps
    AND precision = :precision
    AND optimizer = :optimizer
    AND decay = :decay
"""

_INSERT_CONF = """
INSERT INTO conf (spec, weights, lr, length, batch, steps, has_neighbors, precision, optimizer, decay)
VALUES (:spec, :weights, :lr, :length, :batch, :steps, 0, :precision, :optimizer, :decay)
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

_GET_CONF_RUNS_DIFF_STEPS = """
SELECT conf.steps, conf_time.median_time
FROM conf, conf_time
WHERE conf.id = conf_time.conf_id
    AND conf.spec = :spec
    AND conf.precision = :precision
    AND conf.lr = :lr
    AND conf.length = :length
    AND conf.batch = :batch
    AND conf.optimizer = :optimizer
    AND conf.decay = :decay
    AND conf_time.system = :system
"""

_GET_CONF = """
SELECT spec, precision, lr, length, batch, steps, optimizer, decay
FROM conf
WHERE id = :conf_id
"""

_GET_NEIGHBORS_BY_RUNS = """
SELECT conf.id AS conf_id, spec, precision, lr, length, batch,
       steps, optimizer, decay, median_score,
       (SELECT median_time FROM conf_time
        WHERE conf_id = conf.id AND system = :system) AS median_time,
       (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs
FROM conf, neighbor
WHERE neighbor.conf1_id = :conf_id
    AND neighbor.conf2_id = conf.id
ORDER BY num_runs ASC
"""

_GET_CONFS_RUNS = """
SELECT conf.id AS conf_id,
        spec,
        precision,
        lr,
        length,
        batch,
        steps,
        optimizer,
        decay,
        run.id AS run_id,
        system,
        train_time,
        timestamp,
        loss,
        step_loss,
        loss_model_v,
        loss_model,
        optimizer,
        decay
FROM conf, run
WHERE conf.id = run.conf_id
"""



class ResultDB(object):
    @staticmethod
    def from_args(db: Optional[str]) -> 'ResultDB':
        return ResultDB(db)

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = ':memory:'
        exists = path != ':memory:' and os.path.exists(path)
        if path != ':memory:':
            console.log(f'Connecting to results DB {path}')
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.create_function('REGEXP', 2, _regexp)

        if not exists:
            schema_path = os.path.join(
                os.path.dirname(__file__), 'persistent-db.sql'
            )
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()

    def commit(self):
        self._db.commit()

    def _init_neighbors(
        self, cur: sqlite3.Cursor, conf: Configuration2, conf_id: int
    ):
        with latency.timer('ResultDB._init_neighbors'):
            for neighbor in conf.neighbors():
                neighbor_id = self._find_or_add_conf(
                    cur, neighbor, init_neighbors=False
                )
                cur.executemany(
                    _INSERT_NEIGHBORS,
                    [
                        {'conf1_id': conf_id, 'conf2_id': neighbor_id},
                        {'conf1_id': neighbor_id, 'conf2_id': conf_id},
                    ],
                )
            cur.execute(_SET_HAS_NEIGHBORS, {'conf_id': conf_id})

    def _find_or_add_conf(
        self, cur: sqlite3.Cursor, conf: Configuration2, init_neighbors: bool
    ) -> int:
        conf_dict = conf.to_dict()
        conf_dict['weights'] = conf.model.weights
        cur.execute(_FIND_CONF, conf_dict)
        rows = cur.fetchall()
        assert len(rows) <= 1
        if rows:
            conf_id, has_neighbors = rows[0]
            if init_neighbors and not has_neighbors:
                self._init_neighbors(cur, conf, conf_id)
            return conf_id
        else:
            cur = self._db.cursor()
            cur.execute(_INSERT_CONF, conf_dict)
            conf_id = cur.lastrowid
            if init_neighbors:
                self._init_neighbors(cur, conf, conf_id)
            return conf_id

    def find_or_add_conf(
        self, conf: Configuration2, init_neighbors: bool
    ) -> int:
        """Finds the conf in the db and returns the configuration id."""
        with latency.timer('ResultDB.find_or_add_conf'):
            cur = self._db.cursor()
            cur.execute('BEGIN TRANSACTION')
            conf_id = self._find_or_add_conf(cur, conf, init_neighbors)
            cur.execute('COMMIT')
            return conf_id

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

    def _add_run(
        self,
        conf: Configuration2,
        run: Run,
        conf_id: Optional[int],
        timestamp: Optional[datetime],
        update_neighbors: bool,
    ):
        cur = self._db.cursor()
        cur.execute('BEGIN TRANSACTION')

        if conf_id is None:
            conf_id = self._find_or_add_conf(cur, conf, update_neighbors)

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
        conf: Configuration2,
        run: Run,
        conf_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        update_neighbors: bool = True,
    ):
        assert run.train_time is None or run.train_time > 0.0001
        with latency.timer('ResultDB.add_run'):
            self._add_run(conf, run, conf_id, timestamp, update_neighbors)

    def top_confs_global(self, template: Template):
        assert type(template) is Template

        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')
        conditions.append('num_runs > 1')

        conf_fields = ', '.join([
            'spec', 'precision', 'optimizer', 'lr',
            'decay', 'length', 'batch', 'steps'])

        where = 'WHERE ' + ' AND '.join(conditions)

        query = f"""
            WITH ranked_conf AS (
                SELECT id,
                       {conf_fields},
                       weights,
                       median_score,
                       (SELECT COUNT(*) FROM run WHERE conf_id = ranked_conf.id) AS num_runs,
                       ROW_NUMBER() OVER (PARTITION BY weights ORDER BY median_score) AS rn
                FROM conf
                {where}
            )
            SELECT id as conf_id,
                   {conf_fields},
                   median_score,
                   num_runs,
                   (SELECT system FROM conf_time WHERE conf_id=ranked_conf.id
                    ORDER BY median_time LIMIT 1) AS system,
                   (SELECT MIN(median_time) FROM conf_time WHERE conf_id=ranked_conf.id) AS median_time
            FROM ranked_conf
            WHERE rn = 1
        """

        cur = self._db.execute(query, params)

        best_score = INF

        for row in cur:
            conf_score = ConfScore._from_row(row)
            if conf_score.median_score < best_score:
                best_score = conf_score.median_score
                yield conf_score


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
                optimizer, decay, steps, median_score,
                (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                (SELECT median_time FROM conf_time
                 WHERE conf_id = conf.id AND system = :system
                 ORDER BY median_time LIMIT 1) AS median_time
            FROM conf {where} ORDER BY median_score ASC {limit}
            """

        cur = self._db.execute(query, params)

        for row in cur:
            yield ConfScore._from_row(row, system)

    def get_conf_runs_diff_steps(
        self, conf: Configuration2, system: str
    ) -> Iterable[tuple[int, float]]:
        """Find all runs on a given system of configurations which differ from
        `conf` only in the number of steps.
        """
        cur = self._db.execute(
            _GET_CONF_RUNS_DIFF_STEPS,
            {
                'spec': str(conf.model),
                'precision': str(conf.precision),
                'lr': conf.lr,
                'length': conf.length,
                'batch': conf.batch,
                'optimizer': conf.optimizer,
                'decay': conf.decay,
                'system': system,
            },
        )

        return cur

    def get_neighbors_by_runs(
        self, conf_id: int, system: str
    ) -> Iterable[ConfScore]:
        cur = self._db.cursor()
        cur.execute('BEGIN TRANSACTION')
        cur.execute(
            'SELECT has_neighbors FROM conf WHERE id = :conf_id',
            {'conf_id': conf_id},
        )
        has_neighbors = cur.fetchone()[0]
        if not has_neighbors:
            cur.execute(_GET_CONF, {'conf_id': conf_id})
            row = cur.fetchone()
            conf = Configuration2.from_dict(row)
            self._init_neighbors(cur, conf, conf_id)
        cur.execute(_GET_NEIGHBORS_BY_RUNS,
                    {'conf_id': conf_id, 'system': system})
        rows = cur.fetchall()
        cur.execute('COMMIT')
        for row in rows:
            yield ConfScore._from_row(row, system)

    def total_runs(self) -> int:
        cur = self._db.execute('SELECT COUNT(*) AS count FROM run')
        total = cur.fetchone()['count']
        assert isinstance(total, int)
        return total

    def get_confs_runs(
        self, with_timestamps: bool = False
    ) -> Iterable[tuple[int, Configuration2, Run]]:
        cur = self._db.cursor()
        cur.execute(_GET_CONFS_RUNS)

        for row in cur:
            with latency.timer('ResultDB.get_confs_runs-row'):
                conf_id = row['conf_id']

                model = build_model(row['spec'])
                conf = Configuration2(model=model, precision=row['precision'],
                                      lr=row['lr'], length=row['length'],
                                      batch=row['batch'], steps=row['steps'],
                                      optimizer=row['optimizer'], decay=row['decay'])

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
        self, conf: Configuration2, run: Run, timestamp: datetime
    ) -> bool:
        """Check if a run with the same configuration and timestamp exists."""
        conf_id = self.find_or_add_conf(conf, init_neighbors=False)
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
