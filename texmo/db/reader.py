"""Read-only access to the results DB.

`DbReader` opens the SQLite file with `mode=ro` (URI form). Each
thread that needs DB reads instantiates its own — per-request for
report handlers, persistent for Search and Model threads. The class
is single-connection by construction; concurrent readers in the same
process get separate `DbReader` instances.
"""

import pickle
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .. import latency
from ..common import INF
from ..configuration import Configuration, Template
from ..precision import Precision
from ..run import Run
from .common import (
    FIND_CONF,
    _build_loss_trend,
    _make_template_conditions,
    _unpack_ndarray,
    open_connection,
)


# --- return-value dataclasses (only produced by reads) ----------------------


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
    def _from_row(row: sqlite3.Row, system: Optional[str] = None) -> 'ConfScore':
        if system is None:
            system = row['system']
        conf = Configuration.from_dict(row)
        return ConfScore(
            row['conf_id'],
            conf,
            median_score=row['median_score'],
            system=system,
            median_time=row['median_time'],
            num_runs=row['num_runs'],
        )


@dataclass
class ConfWithRuns:
    """Candidate for the time-budget search strategy.

    Unlike ConfScore, carries both total and per-system run counts and
    a time estimate from any source (median or predicted).
    """
    conf_id: int
    conf: Configuration
    median_score: float
    time_estimate: float
    total_runs: int
    system_runs: int


# --- SQL constants (read-only side) -----------------------------------------


GET_CONFS_RUNS = """
SELECT conf.id AS conf_id,
        spec,
        precision,
        lr,
        length,
        batch,
        steps,
        decay,
        cosine,
        run.id AS run_id,
        system,
        -- 0 is the schema's "no usable time" sentinel; surface it as
        -- NULL so downstream code can use the standard `is None` check.
        NULLIF(train_time, 0) AS train_time,
        timestamp,
        loss,
        step_loss,
        loss_model_v,
        loss_model
FROM conf, run
WHERE conf.id = run.conf_id
"""


class DbReader(object):
    """Read-only handle to the results DB."""

    @staticmethod
    def from_args(db: Optional[str]) -> 'DbReader':
        return DbReader(db)

    def __init__(self, path: Optional[str]):
        assert path is not None, "DbReader needs a real path (mode=ro)"
        self._path = path
        self._db = open_connection(path, readonly=True)
        # Positive-only cache: once a conf has an ID, that ID is
        # permanent. Misses always hit disk so writers adding a new
        # conf become visible on the next lookup.
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

    # --- single-conf lookups ------------------------------------------------

    def get_conf_id(self, conf: Configuration) -> int | None:
        """Return conf_id if this configuration exists in the DB, else None.

        Uses an in-memory cache for positive lookups only.
        """
        conf_id = self._conf_id_cache.get(conf)
        if conf_id is not None:
            return conf_id

        cur = self._db.execute(FIND_CONF, conf.to_dict())
        row = cur.fetchone()
        if row is None:
            return None
        conf_id = row[0]
        self._conf_id_cache[conf] = conf_id
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

    def get_systems(self) -> list[str]:
        """Return a sorted list of all systems that have runs in the DB."""
        cur = self._db.execute(
            'SELECT DISTINCT system FROM run ORDER BY system')
        return [row[0] for row in cur]

    def has_covering_run(
        self, conf: Configuration, min_steps: int, system: str,
    ) -> bool:
        """True if `system` has a run with the same (spec, lr, length,
        batch, precision, decay, cosine) as `conf` and at least
        `min_steps` steps. Used by the coverage-walk strategy to skip
        confs that already have a comparable or longer run on the
        target system."""
        params = dict(conf.to_dict())
        params['min_steps'] = min_steps
        params['system'] = system
        cur = self._db.execute(
            '''SELECT 1 FROM run
               JOIN conf ON run.conf_id = conf.id
               WHERE conf.spec = :spec AND conf.lr = :lr
                 AND conf.length = :length AND conf.batch = :batch
                 AND conf.precision = :precision
                 AND conf.decay = :decay AND conf.cosine = :cosine
                 AND conf.steps >= :min_steps
                 AND run.system = :system
               LIMIT 1''',
            params,
        )
        return cur.fetchone() is not None

    # --- search-strategy queries -------------------------------------------

    def top_confs_global(
        self, template: Template, system: Optional[str] = None,
        max_weights: Optional[int] = None,
        max_time: Optional[float] = None,
    ):
        """Yield top confs ordered by weights, one per weight count.

        Args:
            template: filter for the configurations.
            system: if provided, only include configurations that have been
                run on this system, and report median_time for this system
                only. Score is still the global median across all systems.
            max_weights: if provided, exclude confs with more weights.
            max_time: if provided, exclude confs whose smallest
                across-systems median_time is above this. Used by the
                compare page; means "at least one system trains this in
                under `max_time`".
        """
        assert type(template) is Template

        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')
        conditions.append('num_runs > 1')

        if max_weights is not None:
            conditions.append('weights <= :max_weights')
            params['max_weights'] = max_weights

        if max_time is not None:
            conditions.append(
                "(SELECT MIN(time_s) FROM conf_time_estimate "
                " WHERE conf_id=conf.id AND source='median')"
                " <= :max_time")
            params['max_time'] = max_time

        if system is not None:
            conditions.append(
                'EXISTS (SELECT 1 FROM run '
                'WHERE run.conf_id = conf.id AND run.system = :system)')
            params['system'] = system

        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'cosine', 'length', 'batch', 'steps'])

        where = 'WHERE ' + ' AND '.join(conditions)

        if system is None:
            # Best median_time across all systems, report the winning system.
            time_select = (
                "(SELECT system FROM conf_time_estimate "
                " WHERE conf_id=conf.id AND source='median' "
                " ORDER BY time_s LIMIT 1) AS system, "
                "(SELECT MIN(time_s) FROM conf_time_estimate "
                " WHERE conf_id=conf.id AND source='median') AS median_time"
            )
        else:
            # Median_time on the selected system specifically.
            time_select = (
                ":system AS system, "
                "(SELECT time_s FROM conf_time_estimate "
                " WHERE conf_id=conf.id AND system=:system "
                " AND source='median') AS median_time"
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
            'decay', 'cosine', 'length', 'batch', 'steps'])

        segments: list[tuple[int, Optional[int], ConfScore]] = []

        n = len(pareto)
        ws = [cs.conf.model.num_weights for cs in pareto]
        ls = [cs.median_score for cs in pareto]

        for i in range(n):
            w_left = ws[i]
            w_right: Optional[int] = ws[i + 1] if i + 1 < n else None
            threshold = ls[i] * (1.0 + tolerance)

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
                       ct.time_s AS median_time
                FROM conf
                JOIN conf_time_estimate AS ct
                    ON ct.conf_id = conf.id AND ct.system = :system
                    AND ct.source = 'median'
                {where}
                ORDER BY ct.time_s ASC
            """
            cur = self._db.execute(query, params)

            uncovered_right: Optional[int] = w_right
            done = False
            for row in cur:
                w = row['weights']
                if uncovered_right is not None and w >= uncovered_right:
                    continue
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
            params['max_weight'] = max_weights

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
                decay, cosine, steps, median_score,
                (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                (SELECT time_s FROM conf_time_estimate
                 WHERE conf_id = conf.id AND system = :system
                   AND source = 'median'
                 ORDER BY time_s LIMIT 1) AS median_time
            FROM conf {where} ORDER BY median_score ASC {limit}
            """

        cur = self._db.execute(query, params)

        for row in cur:
            yield ConfScore._from_row(row, system)

    def best_conf_for_spec_on_system(
        self,
        spec: str,
        system: str,
        max_time: float,
    ) -> Optional[ConfScore]:
        """Lowest-`median_score` conf matching `spec` exactly that has a
        `median` time row on `system` no greater than `max_time`.

        Used for the per-system throughput report: with the model
        architecture pinned, we still let each system pick its own
        optimal (batch, length, lr, steps).
        """
        query = """
            SELECT conf.id AS conf_id, spec, precision, lr, length, batch,
                   decay, cosine, steps, median_score,
                   (SELECT COUNT(*) FROM run
                    WHERE conf_id = conf.id) AS num_runs,
                   ct.time_s AS median_time
            FROM conf
            JOIN conf_time_estimate ct
                ON ct.conf_id = conf.id AND ct.system = :system
                   AND ct.source = 'median'
            WHERE conf.spec = :spec
              AND conf.median_score IS NOT NULL
              AND ct.time_s <= :max_time
            ORDER BY conf.median_score ASC, ct.time_s ASC
            LIMIT 1
        """
        cur = self._db.execute(query, {
            'spec': spec, 'system': system, 'max_time': max_time,
        })
        row = cur.fetchone()
        if row is None:
            return None
        return ConfScore._from_row(row, system)

    def confs_under_time(
        self,
        template: Template,
        system: str,
        max_weights: int,
        max_time: float,
    ) -> Iterable[ConfWithRuns]:
        """Yield scored confs with time_estimate <= max_time, ordered by score.

        Time estimate comes from `conf_time_estimate` regardless of
        source (median or predicted). Confs without any score
        (median_score IS NULL) are excluded.
        """
        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')
        conditions.append('weights <= :max_weights')
        conditions.append('cte.time_s <= :max_time')
        params['max_weights'] = max_weights
        params['max_time'] = max_time
        params['system'] = system
        where = 'WHERE ' + ' AND '.join(conditions)

        query = f"""
            SELECT conf.id AS conf_id,
                   spec, precision, lr, decay, cosine, length, batch, steps,
                   median_score,
                   cte.time_s AS time_estimate,
                   (SELECT COUNT(*) FROM run
                    WHERE conf_id = conf.id) AS total_runs,
                   (SELECT COUNT(*) FROM run
                    WHERE conf_id = conf.id AND system = :system
                   ) AS system_runs
            FROM conf
            JOIN conf_time_estimate cte
                ON cte.conf_id = conf.id AND cte.system = :system
            {where}
            ORDER BY median_score ASC
        """
        cur = self._db.execute(query, params)
        for row in cur:
            yield ConfWithRuns(
                conf_id=row['conf_id'],
                conf=Configuration.from_dict(row),
                median_score=row['median_score'],
                time_estimate=row['time_estimate'],
                total_runs=row['total_runs'],
                system_runs=row['system_runs'],
            )

    # --- bulk iteration / aggregates ---------------------------------------

    def total_runs(self) -> int:
        cur = self._db.execute('SELECT COUNT(*) AS count FROM run')
        total = cur.fetchone()['count']
        assert isinstance(total, int)
        return total

    def get_confs_runs(
        self, with_timestamps: bool = False
    ) -> Iterable[tuple[int, Configuration, Run]]:
        cur = self._db.cursor()
        cur.execute(GET_CONFS_RUNS)

        for row in cur:
            with latency.timer('DbReader.get_confs_runs-row'):
                conf_id = row['conf_id']
                conf = Configuration.from_dict(row)

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

    def get_runs_for_timing(
        self, system: str, precision: Precision
    ) -> Iterable[tuple[Configuration, Run]]:
        """Yield (conf, run) pairs for fitting a timing model.

        Only the fields the timing model needs are populated on Run:
        system and train_time. step_loss/loss/loss_trend are omitted.
        """
        cur = self._db.execute(
            """
            SELECT conf.spec AS spec, conf.precision AS precision,
                   conf.lr AS lr, conf.length AS length, conf.batch AS batch,
                   conf.steps AS steps, conf.decay AS decay,
                   conf.cosine AS cosine,
                   -- See GET_CONFS_RUNS: surface the 0 sentinel as NULL.
                   NULLIF(run.train_time, 0) AS train_time
            FROM conf JOIN run ON conf.id = run.conf_id
            WHERE run.system = :system AND conf.precision = :precision
            """,
            {'system': system, 'precision': str(precision)},
        )
        for row in cur:
            conf = Configuration.from_dict(row)
            run = Run(system=system, train_time=row['train_time'])
            yield conf, run

    def iter_labeled_runs(
        self,
    ) -> Iterable[tuple[int, Configuration, float]]:
        """Yield (conf_id, Configuration, run_loss) for every run with a loss.

        Used by loss-prediction training: one example per run, the
        caller decides how to split (typically by conf_id).
        """
        cur = self._db.execute(
            """
            SELECT conf.id AS conf_id,
                   spec, precision, lr, length, batch, steps, decay, cosine,
                   run.loss AS loss
            FROM conf JOIN run ON run.conf_id = conf.id
            WHERE run.loss IS NOT NULL
            """
        )
        for row in cur:
            yield row['conf_id'], Configuration.from_dict(row), row['loss']

    def iter_confs_by_precision(
        self, precision: Precision
    ) -> Iterable[tuple[int, Configuration]]:
        """Yield (conf_id, Configuration) for every conf of the given precision."""
        cur = self._db.execute(
            """
            SELECT id, spec, precision, lr, length, batch, steps, decay, cosine
            FROM conf WHERE precision = :precision
            """,
            {'precision': str(precision)},
        )
        for row in cur:
            yield row['id'], Configuration.from_dict(row)

    def get_conf_ids_with_median_time(
        self, system: str, precision: Precision
    ) -> set[int]:
        """Return the set of conf_ids with a median_time for this (system, precision)."""
        cur = self._db.execute(
            """
            SELECT conf.id
            FROM conf JOIN conf_time_estimate ct ON conf.id = ct.conf_id
            WHERE ct.system = :system AND conf.precision = :precision
              AND ct.source = 'median'
            """,
            {'system': system, 'precision': str(precision)},
        )
        return {row[0] for row in cur}

    def get_losses_by_conf_ids(
        self, conf_ids: list[int],
    ) -> dict[int, list[float]]:
        """Return {conf_id: [loss, loss, ...]} across all systems."""
        if not conf_ids:
            return {}
        placeholders = ','.join('?' for _ in conf_ids)
        cur = self._db.execute(
            f'SELECT conf_id, loss FROM run '
            f'WHERE conf_id IN ({placeholders}) AND loss IS NOT NULL',
            conf_ids,
        )
        out: dict[int, list[float]] = {cid: [] for cid in conf_ids}
        for cid, loss in cur:
            out[cid].append(loss)
        return out

    # --- single-row lookups -------------------------------------------------

    def load_model(self, name: str):
        """Return the unpickled model stored under `name`, or None."""
        cur = self._db.execute(
            'SELECT data FROM model WHERE name = :name',
            {'name': name},
        )
        row = cur.fetchone()
        if row is None:
            return None
        return pickle.loads(row['data'])

    def get_time_estimate(
        self, conf_id: int, system: str
    ) -> Optional[tuple[float, str]]:
        """Return (time_s, source) or None if no estimate stored."""
        cur = self._db.execute(
            'SELECT time_s, source FROM conf_time_estimate '
            'WHERE conf_id = :conf_id AND system = :system',
            {'conf_id': conf_id, 'system': system},
        )
        row = cur.fetchone()
        if row is None:
            return None
        return row['time_s'], row['source']
