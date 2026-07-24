"""Read-only access to the results DB.

`DbReader` opens the SQLite file with `mode=ro` (URI form). Each
thread that needs DB reads instantiates its own — per-request for
report handlers, persistent for Search and Model threads. The class
is single-connection by construction; concurrent readers in the same
process get separate `DbReader` instances.
"""

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

        The DB already has a UNIQUE covering index over every conf
        field FIND_CONF queries on, so this lookup is ~4 us -- not
        worth caching at the Python level (the previous
        Configuration-keyed cache was the main culprit behind
        long-running memory growth on the search/model threads).
        """
        cur = self._db.execute(FIND_CONF, conf.to_dict())
        row = cur.fetchone()
        if row is None:
            return None
        return row[0]

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
        self, conf: Configuration, system: str,
    ) -> bool:
        """True if `system` has a run with the same (spec, lr, length,
        batch, precision, decay, cosine) as `conf` and at least
        `conf.steps` steps. Used by the coverage-walk strategy to skip
        confs that already have a comparable or longer run on the
        target system."""
        params = dict(conf.to_dict())
        params['min_steps'] = conf.steps
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
        min_num_runs: int = 2,
        include_invalid: bool = False,
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
            min_num_runs: minimum total run count (across all systems)
                for a conf to be included. Default 2 acts as a
                single-run-outlier guard for display + throughput; the
                coverage walk passes 1 so even fresh top confs surface
                for cross-system verification.
            include_invalid: when True, yield confs even if
                `model.is_valid()` is False. Default skips them — they
                shouldn't be shown to the user, picked by search, or
                included in reports. Migrations that want to act on
                invalid Pareto entries (e.g. the strip-leading-norm
                cleanup) opt in.
        """
        assert type(template) is Template

        # Walk conf in (weights ASC, median_score ASC) order using the
        # conf_weights_score index, then resolve num_runs / cte info
        # per candidate. The previous CTE+window form materialised the
        # whole conf table (384k rows on the live DB) with four
        # correlated subqueries per row -- 4-7s. This form is sub-200
        # ms because we short-circuit per weight bucket as soon as a
        # bucket is known to not improve the running Pareto best.
        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')

        if max_weights is not None:
            conditions.append('weights <= :max_weights')
            params['max_weights'] = max_weights

        if system is not None:
            conditions.append(
                'EXISTS (SELECT 1 FROM run '
                'WHERE run.conf_id = conf.id AND run.system = :system)')
            params['system'] = system

        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'cosine', 'length', 'batch', 'steps'])
        where = 'WHERE ' + ' AND '.join(conditions)

        # Per-candidate enrichment queries (small, indexed).
        q_runs = 'SELECT COUNT(*) FROM run WHERE conf_id = ?'
        if system is None:
            # Best (lowest time) median entry across all systems.
            q_cte = (
                "SELECT system, time_s FROM conf_time_estimate "
                "WHERE conf_id = ? AND source = 'median' "
                "ORDER BY time_s LIMIT 1"
            )
        else:
            q_cte = (
                "SELECT :system AS system, time_s "
                "FROM conf_time_estimate "
                "WHERE conf_id = :id AND source = 'median' "
                "AND system = :system"
            )

        # INDEXED BY hint locks in the (weights, median_score) walk so
        # the planner doesn't choose the unrelated conf_median_score
        # index (which would order by score alone, defeating the
        # per-bucket short-circuit).
        query = f"""
            SELECT id, {conf_fields}, weights, median_score
            FROM conf INDEXED BY conf_weights_score
            {where}
            ORDER BY weights ASC, median_score ASC
        """

        cur = self._db.execute(query, params)

        best_score = INF
        cur_weight = -1
        bucket_done = False

        for row in cur:
            w = row['weights']
            if w != cur_weight:
                cur_weight = w
                bucket_done = False
            elif bucket_done:
                continue

            score = row['median_score']
            if score >= best_score:
                # This bucket's best (we walk in score ASC) can't
                # improve the running Pareto loss; skip the rest of
                # the bucket without paying for any per-row lookups.
                bucket_done = True
                continue

            # Per-bucket filters: total runs and cte time.
            if min_num_runs > 0:
                runs = self._db.execute(
                    q_runs, (row['id'],)).fetchone()[0]
                if runs < min_num_runs:
                    continue
            else:
                runs = self._db.execute(
                    q_runs, (row['id'],)).fetchone()[0]

            if system is None:
                cte_row = self._db.execute(
                    q_cte, (row['id'],)).fetchone()
            else:
                cte_row = self._db.execute(
                    q_cte, {'id': row['id'], 'system': system}
                ).fetchone()
            if cte_row is None:
                continue
            cte_system, cte_time = cte_row[0], cte_row[1]
            if max_time is not None and cte_time > max_time:
                continue

            # Build the enriched row dict for ConfScore._from_row.
            enriched = {
                'conf_id': row['id'],
                'spec': row['spec'],
                'precision': row['precision'],
                'lr': row['lr'],
                'decay': row['decay'],
                'cosine': row['cosine'],
                'length': row['length'],
                'batch': row['batch'],
                'steps': row['steps'],
                'median_score': score,
                'num_runs': runs,
                'system': cte_system,
                'median_time': cte_time,
            }
            conf_score = ConfScore._from_row(enriched)
            if (not include_invalid
                    and not conf_score.conf.model.is_valid()):
                # Don't mark bucket done -- a higher-score valid conf
                # at the same weight could still surface from the
                # bucket. (We've already returned this row from SQL,
                # so the next iteration moves on naturally.)
                continue

            best_score = score
            bucket_done = True
            yield conf_score

    def pick_me_conf(
        self, template: Template, min_runs: int = 2,
    ) -> Optional[Configuration]:
        """Return one random pick_me conf with fewer than `min_runs`
        total runs and matching the template, or None.

        The min-runs filter is computed on the fly (SELECT COUNT(*)
        FROM run WHERE conf_id=...) so the writer doesn't have to
        clear the pick_me flag — once a conf has enough runs it's
        simply skipped by this query and effectively retired from
        priority pick.
        """
        conditions, params = _make_template_conditions(template)
        conditions.append('pick_me = 1')
        conditions.append(
            '(SELECT COUNT(*) FROM run WHERE conf_id = conf.id) '
            '< :min_runs')
        params['min_runs'] = min_runs
        where = 'WHERE ' + ' AND '.join(conditions)
        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'cosine', 'length', 'batch', 'steps'])
        query = (
            f'SELECT {conf_fields} FROM conf {where} '
            'ORDER BY RANDOM()'
        )
        for row in self._db.execute(query, params):
            conf = Configuration.from_dict(row)
            # Skip confs that no longer pass validity (e.g. under the
            # Model2 rules after the representation switch) -- the search
            # can't build/run them. The other DB-extraction paths filter
            # the same way; pick_me is the last one to honor it.
            if conf.model.is_valid():
                return conf
        return None

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
                cs = ConfScore._from_row(row)
                # Skip invalid candidates -- the Pareto inputs are
                # already filtered (default `include_invalid=False`),
                # so the w_left Pareto point itself remains in the
                # query results and the `done` assertion still holds.
                if not cs.conf.model.is_valid():
                    continue
                seg_low = max(w, w_left)
                segments.append((seg_low, uncovered_right, cs))
                uncovered_right = seg_low
                if w <= w_left:
                    done = True
                    break
            assert done, (
                f"interval [{w_left}, {w_right}) not fully covered — "
                "the Pareto point itself should have been in the results")

        segments.sort(key=lambda seg: seg[0])
        return segments

    def fastest_near_best_segments_any_system(
        self,
        template: Template,
        pareto: list[ConfScore],
        tolerance: float = 0.01,
    ) -> list[tuple[int, Optional[int], ConfScore]]:
        """Like `fastest_near_best_segments` but considers every system
        — for each qualifying conf, the system with the lowest
        median_time wins, and that (system, time) is attached to the
        returned `ConfScore`. Used by the cross-system fastest report
        to answer "across all systems, what's the cheapest way to
        reach near-best loss at this weight budget?".
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
            if w_right is not None:
                conditions.append('weights < :w_right')
                params['w_right'] = w_right
            where = 'WHERE ' + ' AND '.join(conditions)

            # Fetch every (conf, system) pair with a `median` cte row;
            # ORDER BY ct.time_s ASC means the first occurrence of each
            # conf_id is the system with the fastest time. We dedupe in
            # Python because per-conf MIN-with-tied-system is awkward
            # in vanilla SQL.
            query = f"""
                SELECT conf.id AS conf_id, {conf_fields},
                       weights,
                       median_score,
                       (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                       ct.system AS system,
                       ct.time_s AS median_time
                FROM conf
                JOIN conf_time_estimate AS ct
                    ON ct.conf_id = conf.id AND ct.source = 'median'
                {where}
                ORDER BY ct.time_s ASC
            """
            cur = self._db.execute(query, params)

            # Same dedup-by-weight walk as `fastest_near_best_segments`,
            # but each conf appears once (we drop subsequent
            # per-system entries via `seen`).
            uncovered_right: Optional[int] = w_right
            done = False
            seen: set[int] = set()
            for row in cur:
                cid = row['conf_id']
                if cid in seen:
                    continue
                seen.add(cid)
                w = row['weights']
                if uncovered_right is not None and w >= uncovered_right:
                    continue
                cs = ConfScore._from_row(row)
                # Skip invalid candidates -- see the per-system
                # variant above for the same reasoning.
                if not cs.conf.model.is_valid():
                    continue
                seg_low = max(w, w_left)
                segments.append((seg_low, uncovered_right, cs))
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

        # INDEXED BY conf_median_score: walk confs in median_score order
        # and early-exit at the LIMIT, instead of the planner's default
        # (scan all confs under the weight cap, then temp-B-tree sort),
        # which touches a large fraction of the 400k-conf table every call
        # (~1.5s on prod). Only called on the search hot path with the
        # worker's active system, where the top conf usually qualifies ->
        # a shallow walk (ms-to-100ms). See `Search._select_max_weights`.
        query = f"""
            SELECT conf.id AS conf_id, spec, precision, lr, length, batch,
                decay, cosine, steps, median_score,
                (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
                (SELECT time_s FROM conf_time_estimate
                 WHERE conf_id = conf.id AND system = :system
                   AND source = 'median'
                 ORDER BY time_s LIMIT 1) AS median_time
            FROM conf INDEXED BY conf_median_score {where}
            ORDER BY median_score ASC {limit}
            """

        cur = self._db.execute(query, params)

        for row in cur:
            cs = ConfScore._from_row(row, system)
            # Invalid confs (e.g. norm-as-first-layer post-rename)
            # mustn't surface as candidates for the search strategies
            # that call this. Same convention as `top_confs_global`.
            if not cs.conf.model.is_valid():
                continue
            yield cs

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
        optimal (batch, length, lr, steps). Requires `num_runs > 1`
        across all systems — single-run scores are outliers and would
        otherwise make the throughput graph noisy when one system
        latched onto a thinly-sampled conf and another didn't.
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
              AND (SELECT COUNT(*) FROM run
                   WHERE conf_id = conf.id) > 1
            ORDER BY conf.median_score ASC, ct.time_s ASC
            LIMIT 1
        """
        cur = self._db.execute(query, {
            'spec': spec, 'system': system, 'max_time': max_time,
        })
        row = cur.fetchone()
        if row is None:
            return None
        cs = ConfScore._from_row(row, system)
        if not cs.conf.model.is_valid():
            return None
        return cs

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
        # EXISTS form so SQLite walks conf in `conf_median_score`
        # ASC order and stops at the first row the caller pulls.
        # The previous JOIN form had to filter+sort the entire cte
        # slice for the system first -- 3-10s on the live DB; this
        # is sub-100ms.
        conditions.append(
            'EXISTS (SELECT 1 FROM conf_time_estimate cte '
            '        WHERE cte.conf_id = conf.id '
            '          AND cte.system = :system '
            '          AND cte.time_s <= :max_time)')
        params['max_weights'] = max_weights
        params['max_time'] = max_time
        params['system'] = system
        where = 'WHERE ' + ' AND '.join(conditions)

        query = f"""
            SELECT conf.id AS conf_id,
                   spec, precision, lr, decay, cosine, length, batch, steps,
                   median_score,
                   (SELECT cte.time_s FROM conf_time_estimate cte
                    WHERE cte.conf_id = conf.id
                      AND cte.system = :system) AS time_estimate,
                   (SELECT COUNT(*) FROM run
                    WHERE conf_id = conf.id) AS total_runs,
                   (SELECT COUNT(*) FROM run
                    WHERE conf_id = conf.id AND system = :system
                   ) AS system_runs
            FROM conf
            {where}
            ORDER BY median_score ASC
        """
        cur = self._db.execute(query, params)
        for row in cur:
            conf = Configuration.from_dict(row)
            # Same invalid-conf filter as `top_confs_global`: the
            # time-budget search strategy must not return architectures
            # that fail `model.is_valid()` (e.g. the norm-as-first-
            # layer rows the rename migration leaves behind).
            if not conf.model.is_valid():
                continue
            yield ConfWithRuns(
                conf_id=row['conf_id'],
                conf=conf,
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

                # The per-step history is no longer stored; the trend
                # rebuilds from its fitted params alone (or stays
                # unfitted for rows without params).
                loss_trend = _build_loss_trend(
                    None,
                    row['loss_model_v'],
                    _unpack_ndarray(row['loss_model']),
                )

                run = Run(
                    id=row['run_id'],
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
        self, system: str, precision: Precision,
        limit: int | None = None,
    ) -> Iterable[tuple[Configuration, Run]]:
        """Yield (conf, run) pairs for fitting a timing model.

        Only the fields the timing model needs are populated on Run:
        system and train_time. step_loss/loss/loss_trend are omitted.
        With `limit`, only the most recent runs (by run id) are
        returned -- recent runs both bound the read/parse cost and
        reflect the machine's current performance.
        """
        cur = self._db.execute(
            f"""
            SELECT conf.spec AS spec, conf.precision AS precision,
                   conf.lr AS lr, conf.length AS length, conf.batch AS batch,
                   conf.steps AS steps, conf.decay AS decay,
                   conf.cosine AS cosine,
                   -- See GET_CONFS_RUNS: surface the 0 sentinel as NULL.
                   NULLIF(run.train_time, 0) AS train_time
            FROM conf JOIN run ON conf.id = run.conf_id
            WHERE run.system = :system AND conf.precision = :precision
            {'ORDER BY run.id DESC LIMIT :limit' if limit else ''}
            """,
            {'system': system, 'precision': str(precision), 'limit': limit},
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

    def iter_labeled_runs_raw(self) -> Iterable[tuple]:
        """Yield raw (spec, precision, lr, length, batch, steps, decay,
        cosine, loss) tuples for every run with a loss.

        Unlike `iter_labeled_runs` this skips Configuration parsing --
        used to stream training data to a client-side refit, where the
        client parses (deduplicating by spec+precision on its end).
        """
        cur = self._db.execute(
            """
            SELECT spec, precision, lr, length, batch, steps, decay,
                   cosine, run.loss AS loss
            FROM conf JOIN run ON run.conf_id = conf.id
            WHERE run.loss IS NOT NULL
            """
        )
        for row in cur:
            yield (row['spec'], row['precision'], row['lr'],
                   row['length'], row['batch'], row['steps'],
                   row['decay'], row['cosine'], row['loss'])

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

    def iter_confs_missing_estimate(
        self, system: str, precision: Precision
    ) -> Iterable[tuple[int, Configuration]]:
        """Yield (conf_id, Configuration) for confs of this precision
        with no estimate row at all (median or predicted) for this
        system -- i.e. confs added since the last estimate refresh.
        Parses only the matching rows, which keeps the incremental
        refresh path cheap (the full sweep re-parses everything)."""
        cur = self._db.execute(
            """
            SELECT id, spec, precision, lr, length, batch, steps, decay, cosine
            FROM conf WHERE precision = :precision
              AND NOT EXISTS (
                SELECT 1 FROM conf_time_estimate ct
                WHERE ct.conf_id = conf.id AND ct.system = :system
              )
            """,
            {'system': system, 'precision': str(precision)},
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
