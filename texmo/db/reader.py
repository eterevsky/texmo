"""Read-only access to the results DB.

`DbReader` opens the SQLite file with `mode=ro` (URI form). Each
thread that needs DB reads instantiates its own — per-request for
report handlers, persistent for Search and Model threads. The class
is single-connection by construction; concurrent readers in the same
process get separate `DbReader` instances.
"""

import logging
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import Any, Optional

import numpy as np

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
from .conf_index import (
    SHADOW_EVERY,
    ConfIndex,
    ConfIndexUnavailable,
    ConfIndexUnsupported,
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


@dataclass
class PickMeCandidate:
    """A pick_me conf still short of its target, with its current
    total run count (all systems). See `DbReader.pick_me_candidates`."""
    conf_id: int
    conf: Configuration
    num_runs: int


# --- SQL constants (read-only side)-----------------------------------------


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


# Output rows for the index-backed queries, fetched by id once the
# index has picked and ordered the ids. Same columns (and the same
# per-row subqueries) as the SQL versions, plus `weights` for the
# drift check. `{marks}` is the `?` list for the ids.
_TOP_FOR_SYSTEM_ROWS = """
    SELECT conf.id AS conf_id, spec, precision, lr, length, batch,
        decay, cosine, steps, median_score, weights,
        (SELECT COUNT(*) FROM run WHERE conf_id = conf.id) AS num_runs,
        (SELECT time_s FROM conf_time_estimate
         WHERE conf_id = conf.id AND system = ?
           AND source = 'median'
         ORDER BY time_s LIMIT 1) AS median_time
    FROM conf WHERE id IN ({marks})
"""

_UNDER_TIME_ROWS = """
    SELECT conf.id AS conf_id,
           spec, precision, lr, decay, cosine, length, batch, steps,
           median_score, weights,
           (SELECT cte.time_s FROM conf_time_estimate cte
            WHERE cte.conf_id = conf.id
              AND cte.system = ?) AS time_estimate,
           (SELECT COUNT(*) FROM run
            WHERE conf_id = conf.id) AS total_runs,
           (SELECT COUNT(*) FROM run
            WHERE conf_id = conf.id AND system = ?
           ) AS system_runs
    FROM conf WHERE id IN ({marks})
"""

# Rows per output fetch: lazy queries start small (most callers stop
# after a few) and double up to the cap.
_FETCH_FIRST = 16
_FETCH_MAX = 512
# A lazy (unlimited) query's shadow check compares this long a prefix.
_SHADOW_PREFIX = 32


class _IndexDrift(Exception):
    """A row the index selected doesn't satisfy the query in SQL --
    the index disagrees with the snapshot it claims to describe."""


@dataclass
class _IndexQuery:
    """One index-backed query, as `DbReader._index_query` runs it.

    `ids` picks and orders the conf ids from the synced index; `sql`
    (with `params` before the ids) fetches their output rows; `check`
    returns a drift description if a fetched row contradicts the
    index; `make` builds the result object (None skips the row -- the
    validity filter); `key` is what the shadow check compares;
    `reference` runs the SQL version of the same query.
    """
    name: str
    args: dict[str, Any]
    limit: Optional[int]
    ids: Callable[[ConfIndex], np.ndarray]
    sql: str
    params: list
    check: Callable[[sqlite3.Row, ConfIndex], Optional[str]]
    make: Callable[[sqlite3.Row], Any]
    key: Callable[[Any], tuple]
    reference: Callable[[], Iterator]

    def describe(self) -> str:
        args = ', '.join(
            f'{k}={v!s}' if k == 'template' else f'{k}={v!r}'
            for k, v in self.args.items())
        return f'{self.name}({args})'


class DbReader(object):
    """Read-only handle to the results DB.

    With `conf_index=True` the hot ranked queries
    (`top_confs_for_system`, `confs_under_time`) are answered from an
    in-memory `ConfIndex` owned by this reader -- so by the thread
    that owns the reader (the SearchThread in the server); every other
    reader stays on SQL. Every `shadow_every`-th index-backed query
    also runs the SQL version and compares (0 disables).
    """

    @staticmethod
    def from_args(db: Optional[str]) -> 'DbReader':
        return DbReader(db)

    def __init__(
        self,
        path: Optional[str],
        conf_index: bool = False,
        shadow_every: int = SHADOW_EVERY,
    ):
        assert path is not None, "DbReader needs a real path (mode=ro)"
        self._path = path
        self._db = open_connection(path, readonly=True)
        self._index: Optional[ConfIndex] = None
        self._shadow_every = shadow_every
        self._index_queries = 0
        self._shadow_checks = 0
        self._shadow_mismatches = 0
        self._unsupported_logged: set[str] = set()
        # Nesting depth of `_snapshot()`; the outermost one owns the
        # BEGIN/COMMIT. See `_snapshot`.
        self._txn_depth = 0
        if conf_index:
            self.enable_conf_index()

    # --- conf index ---------------------------------------------------------

    def enable_conf_index(self) -> bool:
        """Attach and build a `ConfIndex` (seconds on the full DB, so
        the owning thread pays it up front rather than on its first
        query). False -- and the reader stays on SQL -- if the DB has
        no changelog machinery (never opened by a current `DbWriter`).
        """
        index = ConfIndex()
        try:
            with self._snapshot():
                index.sync(self._db)
        except ConfIndexUnavailable as e:
            logging.warning(f'ConfIndex unavailable, staying on SQL: {e}')
            return False
        self._index = index
        return True

    @property
    def conf_index(self) -> Optional[ConfIndex]:
        return self._index

    @property
    def shadow_stats(self) -> tuple[int, int]:
        """(shadow checks run, mismatches seen)."""
        return self._shadow_checks, self._shadow_mismatches

    @contextmanager
    def _snapshot(self) -> Iterator[None]:
        """Pin one read snapshot for the enclosed statements.

        An explicit BEGIN (the connection is in the sqlite3 module's
        legacy transaction mode, where SELECTs alone never open one)
        makes every statement until COMMIT read the same snapshot: the
        index sync and the row fetches that follow must agree, or a
        write committed in between could hand back a row the index
        didn't vet. Re-entrant by depth count rather than by
        `in_transaction`, because index-backed results are generators
        whose lifetimes can interleave: whichever leaves last commits,
        and anything nested joins the outer snapshot.
        """
        if self._txn_depth == 0:
            self._db.execute('BEGIN')
        self._txn_depth += 1
        try:
            yield
        finally:
            self._txn_depth -= 1
            if self._txn_depth == 0:
                self._db.commit()

    def _index_query(self, q: _IndexQuery) -> Iterator:
        """Run `q` against the index inside one pinned snapshot.

        A limited query is materialized whole inside the snapshot and
        the transaction closed before the first yield, so the read
        transaction lasts one query. An unlimited one (time-budget's
        lazy scan) keeps its snapshot until the caller drops the
        generator -- the same lifetime the SQL cursor it replaces had.
        """
        if q.limit is not None:
            with self._snapshot():
                results = self._index_eager(q)
            yield from results
        else:
            with self._snapshot():
                yield from self._index_lazy(q)

    def _index_ids(self, q: _IndexQuery) -> Optional[np.ndarray]:
        """Sync and pick ids, or None to answer `q` from SQL."""
        index = self._index
        if index is None:
            return None
        try:
            index.sync(self._db)
        except ConfIndexUnavailable as e:
            logging.error(
                f'ConfIndex disabled, falling back to SQL for good: {e}')
            self._index = None
            return None
        try:
            return q.ids(index)
        except ConfIndexUnsupported as e:
            if str(e) not in self._unsupported_logged:
                self._unsupported_logged.add(str(e))
                logging.error(
                    f'ConfIndex cannot answer {q.describe()}: {e}; '
                    f'answering such queries from SQL')
            return None

    def _fetch(
        self, q: _IndexQuery, index: ConfIndex, ids: np.ndarray,
        first: int,
    ) -> Iterator:
        """Output objects for `ids`, in order, fetched by id in chunks
        of `first` doubling to `_FETCH_MAX`. Raises `_IndexDrift` on a
        row that contradicts the index."""
        pos, size = 0, first
        while pos < len(ids):
            chunk = ids[pos:pos + size].tolist()
            pos += len(chunk)
            size = min(size * 2, _FETCH_MAX)
            marks = ','.join('?' * len(chunk))
            rows = {
                row['conf_id']: row for row in self._db.execute(
                    q.sql.format(marks=marks), q.params + chunk)}
            for cid in chunk:
                row = rows.get(cid)
                problem = ('conf row missing' if row is None
                           else q.check(row, index))
                if problem is not None:
                    raise _IndexDrift(f'conf {cid}: {problem}')
                obj = q.make(row)
                if obj is not None:
                    yield obj

    def _on_drift(self, q: _IndexQuery, index: ConfIndex, e: Exception):
        logging.error(
            f'ConfIndex drift in {q.describe()}: {e}. Rebuilding the '
            f'index; this query answers from SQL.')
        index.invalidate()

    def _shadow_due(self) -> bool:
        due = (self._shadow_every > 0
               and self._index_queries % self._shadow_every == 0)
        self._index_queries += 1
        return due

    def _index_eager(self, q: _IndexQuery) -> list:
        t0 = time.perf_counter()
        index = self._index
        with latency.timer(f'DbReader.{q.name}.index'):
            ids = self._index_ids(q)
            if ids is None:
                return list(q.reference())
            assert index is not None
            ids = ids[:q.limit]
            try:
                results = list(self._fetch(
                    q, index, ids, max(min(len(ids), _FETCH_MAX), 1)))
            except _IndexDrift as e:
                self._on_drift(q, index, e)
                return list(q.reference())
        if self._shadow_due():
            results = self._shadow(
                q, index, results, time.perf_counter() - t0, None)
        return results

    def _index_lazy(self, q: _IndexQuery) -> Iterator:
        t0 = time.perf_counter()
        index = self._index
        with latency.timer(f'DbReader.{q.name}.index'):
            ids = self._index_ids(q)
        if ids is None:
            yield from q.reference()
            return
        assert index is not None
        stream = self._fetch(q, index, ids, _FETCH_FIRST)
        yielded: set[int] = set()
        try:
            prefix = []
            if self._shadow_due():
                prefix = list(islice(stream, _SHADOW_PREFIX))
                checked = self._shadow(
                    q, index, prefix, time.perf_counter() - t0,
                    _SHADOW_PREFIX)
                if checked is not prefix:
                    # Mismatch: the whole answer comes from SQL.
                    yield from q.reference()
                    return
            for obj in prefix:
                yielded.add(obj.conf_id)
                yield obj
            for obj in stream:
                yielded.add(obj.conf_id)
                yield obj
        except _IndexDrift as e:
            # Finish from SQL in the same snapshot, without repeating
            # what the caller already has.
            self._on_drift(q, index, e)
            for obj in q.reference():
                if obj.conf_id not in yielded:
                    yield obj

    def _shadow(
        self, q: _IndexQuery, index: ConfIndex, results: list,
        index_s: float, n: Optional[int],
    ) -> list:
        """Production self-check: run the SQL that `q` replaced, inside
        the same snapshot, and compare the first `n` results (all of
        them for a limited query). Returns `results` itself when they
        agree; on a mismatch logs ERROR with the first differing
        positions, invalidates the index (full rebuild on the next
        query) and returns the SQL answer instead. One INFO line per
        check, so the log shows the index staying honest and what it
        saves."""
        t0 = time.perf_counter()
        with latency.timer(f'DbReader.{q.name}.shadow_sql'):
            ref_iter = q.reference()
            ref = list(ref_iter if n is None else islice(ref_iter, n))
            close = getattr(ref_iter, 'close', None)
            if close is not None:
                close()
        sql_s = time.perf_counter() - t0
        self._shadow_checks += 1
        got = [q.key(r) for r in results]
        want = [q.key(r) for r in ref]
        if got == want:
            logging.info(
                f'ConfIndex shadow ok #{self._shadow_checks}: '
                f'{q.describe()}: {len(got)} rows agree; index '
                f'{index_s * 1e3:.1f} ms, sql {sql_s * 1e3:.1f} ms')
            return results
        self._shadow_mismatches += 1
        diffs = [
            i for i in range(max(len(got), len(want)))
            if (got[i] if i < len(got) else None)
            != (want[i] if i < len(want) else None)]
        shown = '; '.join(
            f'[{i}] index={got[i] if i < len(got) else None} '
            f'sql={want[i] if i < len(want) else None}'
            for i in diffs[:5])
        logging.error(
            f'ConfIndex shadow MISMATCH #{self._shadow_checks} '
            f'({self._shadow_mismatches} of {self._shadow_checks} checks '
            f'so far): {q.describe()}: index {len(got)} rows, sql '
            f'{len(want)}; {len(diffs)} positions differ: {shown}. '
            f'Rebuilding the index; answering from SQL.')
        index.invalidate()
        return ref

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

    def pick_me_candidates(
        self, template: Template, min_runs: int = 2,
    ) -> list[PickMeCandidate]:
        """Every pick_me conf still short of its target run count and
        matching the template, in conf-id order (the caller draws).

        `pick_me` is the conf's own target: the row stays a priority
        pick until it has at least `pick_me` runs. Legacy rows carry
        `pick_me = 1` from when the column was a flag, so the
        effective target is `max(pick_me, min_runs)` — those keep
        their old "run it twice" meaning while a row asking for more
        (`texmo.py pick-me --runs N`) gets what it asked for.

        The run count is computed on the fly (SELECT COUNT(*) FROM run
        WHERE conf_id=...) so the writer doesn't have to clear the
        pick_me column — once a conf has enough runs it's simply
        skipped by this query and effectively retired from priority
        pick. It is also returned: the search compares it against the
        count it saw at dispatch to tell that a result has come back.
        """
        conditions, params = _make_template_conditions(template)
        conditions.append('pick_me >= 1')
        params['min_runs'] = min_runs
        where = 'WHERE ' + ' AND '.join(conditions)
        conf_fields = ', '.join([
            'spec', 'precision', 'lr',
            'decay', 'cosine', 'length', 'batch', 'steps'])
        query = (
            f'SELECT id, {conf_fields}, num_runs FROM ('
            f'SELECT id, {conf_fields}, pick_me, '
            '(SELECT COUNT(*) FROM run WHERE run.conf_id = conf.id) '
            f'AS num_runs FROM conf {where}'
            ') WHERE num_runs < MAX(pick_me, :min_runs) ORDER BY id'
        )
        candidates = []
        for row in self._db.execute(query, params):
            conf = Configuration.from_dict(row)
            # Skip confs that no longer pass validity (e.g. under the
            # Model2 rules after the representation switch) -- the search
            # can't build/run them. The other DB-extraction paths filter
            # the same way; pick_me is the last one to honor it.
            if conf.model.is_valid():
                candidates.append(
                    PickMeCandidate(row['id'], conf, row['num_runs']))
        return candidates

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
    ) -> Iterator[ConfScore]:
        """Scored confs matching `template`, best `median_score` first
        (ties by conf id), skipping invalid models.

        `max_weights` / `max_time` apply only when truthy; `max_time`
        bounds the conf's `median`-source time on `system`, so a conf
        never run there doesn't qualify. `median_time` on the results
        is that time (None without one). `limit` caps the rows
        considered *before* the validity skip, so fewer may come back.

        Answered from the `ConfIndex` when one is attached, else (and
        as the shadow reference) by `_top_confs_for_system_sql`.
        """
        assert type(system) is str
        assert type(template) is Template
        if self._index is None:
            return self._top_confs_for_system_sql(
                system, template, max_weights, max_time, limit)

        def ids(index: ConfIndex) -> np.ndarray:
            return index.ordered_ids(
                template, system,
                weights_max=max_weights if max_weights else None,
                time_max=max_time if max_time else None,
                median_time_only=True)

        def check(row: sqlite3.Row, index: ConfIndex) -> Optional[str]:
            if row['median_score'] != index.score_of(row['conf_id']):
                return (f"median_score {row['median_score']!r} != index "
                        f"{index.score_of(row['conf_id'])!r}")
            if max_weights and not row['weights'] <= max_weights:
                return f"weights {row['weights']} > {max_weights}"
            if max_time and not (row['median_time'] is not None
                                 and row['median_time'] <= max_time):
                return f"median_time {row['median_time']!r} > {max_time}"
            return None

        def make(row: sqlite3.Row) -> Optional[ConfScore]:
            cs = ConfScore._from_row(row, system)
            return cs if cs.conf.model.is_valid() else None

        return self._index_query(_IndexQuery(
            name='top_confs_for_system',
            args=dict(system=system, max_weights=max_weights,
                      max_time=max_time, limit=limit, template=template),
            limit=limit,
            ids=ids,
            sql=_TOP_FOR_SYSTEM_ROWS,
            params=[system],
            check=check,
            make=make,
            key=lambda cs: (
                cs.conf_id, cs.median_score, cs.median_time, cs.num_runs),
            reference=lambda: self._top_confs_for_system_sql(
                system, template, max_weights, max_time, limit),
        ))

    def confs_under_time(
        self,
        template: Template,
        system: str,
        max_weights: int,
        max_time: float,
    ) -> Iterator[ConfWithRuns]:
        """Yield scored confs with time_estimate <= max_time, ordered by
        score (ties by conf id), skipping invalid models.

        Time estimate comes from `conf_time_estimate` regardless of
        source (median or predicted). Confs without any score
        (median_score IS NULL) are excluded. Both bounds always apply
        (None bounds match nothing, as in SQL).

        Answered from the `ConfIndex` when one is attached, else (and
        as the shadow reference) by `_confs_under_time_sql`. Lazy: rows
        are fetched in growing chunks as the caller pulls.
        """
        if self._index is None:
            return self._confs_under_time_sql(
                template, system, max_weights, max_time)

        def ids(index: ConfIndex) -> np.ndarray:
            if max_weights is None or max_time is None:
                return np.zeros(0, dtype=np.int64)
            return index.ordered_ids(
                template, system, weights_max=max_weights,
                time_max=max_time)

        def check(row: sqlite3.Row, index: ConfIndex) -> Optional[str]:
            if row['median_score'] != index.score_of(row['conf_id']):
                return (f"median_score {row['median_score']!r} != index "
                        f"{index.score_of(row['conf_id'])!r}")
            if not row['weights'] <= max_weights:
                return f"weights {row['weights']} > {max_weights}"
            if not (row['time_estimate'] is not None
                    and row['time_estimate'] <= max_time):
                return f"time_estimate {row['time_estimate']!r} > {max_time}"
            return None

        def make(row: sqlite3.Row) -> Optional[ConfWithRuns]:
            conf = Configuration.from_dict(row)
            if not conf.model.is_valid():
                return None
            return ConfWithRuns(
                conf_id=row['conf_id'],
                conf=conf,
                median_score=row['median_score'],
                time_estimate=row['time_estimate'],
                total_runs=row['total_runs'],
                system_runs=row['system_runs'],
            )

        return self._index_query(_IndexQuery(
            name='confs_under_time',
            args=dict(system=system, max_weights=max_weights,
                      max_time=max_time, template=template),
            limit=None,
            ids=ids,
            sql=_UNDER_TIME_ROWS,
            params=[system, system],
            check=check,
            make=make,
            key=lambda c: (
                c.conf_id, c.median_score, c.time_estimate,
                c.total_runs, c.system_runs),
            reference=lambda: self._confs_under_time_sql(
                template, system, max_weights, max_time),
        ))

    def _top_confs_for_system_sql(
        self,
        system: str,
        template: Template,
        max_weights: Optional[float] = None,
        max_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Iterator[ConfScore]:
        """SQL implementation of `top_confs_for_system`: the fallback
        when no `ConfIndex` is attached, and the reference the shadow
        check and tests compare the index against."""
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

        # The `conf.id` tie-break spells out the order the index walk
        # already yields (index entries are (median_score, rowid)), so
        # the planner adds no sort; it makes ties deterministic, which
        # the ConfIndex reproduces exactly and the shadow check needs.
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
            ORDER BY median_score ASC, conf.id ASC {limit}
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

    def _confs_under_time_sql(
        self,
        template: Template,
        system: str,
        max_weights: int,
        max_time: float,
    ) -> Iterator[ConfWithRuns]:
        """SQL implementation of `confs_under_time` (fallback and
        shadow/test reference, like `_top_confs_for_system_sql`)."""
        conditions, params = _make_template_conditions(template)
        conditions.append('median_score IS NOT NULL')
        conditions.append('weights <= :max_weights')
        # Ties break by id explicitly: without INDEXED BY a selective
        # filter (a literal spec) makes the planner sort in a temp
        # B-tree, where tie order would otherwise be arbitrary. With the
        # usual `conf_median_score` walk it costs nothing (same plan).
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
            ORDER BY median_score ASC, conf.id ASC
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
        the caller parses, deduplicating by (spec, precision). Both
        consumers do: `loss_tree.load_training_data` in-process, and
        the /training_data CSV a client-side refit downloads.
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
