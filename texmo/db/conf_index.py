"""In-memory copy of `conf` + `conf_time_estimate` for the hot ranked queries.

The search's per-select queries (`DbReader.top_confs_for_system`,
`DbReader.confs_under_time`) and the writer's winner sample
(`DbWriter._top_conf_at`) all have one shape: "confs passing these
filters, best `median_score` first". SQLite answers it by walking
`conf_median_score` and testing each row until LIMIT. The live search
runs at small weights, which are the worst-scoring half of the table,
so every such query walked ~400k rows (1.5-2 s, plus the Python
REGEXP callback for sub-search regexes) and one select issued several.

`ConfIndex` holds the filterable columns as numpy arrays indexed
directly by conf id, plus one time-estimate row per system, plus the
global (median_score, id) order. A query is a handful of vectorized
comparisons and one gather through that order -- milliseconds. It
decides only WHICH conf ids qualify and in WHAT order; the caller then
reads the output columns for those ids from SQL. See
`DbReader._index_query`.

**Consistency.** The index is never updated from what our code
*thinks* it wrote. Triggers inside the DB (`CONF_CHANGE_DDL` in
common.py) log every touched conf id -- and system, for estimate rows
-- into `conf_change`, whoever the writer is. `sync(conn)` reads the
log past its last-seen seq and re-reads exactly those rows from the DB,
overwriting its copy. So the index state is always a function of a
committed snapshot (or, on the writer's connection, of the snapshot
plus its own open transaction), and running `sync` and the follow-up
SQL inside one transaction makes the combined result equal to what the
plain SQL query would have returned on that snapshot.

Whenever catching up incrementally can't be proven complete, `sync`
rebuilds from scratch instead: the log was pruned past our last seq,
its AUTOINCREMENT counter went backwards (a rolled-back transaction we
had read, or a recreated table), its epoch changed, the schema changed
(a dropped/recreated trigger or table -- `PRAGMA schema_version`), or
we're so far behind that re-reading the delta would cost as much.

Mirroring the SQL: the template filter is not reimplemented from
`Template`; it interprets the exact condition strings
`_make_template_conditions` produces, with SQL NULL semantics (NaN
compares false). A condition it doesn't recognize raises
`ConfIndexUnsupported`, and callers fall back to SQL -- a future
template field can't be silently ignored. Ordering is (median_score,
conf id): what walking the `conf_median_score` index yields, ties
included. Numeric columns are float64, which holds both the REAL
columns and the INTEGER ones (all far below 2**53) exactly, so every
comparison sees the same doubles SQLite compares.

**Threading.** Mutable and unsynchronized: each owning thread has its
own instance (see docs/threads.md). Not a shared object.
"""

import logging
import re
import sqlite3
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import latency
from ..configuration import Template
from .common import CONF_CHANGE_TRIGGERS, _make_template_conditions


class ConfIndexUnavailable(Exception):
    """The DB lacks the changelog machinery (tables or triggers), so
    the index can't be kept consistent at all. Callers drop the index
    and stay on SQL."""


class ConfIndexUnsupported(ConfIndexUnavailable):
    """This query uses a condition the index can't evaluate. Callers
    answer just this query from SQL."""


# A sync more than this many changelog rows behind rebuilds instead of
# catching up: re-reading that many rows one id-list at a time costs
# about as much as one sequential full load, and a bulk tool
# (`update_all_scores`, a full estimate refresh) can log millions.
SYNC_REBUILD_ROWS = 300_000

# `DbWriter.prune_conf_change` keeps this many newest rows. Anything a
# consumer could still catch up from incrementally (< SYNC_REBUILD_ROWS
# behind) is kept, with margin.
CONF_CHANGE_KEEP = 500_000

# Default for the production shadow check: every Nth index-backed
# query also runs the SQL it replaced, in the same snapshot, and
# compares. See `DbReader._shadow`.
SHADOW_EVERY = 100

_FETCH_CHUNK = 100_000  # rows per fetchmany on a full load
_IN_CHUNK = 500  # ids per `IN (...)` list on a catch-up re-read
_REGEX_CACHE_SIZE = 64  # spec-pattern match arrays kept (LRU)
# Above this many conf-row changes, re-sort the whole order (~0.1 s)
# instead of patching it one entry at a time.
_ORDER_PATCH_MAX = 2_000

# The columns the template filter can touch. `median_score` and
# `num_layers` are nullable: loaded as (is_null, value) pairs so the
# conversion stays vectorized, then NULL becomes NaN.
_CONF_COLS_FULL = (
    'id, weights, median_score IS NULL, IFNULL(median_score, 0), '
    'precision, lr, decay, cosine, length, batch, steps, '
    'num_layers IS NULL, IFNULL(num_layers, 0), spec')
# The writer only ranks by score under a weight cap.
_CONF_COLS_LEAN = 'id, weights, median_score IS NULL, IFNULL(median_score, 0)'


@dataclass(frozen=True)
class _Marker:
    """The changelog's position, read inside the caller's snapshot."""
    epoch: str
    floor: int
    hw: int  # high-water seq (sqlite_sequence), >= every logged row
    schema_version: int


def _read_marker(conn: sqlite3.Connection) -> _Marker:
    try:
        row = conn.execute(
            'SELECT epoch, floor FROM conf_change_floor WHERE id = 0'
        ).fetchone()
        seq = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'conf_change'"
        ).fetchone()
    except sqlite3.OperationalError as e:
        raise ConfIndexUnavailable(f'no conf_change tables: {e}') from e
    if row is None:
        raise ConfIndexUnavailable('conf_change_floor has no row')
    schema_version = conn.execute('PRAGMA schema_version').fetchone()[0]
    return _Marker(
        epoch=row[0], floor=row[1], hw=seq[0] if seq is not None else 0,
        schema_version=schema_version)


def _chunks(ids: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


_CMP_CONDITION = re.compile(r'(\w+) (>=|<=) :(\w+)')
_PRECISION_IN = re.compile(r'precision IN \((.*)\)')
_QUOTED = re.compile(r'"([^"]*)"')


class ConfIndex(object):
    """Columnar in-memory copy of the indexed conf/estimate columns.

    `full=False` keeps only what `DbWriter._top_conf_at` needs
    (weights, score, estimates): no spec table, roughly half the
    memory and a faster build. Template queries need `full=True`.
    """

    def __init__(
        self, full: bool = True, rebuild_rows: int = SYNC_REBUILD_ROWS,
    ):
        self._full = full
        self._rebuild_rows = rebuild_rows
        self._built = False
        # Full loads so far, and why the last one happened -- for the
        # server log and for tests of the rebuild triggers.
        self.builds = 0
        self.last_build_reason: Optional[str] = None
        self._reset()

    # --- state --------------------------------------------------------------

    def _reset(self) -> None:
        self._epoch: Optional[str] = None
        self._schema_version: Optional[int] = None
        self._last_seq = 0
        self._cap = 0
        self._present = np.zeros(0, dtype=bool)
        self._weights = np.zeros(0)
        self._score = np.zeros(0)
        if self._full:
            self._precision = np.zeros(0, dtype=np.int16)
            self._lr = np.zeros(0)
            self._decay = np.zeros(0)
            self._cosine = np.zeros(0)
            self._length = np.zeros(0)
            self._batch = np.zeros(0)
            self._steps = np.zeros(0)
            self._num_layers = np.zeros(0)
            self._spec_id = np.zeros(0, dtype=np.int32)
        # Spec strings, interned to small ints. Never shrinks between
        # builds (a deleted conf's spec stays); a rebuild resets it.
        self._specs: list[str] = []
        self._spec_ids: dict[str, int] = {}
        self._precisions: dict[str, int] = {}
        # Per-system estimate, one row per system: time_s (NaN = no
        # row) and whether source = 'median'.
        self._systems: dict[str, int] = {}
        self._time = np.zeros((0, 0))
        self._median = np.zeros((0, 0), dtype=bool)
        # Present confs with a score, sorted by (median_score, id),
        # and their scores in the same order (for searchsorted).
        self._order = np.zeros(0, dtype=np.int64)
        self._order_score = np.zeros(0)
        # pattern -> bool array over spec ids, one trailing False slot
        # so spec id -1 (no conf in that slot) never matches.
        self._regex_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def built(self) -> bool:
        return self._built

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def invalidate(self) -> None:
        """Force a full rebuild on the next `sync`. For when the caller
        knows the index may have absorbed state that never committed
        (the writer's transaction rolled back) or saw it disagree with
        SQL (a shadow-check mismatch)."""
        self._built = False

    def _grow(self, needed: int) -> None:
        if needed <= self._cap:
            return
        cap = max(needed, self._cap + self._cap // 4, 1024)

        def grow(a: np.ndarray, fill) -> np.ndarray:
            out = np.full(a.shape[:-1] + (cap,), fill, dtype=a.dtype)
            out[..., :a.shape[-1]] = a
            return out

        self._present = grow(self._present, False)
        self._weights = grow(self._weights, np.nan)
        self._score = grow(self._score, np.nan)
        if self._full:
            self._precision = grow(self._precision, -1)
            self._lr = grow(self._lr, np.nan)
            self._decay = grow(self._decay, np.nan)
            self._cosine = grow(self._cosine, np.nan)
            self._length = grow(self._length, np.nan)
            self._batch = grow(self._batch, np.nan)
            self._steps = grow(self._steps, np.nan)
            self._num_layers = grow(self._num_layers, np.nan)
            self._spec_id = grow(self._spec_id, -1)
        self._time = grow(self._time, np.nan)
        self._median = grow(self._median, False)
        self._cap = cap

    def _system_code(self, system: str) -> int:
        code = self._systems.get(system)
        if code is None:
            code = len(self._systems)
            self._systems[system] = code
            self._time = np.vstack(
                [self._time, np.full((1, self._cap), np.nan)])
            self._median = np.vstack(
                [self._median, np.zeros((1, self._cap), dtype=bool)])
        return code

    def _spec_code(self, spec: str) -> int:
        code = self._spec_ids.get(spec)
        if code is None:
            code = len(self._specs)
            self._specs.append(spec)
            self._spec_ids[spec] = code
        return code

    def _precision_code(self, precision: str) -> int:
        code = self._precisions.get(precision)
        if code is None:
            code = len(self._precisions)
            self._precisions[precision] = code
        return code

    def memory_bytes(self) -> int:
        """Approximate resident size: the arrays plus the spec table."""
        arrays = [
            self._present, self._weights, self._score, self._time,
            self._median, self._order, self._order_score]
        if self._full:
            arrays += [
                self._precision, self._lr, self._decay, self._cosine,
                self._length, self._batch, self._steps, self._num_layers,
                self._spec_id]
        total = sum(a.nbytes for a in arrays)
        total += sum(a.nbytes for a in self._regex_cache.values())
        total += sys.getsizeof(self._specs) + sys.getsizeof(self._spec_ids)
        total += sum(sys.getsizeof(s) for s in self._specs)
        return total

    # --- loading ------------------------------------------------------------

    def _apply_conf_rows(self, rows: list[tuple]) -> None:
        """Overwrite the slots of `rows` (tuples in `_CONF_COLS_*`
        order) and mark them present."""
        if not rows:
            return
        cols = list(zip(*rows))
        ids = np.array(cols[0], dtype=np.int64)
        self._grow(int(ids.max()) + 1)
        self._present[ids] = True
        self._weights[ids] = np.array(cols[1], dtype=np.float64)
        score = np.array(cols[3], dtype=np.float64)
        score[np.array(cols[2], dtype=bool)] = np.nan
        self._score[ids] = score
        if not self._full:
            return
        self._precision[ids] = np.array(
            [self._precision_code(p) for p in cols[4]], dtype=np.int16)
        self._lr[ids] = np.array(cols[5], dtype=np.float64)
        self._decay[ids] = np.array(cols[6], dtype=np.float64)
        self._cosine[ids] = np.array(cols[7], dtype=np.float64)
        self._length[ids] = np.array(cols[8], dtype=np.float64)
        self._batch[ids] = np.array(cols[9], dtype=np.float64)
        self._steps[ids] = np.array(cols[10], dtype=np.float64)
        num_layers = np.array(cols[12], dtype=np.float64)
        num_layers[np.array(cols[11], dtype=bool)] = np.nan
        self._num_layers[ids] = num_layers
        self._spec_id[ids] = np.array(
            [self._spec_code(s) for s in cols[13]], dtype=np.int32)

    def _apply_estimate_rows(
        self, codes: np.ndarray, ids: np.ndarray,
        times: np.ndarray, median: np.ndarray,
    ) -> None:
        if len(ids) == 0:
            return
        self._grow(int(ids.max()) + 1)
        self._time[codes, ids] = times
        self._median[codes, ids] = median

    def _build(
        self, conn: sqlite3.Connection, marker: _Marker, reason: str,
    ) -> None:
        t0 = time.perf_counter()
        self._built = False
        triggers = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        missing = set(CONF_CHANGE_TRIGGERS) - triggers
        if missing:
            raise ConfIndexUnavailable(
                f'conf_change triggers missing: {sorted(missing)}')
        self._reset()

        cur = conn.cursor()
        cur.row_factory = None
        cols = _CONF_COLS_FULL if self._full else _CONF_COLS_LEAN
        (max_id,) = cur.execute(
            'SELECT IFNULL(MAX(id), 0) FROM conf').fetchone()
        self._grow(max_id + 1)
        cur.execute(f'SELECT {cols} FROM conf')
        num_confs = 0
        while rows := cur.fetchmany(_FETCH_CHUNK):
            self._apply_conf_rows(rows)
            num_confs += len(rows)

        # Map system -> row code inside SQL, so 6M+ estimate rows come
        # back as plain numbers rather than one Python string each.
        systems = [row[0] for row in cur.execute(
            'SELECT DISTINCT system FROM conf_time_estimate')]
        num_estimates = 0
        if systems:
            codes = [self._system_code(s) for s in systems]
            case = ' '.join('WHEN ? THEN ?' for _ in systems)
            params = [x for pair in zip(systems, codes) for x in pair]
            cur.execute(
                f"SELECT conf_id, CASE system {case} END, time_s, "
                f"source = 'median' FROM conf_time_estimate", params)
            while rows := cur.fetchmany(_FETCH_CHUNK):
                c = list(zip(*rows))
                self._apply_estimate_rows(
                    np.array(c[1], dtype=np.int64),
                    np.array(c[0], dtype=np.int64),
                    np.array(c[2], dtype=np.float64),
                    np.array(c[3], dtype=bool))
                num_estimates += len(rows)

        self._rebuild_order()
        self._epoch = marker.epoch
        self._schema_version = marker.schema_version
        self._last_seq = marker.hw
        self._built = True
        self.builds += 1
        self.last_build_reason = reason
        logging.info(
            f'ConfIndex built ({reason}): {num_confs} confs, '
            f'{num_estimates} estimates over {len(self._systems)} systems, '
            f'{self.memory_bytes() / 2**20:.0f} MB, '
            f'{time.perf_counter() - t0:.1f} s')

    # --- sync ---------------------------------------------------------------

    def _rebuild_reason(self, marker: _Marker) -> Optional[str]:
        if not self._built:
            return 'initial' if self.builds == 0 else 'invalidated'
        if marker.epoch != self._epoch:
            return 'changelog epoch changed'
        if marker.schema_version != self._schema_version:
            return 'schema changed'
        if marker.hw < self._last_seq:
            return (f'changelog went backwards ({self._last_seq} -> '
                    f'{marker.hw})')
        if self._last_seq < marker.floor:
            return (f'changelog pruned past us (at {self._last_seq}, '
                    f'floor {marker.floor})')
        if marker.hw - self._last_seq > self._rebuild_rows:
            return f'{marker.hw - self._last_seq} changes behind'
        return None

    def sync(self, conn: sqlite3.Connection) -> None:
        """Bring the index up to the snapshot `conn` is reading.

        Must run inside a transaction on `conn` (a reader's explicit
        BEGIN, or the writer's open write transaction), and the caller
        must do its follow-up SQL in that same transaction: the index
        then describes exactly the rows those statements see. Builds
        from scratch the first time and whenever an incremental catch-up
        can't be shown complete (see the module docstring).

        Any failure part-way leaves the index invalidated, never half
        updated. Raises `ConfIndexUnavailable` when the DB has no
        changelog machinery.
        """
        assert conn.in_transaction, 'ConfIndex.sync needs an open transaction'
        try:
            marker = _read_marker(conn)
            reason = self._rebuild_reason(marker)
            if reason is not None:
                with latency.timer('ConfIndex.build'):
                    self._build(conn, marker, reason)
            elif marker.hw != self._last_seq:
                with latency.timer('ConfIndex.catch_up'):
                    self._catch_up(conn, marker)
        except BaseException:
            self._built = False
            raise

    def _catch_up(self, conn: sqlite3.Connection, marker: _Marker) -> None:
        cur = conn.cursor()
        cur.row_factory = None
        rows = cur.execute(
            'SELECT conf_id, system FROM conf_change '
            'WHERE seq > ? AND seq <= ?',
            (self._last_seq, marker.hw)).fetchall()
        conf_ids: set[int] = set()
        pairs: dict[str, set[int]] = {}
        for conf_id, system in rows:
            if system is None:
                conf_ids.add(conf_id)
            else:
                pairs.setdefault(system, set()).add(conf_id)
        if conf_ids:
            self._reread_confs(cur, sorted(conf_ids))
        for system, ids in pairs.items():
            ids -= conf_ids  # already re-read with all their estimates
            if ids:
                self._reread_estimates(cur, system, sorted(ids))
        self._patch_order(conf_ids)
        self._last_seq = marker.hw

    def _reread_confs(self, cur: sqlite3.Cursor, ids: list[int]) -> None:
        """Overwrite these confs, and all their estimates, from the DB.
        Ids no longer in `conf` end up absent."""
        cols = _CONF_COLS_FULL if self._full else _CONF_COLS_LEAN
        for chunk in _chunks(ids, _IN_CHUNK):
            marks = ','.join('?' * len(chunk))
            self._grow(chunk[-1] + 1)
            idx = np.array(chunk, dtype=np.int64)
            self._present[idx] = False
            self._apply_conf_rows(cur.execute(
                f'SELECT {cols} FROM conf WHERE id IN ({marks})',
                chunk).fetchall())
            self._time[:, idx] = np.nan
            self._median[:, idx] = False
            rows = cur.execute(
                f"SELECT conf_id, system, time_s, source = 'median' "
                f"FROM conf_time_estimate WHERE conf_id IN ({marks})",
                chunk).fetchall()
            if rows:
                c = list(zip(*rows))
                self._apply_estimate_rows(
                    np.array([self._system_code(s) for s in c[1]],
                             dtype=np.int64),
                    np.array(c[0], dtype=np.int64),
                    np.array(c[2], dtype=np.float64),
                    np.array(c[3], dtype=bool))

    def _reread_estimates(
        self, cur: sqlite3.Cursor, system: str, ids: list[int],
    ) -> None:
        """Overwrite the (conf, system) estimates for these ids."""
        code = self._system_code(system)
        for chunk in _chunks(ids, _IN_CHUNK):
            marks = ','.join('?' * len(chunk))
            self._grow(chunk[-1] + 1)
            idx = np.array(chunk, dtype=np.int64)
            self._time[code, idx] = np.nan
            self._median[code, idx] = False
            rows = cur.execute(
                f"SELECT conf_id, time_s, source = 'median' "
                f"FROM conf_time_estimate "
                f"WHERE system = ? AND conf_id IN ({marks})",
                [system] + chunk).fetchall()
            if rows:
                c = list(zip(*rows))
                self._apply_estimate_rows(
                    np.full(len(rows), code, dtype=np.int64),
                    np.array(c[0], dtype=np.int64),
                    np.array(c[1], dtype=np.float64),
                    np.array(c[2], dtype=bool))

    # --- (median_score, id) order -------------------------------------------

    def _rebuild_order(self) -> None:
        live = np.flatnonzero(self._present & ~np.isnan(self._score))
        score = self._score[live]
        # `live` is ascending, so a stable sort on score alone orders
        # ties by id -- the `conf_median_score` index walk order.
        perm = np.argsort(score, kind='stable')
        self._order = live[perm]
        self._order_score = score[perm]

    def _patch_order(self, changed: set[int]) -> None:
        """Move the changed confs to their new place in the order (or
        out of it, if deleted or unscored)."""
        if not changed:
            return
        if len(changed) > _ORDER_PATCH_MAX:
            self._rebuild_order()
            return
        ids = np.fromiter(changed, dtype=np.int64, count=len(changed))
        mark = np.zeros(self._cap, dtype=bool)
        mark[ids] = True
        keep = ~mark[self._order]
        order = self._order[keep]
        order_score = self._order_score[keep]
        ids = ids[self._present[ids] & ~np.isnan(self._score[ids])]
        if len(ids):
            ids = np.sort(ids)
            score = self._score[ids]
            perm = np.argsort(score, kind='stable')
            ids, score = ids[perm], score[perm]
            lo = np.searchsorted(order_score, score, side='left')
            hi = np.searchsorted(order_score, score, side='right')
            pos = np.empty(len(ids), dtype=np.int64)
            for i in range(len(ids)):
                pos[i] = lo[i] + np.searchsorted(
                    order[lo[i]:hi[i]], ids[i])
            order = np.insert(order, pos, ids)
            order_score = np.insert(order_score, pos, score)
        self._order = order
        self._order_score = order_score

    # --- queries ------------------------------------------------------------

    def _regex_mask(self, pattern: str) -> np.ndarray:
        """Match array over spec ids for `spec REGEXP pattern`, i.e.
        `re.fullmatch(pattern, spec)` exactly as `_regexp` does it (the
        pattern string, default flags). Cached per pattern and extended
        to specs interned since."""
        n = len(self._specs)
        cached = self._regex_cache.pop(pattern, None)
        if cached is None or len(cached) - 1 < n:
            done = 0 if cached is None else len(cached) - 1
            out = np.zeros(n + 1, dtype=bool)
            if done:
                out[:done] = cached[:done]
            fullmatch = re.compile(pattern).fullmatch
            out[done:n] = [
                fullmatch(s) is not None for s in self._specs[done:n]]
            cached = out
        self._regex_cache[pattern] = cached
        while len(self._regex_cache) > _REGEX_CACHE_SIZE:
            self._regex_cache.popitem(last=False)
        return cached

    def _template_mask(self, template: Template) -> Optional[np.ndarray]:
        """Evaluate `_make_template_conditions(template)`, condition by
        condition, over every slot. None when there are no conditions."""
        if not self._full:
            raise ConfIndexUnsupported('lean ConfIndex has no template columns')
        conditions, params = _make_template_conditions(template)
        mask: Optional[np.ndarray] = None
        for cond in conditions:
            m = self._condition_mask(cond, params)
            mask = m if mask is None else np.logical_and(mask, m, out=mask)
        return mask

    def _condition_mask(self, cond: str, params: dict) -> np.ndarray:
        if cond == 'spec = :spec':
            code = self._spec_ids.get(params['spec'])
            if code is None:
                return np.zeros(self._cap, dtype=bool)
            return self._spec_id == code
        if cond == 'spec REGEXP :regex':
            return self._regex_mask(params['regex'])[self._spec_id]
        if (m := _CMP_CONDITION.fullmatch(cond)) is not None:
            column, op, param = m.groups()
            arrays = {
                'lr': self._lr, 'length': self._length,
                'batch': self._batch, 'steps': self._steps,
                'weights': self._weights, 'num_layers': self._num_layers,
            }
            if column not in arrays:
                raise ConfIndexUnsupported(f'unknown condition {cond!r}')
            a, v = arrays[column], params[param]
            # NaN (SQL NULL) compares false either way, as in SQL.
            return a >= v if op == '>=' else a <= v
        if (m := _PRECISION_IN.fullmatch(cond)) is not None:
            mask = np.zeros(self._cap, dtype=bool)
            for p in _QUOTED.findall(m.group(1)):
                code = self._precisions.get(p)
                if code is not None:
                    mask |= self._precision == code
            return mask
        decay = _DECAY_CLAUSES.get(cond)
        if decay is not None:
            return decay(self)
        raise ConfIndexUnsupported(f'unknown condition {cond!r}')

    def ordered_ids(
        self,
        template: Optional[Template],
        system: str,
        weights_max: Optional[float] = None,
        time_max: Optional[float] = None,
        median_time_only: bool = False,
    ) -> np.ndarray:
        """Conf ids in (median_score, id) order that have a score and
        pass every given condition:

        - `template`: `_make_template_conditions(template)`;
        - `weights_max`: `weights <= weights_max`;
        - `time_max`: an estimate on `system` with `time_s <= time_max`
          (only a `source = 'median'` one if `median_time_only`).

        A None argument means "no such condition" -- callers translate
        their SQL's own None/truthiness rules before calling.
        """
        assert self._built, 'ConfIndex.ordered_ids before sync'
        mask = None if template is None else self._template_mask(template)

        def conjoin(m: np.ndarray) -> None:
            nonlocal mask
            mask = m if mask is None else np.logical_and(mask, m, out=mask)

        if weights_max is not None:
            conjoin(self._weights <= weights_max)
        if time_max is not None:
            code = self._systems.get(system)
            if code is None:
                return np.zeros(0, dtype=np.int64)
            m = self._time[code] <= time_max
            if median_time_only:
                m &= self._median[code]
            conjoin(m)
        if mask is None:
            return self._order.copy()
        return self._order[mask[self._order]]

    # --- per-conf accessors (drift checks) ----------------------------------

    def score_of(self, conf_id: int) -> Optional[float]:
        if conf_id >= self._cap or not self._present[conf_id]:
            return None
        s = self._score[conf_id]
        return None if np.isnan(s) else float(s)

    def weights_of(self, conf_id: int) -> Optional[float]:
        if conf_id >= self._cap or not self._present[conf_id]:
            return None
        return float(self._weights[conf_id])

    # --- test / probe support -----------------------------------------------

    def state(self) -> dict[int, tuple]:
        """Canonical, order-free dump of everything the index holds per
        present conf. Two indexes describe the same snapshot iff their
        `state()`s are equal (plus `order_ok`). Slow; tests only."""
        out = {}
        prec = {v: k for k, v in self._precisions.items()}
        for cid in np.flatnonzero(self._present).tolist():
            est = {}
            for system, code in self._systems.items():
                t = self._time[code, cid]
                if not np.isnan(t):
                    est[system] = (float(t), bool(self._median[code, cid]))
            row: tuple = (
                float(self._weights[cid]), self.score_of(cid), est)
            if self._full:
                nl = self._num_layers[cid]
                row += (
                    prec[int(self._precision[cid])], float(self._lr[cid]),
                    float(self._decay[cid]), float(self._cosine[cid]),
                    float(self._length[cid]), float(self._batch[cid]),
                    float(self._steps[cid]),
                    None if np.isnan(nl) else float(nl),
                    self._specs[int(self._spec_id[cid])])
            out[cid] = row
        return out

    def order_ok(self) -> bool:
        """The maintained order equals a from-scratch sort."""
        order, order_score = self._order, self._order_score
        self._rebuild_order()
        ok = (np.array_equal(order, self._order)
              and np.array_equal(order_score, self._order_score))
        self._order, self._order_score = order, order_score
        return ok


def _decay_masks() -> dict[str, Callable[[ConfIndex], np.ndarray]]:
    """Vectorized twin of every clause `_decay_type_clause` can emit,
    keyed by the exact SQL text -- a new clause there has no entry here
    and fails loudly (`ConfIndexUnavailable`) instead of being skipped."""
    return {
        '0': lambda ix: np.zeros(ix._cap, dtype=bool),
        '(cosine = 0 AND decay = 1)':
            lambda ix: (ix._cosine == 0) & (ix._decay == 1),
        '(cosine = 0 AND decay < 1)':
            lambda ix: (ix._cosine == 0) & (ix._decay < 1),
        '(cosine = 1)': lambda ix: ix._cosine == 1,
        '(cosine = 0)': lambda ix: ix._cosine == 0,
        '(decay = 1)': lambda ix: ix._decay == 1,
        '(cosine = 1 OR decay < 1)':
            lambda ix: (ix._cosine == 1) | (ix._decay < 1),
    }


_DECAY_CLAUSES = _decay_masks()
