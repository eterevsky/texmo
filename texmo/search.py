import copy
import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from math import log2
from queue import Queue
from typing import Iterable, Optional

import numpy as np

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration import (
    Bounds,
    Configuration,
    Template,
    TemplateEntry,
    TemplateSet,
    conf_neighbors,
)
from .db import ConfScore, ConfWithRuns, DbReader
from .predict.loss_rnn import LossModelHolder
from .predict.timing import TrainTimingModel
from .report import format_top_conf_row


@dataclass
class SearchResult:
    """A configuration paired with the system it's for and the strategy
    label that picked it.

    `covered` / `total` are the coverage-walk progress (top confs covered
    on this system out of the total the template admits); set only for
    the `coverage_walk` strategy, None otherwise.
    """
    conf: Configuration
    strategy: str
    system: str
    covered: int | None = None
    total: int | None = None

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "conf": self.conf.to_dict(),
            "strategy": self.strategy,
            "covered": self.covered,
            "total": self.total,
        }


# --- requests_queue message types ------------------------------------------


@dataclass
class Select:
    """Ask `SearchThread` to pick a candidate for `system` and push the
    `SearchResult` onto `confs_by_system[system]`."""
    system: str
    # Stamped at construction; lets the search thread recognize
    # requests whose client has certainly timed out already.
    enqueued_at: float = field(default_factory=time.monotonic)


# Drop Select messages older than this instead of producing a conf
# for them: the handler's client read-timeout (client._HTTP_TIMEOUT,
# 120 s) has expired, so the conf would be delivered to a closed
# connection. Without this, a transient stall leaves a backlog that
# drains at select-conf speed while every result goes to a dead
# handler -- and past N_clients * select_time > client timeout the
# backlog never drains at all (a livelock: the server works at 100%
# producing confs nobody receives). The stale request still gets a
# None response -- exactly one response per Select, or its waiting
# handler thread would block forever.
_SELECT_STALE_S = 120.0


@dataclass
class SetTemplate:
    """Reconfigure `Search` between two `select_conf` calls. The group
    moves together so a partial update never lands on the search loop.

    `templates` is the weighted sub-search set derived from the same
    base `template`; it always has at least one entry."""
    template: Template
    init_conf: Configuration
    train_time: tuple[float, float]
    templates: TemplateSet


@dataclass
class Stop:
    """Stop the search thread."""


SearchMessage = Select | SetTemplate | Stop

# Input encodings retired from SCHEDULING. A conf whose input spec
# starts with one of these is never handed to a client again, whether
# as a re-run of a conf already in the DB or as a freshly generated
# mutation (an architecture mutation of a retired conf is itself a
# retired conf). Their existing runs stay in the DB and they remain
# valid mutation SOURCES: neighbor lists are still generated from
# them, and their non-retired neighbors are still schedulable -- an
# outgoing bridge edge is how the DB population migrates off a
# retired encoding.
#
# Currently empty. tokens.64.shift was retired here 2026-07-28 in
# favor of tokens.64.hexbpe, then re-enabled a day later when the
# search showed shift BEATING hexbpe-64 by ~1-2% loss over most of
# the frontier above ~400 weights (per-letter tokens seem to matter);
# the mechanism stays for the next retirement.
RETIRED_INPUTS: tuple[str, ...] = ()


def _is_retired(spec: str) -> bool:
    """True when the input part of `spec` (everything before `|`) is a
    retired encoding. Prefix match, so `.emb.N` and `.oh` spellings are
    both caught."""
    return spec.split('|', 1)[0].startswith(RETIRED_INPUTS)


def _is_retired_conf(conf: Configuration) -> bool:
    return _is_retired(str(conf.model))


# Per-select strategy split. One weighted draw per `select_conf`
# decides which main strategy fires first; if it returns None (e.g.
# `predicted_*` without a fitted loss model), we fall back to the
# neighbor walk, then to the default conf. Coverage walk runs
# separately at the top of `select_conf`. Weights must sum to 1.0.
_STRATEGY_PROBS: list[tuple[str, float]] = [
    ('predicted_2nd_neighbor', 0.1),
    ('predicted_3rd_neighbor', 0.1),
    ('time_budget', 0.2),
    ('neighbor', 0.6),
]
assert abs(sum(w for _, w in _STRATEGY_PROBS) - 1.0) < 1e-9, (
    f"_STRATEGY_PROBS must sum to 1.0, "
    f"got {sum(w for _, w in _STRATEGY_PROBS)}"
)

# Probability of running the cross-system coverage walk per select.
# When the previous walk on this system found enough uncovered top
# confs (see `_COVERAGE_STICKY_THRESHOLD`), the next call fires
# unconditionally via `Search._coverage_flag`.
_COVERAGE_PROB = 0.1

# Warmup ladder: (max_steps, max_batch, max_length) rungs for systems
# started with `--warmup-systems` (a new machine, or a new backend on
# a known machine). Such a system has no timing model, so nothing
# bounds how long a conf will train: handed a fleet-favourite
# recurrent conf at S8192 it can sit on it for hours, and the search
# learns nothing until it finishes. A rung admits only confs the
# fleet already runs at or below its caps on the three knobs that
# multiply into total work, growing ONE knob at a time (steps x4,
# batch x4, length x2) so the product rises ~2-4x per rung and the
# timing model is fit (first fit at 100 runs) well before the
# expensive rungs open.
_WARMUP_LADDER: list[tuple[int, int, int]] = [
    (2, 1, 64),
    (8, 1, 64),
    (8, 4, 64),
    (8, 4, 128),
    (32, 4, 128),
    (32, 16, 128),
    (32, 16, 256),
    (128, 16, 256),
    (128, 64, 256),
    (128, 64, 512),
    (512, 64, 512),
    (512, 256, 512),
    (512, 256, 1024),
    (2048, 256, 1024),
    (8192, 256, 1024),
]

# Picks per warmup rung before the next one opens. A rung also
# advances early once every top conf under it is covered on the
# system, so a small DB doesn't stall the ladder.
_WARMUP_SELECTS_PER_RUNG = 10

# Layer-count diversification: per select, sample a cap on num_layers
# and run the whole select (seed queries, neighbor/BFS expansion via
# conf_neighbors' match_model filter, and hence the final conf) under
# the base template intersected with that cap. Pulls part of the budget
# toward shallow models so the search doesn't fixate on mutations of one
# long layer chain. None = unrestricted. pick_me, the coverage walk and
# the default fallback stay on the base template (explicit picks and
# cross-system coverage shouldn't be dodged by a sampled cap). Weights
# must sum to 1.0.
_LAYER_CAP_PROBS: list[tuple[Optional[int], float]] = [
    (None, 0.6),
    (1, 0.1),
    (2, 0.1),
    (3, 0.1),
    (4, 0.1),
]
assert abs(sum(w for _, w in _LAYER_CAP_PROBS) - 1.0) < 1e-9


@functools.lru_cache(maxsize=None)
def _layer_cap_probs(
    min_layers: int,
) -> tuple[tuple[Optional[int], float], ...]:
    """`_LAYER_CAP_PROBS` with every cap below `min_layers` dropped and
    the surviving weights renormalized. `None` (unrestricted) always
    survives.

    A sub-search whose smallest conf already has N layers can never
    satisfy a cap below N -- a transformer block is ~9 layers, so 40%
    of its selects would query an empty intersection, pay for the
    (regex-filtered) queries and fall through. Cached: this is called
    on every select, keyed by the entry's cached `min_layers`.
    """
    kept = [
        (cap, w) for cap, w in _LAYER_CAP_PROBS
        if cap is None or cap >= min_layers
    ]
    total = sum(w for _, w in kept)
    return tuple((cap, w / total) for cap, w in kept)


def _cap_bound(b: Bounds, cap: int, floor: int) -> Bounds:
    """`b` with its max lowered to `cap`, never below its own min."""
    return Bounds((b.min, max(b.min, min(b.max, cap))), floor)


def _warmup_template(
    template: Template, max_steps: int, max_batch: int, max_length: int,
) -> Template:
    """`template` intersected with one warmup rung's caps.

    Used as a FILTER on existing confs, not as a rewrite: a shallow
    copy (sub-objects shared, never mutated), so `top_confs_global`
    returns only confs the fleet has already run at a shape this rung
    admits.
    """
    capped = copy.copy(template)
    capped.steps = _cap_bound(template.steps, max_steps, 2)
    capped.batch = _cap_bound(template.batch, max_batch, 1)
    capped.length = _cap_bound(template.length, max_length, 1)
    return capped


def _layer_capped_template(
    template: Template, cap: Optional[int],
) -> tuple[Template, Optional[int]]:
    """Intersect `template` with a num_layers cap.

    Returns (template, cap): a shallow copy with `num_layers` tightened
    to the cap (sub-objects are shared, never mutated), or the base
    template itself (cap None) when the draw is unrestricted -- or when
    the cap is below the base floor / already implied by the base bound.
    """
    if cap is None:
        return template, None
    base = template.num_layers
    if base.min > cap or base.max <= cap:
        return template, None
    capped = copy.copy(template)
    capped.num_layers = Bounds((base.min, cap), 0)
    return capped, cap

# Minimum total runs a pick_me conf needs before it stops being
# picked preferentially. Two runs give a median that's robust to a
# single-run outlier, which is what `top_confs_global` already uses.
PICK_ME_MIN_RUNS = 2

# Sticky-flag threshold: keep firing the coverage walk only when at
# least this many uncovered top confs remain on the system. Set high
# enough that successive selects within the prefetch gap window
# don't pile up on the same few confs before any run finishes.
_COVERAGE_STICKY_THRESHOLD = 5

# Bounds on the predicted-best BFS. Unbounded, the visited set grows
# ~k^depth with the per-conf neighbor count k -- the depth-3 ball from
# a typical seed is ~200k confs -- and each visited conf costs a
# conf_neighbors call, a timing predict, and a get_conf_id query
# (~170us per conf measured end to end, all of it pure Python). A fat
# seed at depth 3 was observed, back when that path cost ~25ms per
# conf, to hold the single SearchThread for 7+ minutes, timing out
# every client (they retry, the queue grows, and each queued select has
# its own chance of another depth-3: the stall self-perpetuates).
# _BFS_FRONTIER_CAP random-samples each BFS level (keeps the depth-N
# reachability diverse instead of truncating by enumeration order);
# _BFS_TIME_BUDGET is the hard wall-clock stop for the expansion loop.
# A clipped BFS still yields a valid, slightly less exhaustive
# candidate set. _BFS_VISITED_CAP is the binding bound at depth 3, at
# ~1.7s per select.
_BFS_FRONTIER_CAP = 256
_BFS_VISITED_CAP = 10_000
_BFS_TIME_BUDGET_S = 5.0


def top_confs_report(
    confs: list[ConfScore], max_weights: int, max_time: Optional[float], system: Optional[str]
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    lines = [f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}:"]
    for c in confs:
        lines.append(format_top_conf_row(c))
    return "\n".join(lines)


def _predicted_best_report(
    seed: ConfScore,
    candidate_data: list[tuple[float, int, Configuration]],
    system: str,
    max_weights: int,
    max_time: float,
    depth: int,
) -> str:
    """Log the seed (with its known median loss for comparison) and
    the top 9 candidates — we only look at the top 9 during the
    run-limit walk."""
    seed_score = f'{seed.median_score:.4f} ({seed.num_runs})'
    lines = [
        f"Predicted-best confs W <= {itoa3(max_weights)} "
        f"T <= {ttoa3(max_time)}, depth {depth} ({system}):",
        f'    {seed_score:<12}  {seed.conf.aligned_str()}  [seed]',
    ]
    for compound, total_runs, c in candidate_data[:9]:
        score = f'{2.0 ** compound:.4f} ({total_runs})'
        lines.append(f'    {score:<12}  {c.aligned_str()}')
    return "\n".join(lines)


def time_budget_report(
    confs: list[ConfWithRuns],
    max_weights: int,
    max_time: float,
    system: str,
) -> str:
    lines = [
        f"Time-budget confs W ≤ {itoa3(max_weights)} "
        f"T ≤ {ttoa3(max_time)} ({system}):"
    ]
    for c in confs:
        t = f'{c.time_estimate:7.3f}s'
        score_runs = (
            f'{c.median_score:.4f} '
            f'({c.total_runs}|{c.system_runs})'
        )
        lines.append(f'    {score_runs:<12} {t}  {c.conf.aligned_str()}')
    return "\n".join(lines)



def _generate_limits():
    """Generate a sequence of expected numbers of completed runs per position
    in the top confs.

    Pairs [x0, x1] mean at least x0 runs for the configuration
    itself, at least x1 runs for the direct neighbors.

    Sequence:

    =0 [1, 0]
    =0 [2, 0]
    =0 [2, 1]
    =0 [2, 1]  <3 [1, 0]
    =0 [3, 1]  <3 [1, 0]
    =0 [3, 2]  <3 [1, 0]
    =0 [3, 2]  <3 [2, 0]
    =0 [3, 2]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 2]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [2, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 1]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [1, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 0]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 1]
    =0 [4, 3]  <3 [3, 2]  <9 [2, 1]  <27 [1, 0]
    """
    seq = [[1, 0]]
    while seq[0][0] <= 7:
        yield seq

        incremented = False
        for row in seq:
            if row[0] > row[1] + 1:
                row[1] += 1
                incremented = True
                break

        if incremented:
            continue

        for i in range(len(seq) - 1):
            if seq[i][0] > seq[i+1][0] + 1:
                seq[i+1][0] += 1
                incremented = True
                break

        if incremented:
            continue

        if seq[-1][0] > 1:
            seq.append([1, 0])
            continue

        seq[0][0] += 1


def _run_limit_sequences():
    """Sequences of expected total-run counts for the time-budget search.

        iter 1: [1]
        iter 2: [2, 1, 1]
        iter 3: [3, 2, 2, 1, 1, 1, 1, 1, 1]
        iter 4: [4, 3, 3, 2×6, 1×18]
        ...

    Iter N has length 3^(N-1). Recurrence: iter N = iter N-1 with each
    value incremented, followed by len(iter N-1) * 2 ones.
    """
    seq = [1]
    for _ in range(7):
        yield seq
        seq = [x + 1 for x in seq] + [1] * (2 * len(seq))


class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(
        self,
        reader: DbReader,
        template: Template,
        init_conf: Configuration,
        train_time: tuple[float, float],
        timing_model: TrainTimingModel,
        loss_model: LossModelHolder,
        warmup_systems: Optional[Iterable[str]] = None,
        templates: Optional[TemplateSet] = None,
    ):
        assert isinstance(reader, DbReader)
        # `_db` is the read handle used by every strategy below. Search
        # is read-only; `add_run` now goes straight to the writer queue
        # from `SearchThread` without a Search detour.
        self._db = reader

        assert isinstance(template, Template)
        # The BASE template: the whole region the server is searching.
        # Used by the template-blind steps (pick_me, warmup ladder, the
        # steps floor in `_cap_steps`). Everything downstream of the
        # sub-template draw uses the drawn entry's template instead.
        self.template = template

        assert isinstance(init_conf, Configuration)
        # The base template's default conf, and the one a stand-in
        # single-entry set is built from. The default FALLBACK uses the
        # drawn entry's own `default` instead, which is what lets a
        # brand-new sub-space bootstrap.
        self.init_conf = init_conf

        # Confs served per entry, in memory since server start (no DB
        # schema change). The web UI shows each entry's count and its
        # share of the total -- the REALIZED share, next to the nominal
        # one. The raw count is the sample size behind that percentage:
        # it takes a few hundred selects before the two are comparable.
        self.entry_selects: dict[str, int] = {}

        # Weighted sub-searches. Absent one, the whole template is a
        # single entry and every select behaves exactly as it did
        # before template sets existed.
        if templates is None:
            templates = TemplateSet.single(
                template, spec=template.spec_pattern, default=init_conf)
        self.set_templates(templates)

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self.train_time = train_time

        # Shared references with the Model thread. `timing_model` is
        # mutated in place (per-(system, precision) dict inserts are
        # atomic in CPython); `loss_model` is a holder whose underlying
        # `LossModel` gets swapped atomically on refit.
        self.timing_model = timing_model
        self.loss_model = loss_model

        # Systems climbing the `_WARMUP_LADDER` before ordinary
        # selection (`--warmup-systems`), with their current rung and
        # picks spent on it. In-memory only: a restart replays the
        # ladder, which is cheap by construction and re-verifies the
        # machine.
        self.warmup_systems: set[str] = set(warmup_systems or ())
        self._warmup_rung: dict[str, int] = {}
        self._warmup_selects: dict[str, int] = {}
        # (rung, top confs) cache: `top_confs_global` is one of the
        # priciest queries we run (~5 s on the live DB) and warmup
        # fires on every select for its system, so the candidate list
        # is fetched once per rung rather than once per pick.
        self._warmup_top: dict[str, tuple[int, list]] = {}

        # Sticky flag set by `_select_uncovered_top` when it finds
        # `_COVERAGE_STICKY_THRESHOLD`+ uncovered top confs — keeps the
        # coverage walk firing at 100% on the next select for that
        # (system, entry) until the gap closes; reset by the same call
        # when fewer are found. Keyed per entry because a conf can be
        # Pareto-optimal for one sub-template and invisible under
        # another: covering it under `main` says nothing about the
        # frontier a narrow entry sees.
        self._coverage_flag: dict[tuple[str, str], bool] = {}

        # Last (covered, total) top-conf counts from
        # `_select_uncovered_top`, per (system, entry). Attached to
        # coverage-walk SearchResults so the client can log progress.
        self._coverage_stats: dict[tuple[str, str], tuple[int, int]] = {}

    def set_templates(self, templates: TemplateSet) -> None:
        """Install a template set and make sure every entry has its
        select counter, so concurrent readers (the web UI) never see
        the counter dict grow under them."""
        assert isinstance(templates, TemplateSet)
        self.templates = templates
        for entry in templates:
            self.entry_selects.setdefault(entry.name, 0)

    def _select_time(self) -> float:
        tmin, tmax = self.train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t, system: str, template: Template):
        with latency.timer("Search._select_max_weights"):
            try:
                top_conf = next(
                    self._db.top_confs_for_system(
                        max_time=t, system=system, limit=1, template=template)
                )
            except StopIteration:
                return template.max_weights.min
            maxw = 8 * top_conf.conf.model.num_weights
            if template.max_weights.min >= maxw:
                return template.max_weights.min
            if template.max_weights.max <= maxw:
                maxw = template.max_weights.max
            l = random.uniform(
                log2(template.max_weights.min), log2(maxw))
            return round(2**l)

    def _predicted_time(
        self, conf: Configuration, system: str
    ) -> Optional[float]:
        """Predicted train_time for `conf` on `system`, or None if no
        prediction is available.

        Prefer the persisted estimate (median if any runs exist, else
        the predicted-from-fit value the Model thread maintains); fall
        back to a live timing-model prediction for confs we've never
        seen on this system.
        """
        conf_id = self._db.get_conf_id(conf)
        if conf_id is not None:
            est = self._db.get_time_estimate(conf_id, system)
            if est is not None:
                return est[0]
        return self.timing_model.predict(system, conf)

    def _timing_ready(self, system: str) -> bool:
        """True once the timing model has been fit for `system` (any
        precision).

        Until then there are no step-time predictions, so `_cap_steps`
        falls back to the full step count. `select_conf` uses this to
        hold the coverage walk back on a fresh system: that walk pulls
        *global* top confs (often slow recurrent architectures tuned on
        faster systems), and without a timing model it would run them
        at full length here. The neighbor walk -- local, budget-bounded
        confs -- still runs and is what seeds the timing model; once
        it's fit, the coverage walk turns on and catches up.
        """
        return any(s == system for s, _ in self.timing_model.keys())

    def _cap_steps(
        self, conf: Configuration, system: str
    ) -> Configuration:
        """Return `conf` with `steps` halved until predicted train_time
        fits the global maxt budget.

        Used by neighbor selection on dense<->rnn / latent<->lrnn
        transitions where per-step cost can blow up 1000x. We can't
        just drop the candidate (it'd be re-picked next iteration) and
        we can't time-out the training (cosine decay needs the full
        schedule), so we cap the work up front.

        If no prediction exists (cold start on a fresh (system,
        precision)), returns `conf` unchanged — first run on a new pair
        seeds the timing model. If even `template.steps.min` predicts
        over budget, returns the conf at the floor (bounded overrun).
        """
        maxt = self.train_time[1]
        min_steps = max(self.template.steps.min, 2)
        steps = conf.steps
        pred = self._predicted_time(conf, system)
        while steps > min_steps and pred is not None and pred > maxt:
            steps = max(steps // 2, min_steps)
            conf = conf.replace(steps=steps)
            pred = self._predicted_time(conf, system)
        return conf

    def _select_neighbor_fewest_runs(
        self, conf: Configuration, system: str, template: Template,
    ) -> Optional[tuple[Configuration, int, int]]:
        """Find the neighbor with fewest runs that matches the template.

        Returns (neighbor_conf, total_runs, system_runs) or None. The
        caller decides whether the candidate is urgent enough to run:
        a neighbor with system_runs == 0 is always eligible to be
        selected; otherwise it's compared against the min-runs threshold
        from _generate_limits.
        """
        # `conf` itself may be retired -- it stays a legal source, only
        # its retired neighbors (including the mode swap back onto
        # itself) drop out.
        neighbors = [
            n for n in conf_neighbors(conf, template)
            if not _is_retired_conf(n)
        ]
        if not neighbors:
            return None

        # Cap predicted train_time at maxt per-neighbor by halving steps.
        # The set dedupes in case two original neighbors collapse onto
        # the same capped conf, and we drop `conf` itself in case the
        # cap brought a higher-steps neighbor back onto the seed.
        neighbors = {self._cap_steps(n, system) for n in neighbors}
        neighbors.discard(conf)

        neighbor_ids = [
            cid for cid in (self._db.get_conf_id(n) for n in neighbors)
            if cid is not None
        ]
        run_counts = self._db.get_run_counts(neighbor_ids, system=system)

        # Prefer neighbors with system_runs == 0 (bootstrap visibility on
        # this system), then among the rest pick the one with the fewest
        # total runs.
        best_conf = None
        best_total = INF
        best_system = INF
        for n in neighbors:
            cid = self._db.get_conf_id(n)
            if cid is None:
                total, sys_runs = 0, 0
            else:
                total, sys_runs = run_counts.get(cid, (0, 0))

            # Tuple ordering: (system_runs == 0 first, then fewest total).
            key = (sys_runs > 0, total)
            best_key = (best_system > 0, best_total)
            if key < best_key:
                best_conf = n
                best_total = total
                best_system = sys_runs

        if best_conf is not None:
            return best_conf, best_total, best_system
        return None

    def _select_top_neighbor(
            self, t: float, max_weights: int, system: str,
            template: Template,
    ) -> Optional[Configuration]:
        with latency.timer("Search._select_top_neighbor"):
            top_confs = list(
                self._db.top_confs_for_system(
                    max_time=t, max_weights=max_weights, system=system,
                    limit=10, template=template
                )
            )
            if not top_confs:
                return None
            logging.info(top_confs_report(confs=top_confs,
                         max_weights=max_weights, max_time=t, system=system))

            have_confs = 10
            min_runs_neighbor = []

            for seq in _generate_limits():
                for i, (min_self, min_neighbor) in enumerate(seq):
                    end = 3**i
                    start = end // 3

                    if end > have_confs:
                        logging.info(f'Getting top confs up to {end}')
                        top_confs = list(
                            self._db.top_confs_for_system(
                                max_time=t, max_weights=max_weights, system=system,
                                limit=end, template=template
                            )
                        )
                        have_confs = end

                    for j in range(start, min(end, len(top_confs))):
                        if _is_retired_conf(top_confs[j].conf):
                            # Not re-runnable, but still expanded as a
                            # mutation source in the neighbor pass below.
                            continue
                        if top_confs[j].num_runs < min_self:
                            logging.info(
                                f'Getting conf {j} because it has {top_confs[j].num_runs} runs < {min_self}')
                            return top_confs[j].conf

                    if min_neighbor == 0:
                        continue

                    for j in range(start, min(end, len(top_confs))):
                        if len(min_runs_neighbor) < j + 1:
                            result = self._select_neighbor_fewest_runs(
                                top_confs[j].conf, system, template)
                            min_runs_neighbor.append(result)

                        result = min_runs_neighbor[j]
                        if result is not None:
                            neighbor_conf, total_runs, system_runs = result
                            # Pick the neighbor if:
                            # (a) it has no runs on this system yet, or
                            # (b) its total run count is below the threshold.
                            if system_runs == 0:
                                logging.info(
                                    f'Getting neighbor of conf {j}: {system_runs} runs on {system}')
                                return neighbor_conf
                            if total_runs < min_neighbor:
                                logging.info(
                                    f'Getting neighbor of conf {j}: {total_runs} total runs < {min_neighbor}')
                                return neighbor_conf

    def _select_time_budget(
        self, t: float, max_weights: int, system: str,
        template: Template,
    ) -> Optional[Configuration]:
        """Score-ordered scan with a widening total-runs requirement.

        Pulls confs from the DB lazily in score order — we only
        instantiate as many as we actually look at across iterations.
        For each iteration of `_run_limit_sequences`, scan positions
        until we find a conf whose total_runs is below the iteration's
        requirement at its position (or which has zero runs on this
        system). Termination: iteration N's position-0 requirement is
        N, so once N exceeds candidates[0].total_runs we pick it.
        """
        with latency.timer("Search._select_time_budget"):
            # Pure re-run picks -- nothing is mutated here, so retired
            # confs are dropped from the stream outright.
            source = (
                c for c in self._db.confs_under_time(
                    template=template, system=system,
                    max_weights=max_weights, max_time=t,
                )
                if not _is_retired_conf(c.conf)
            )
            candidates: list[ConfWithRuns] = []
            exhausted = False

            # Materialize the top 10 up front so we can log them like the
            # neighbor-walk strategy does.
            while len(candidates) < 10 and not exhausted:
                try:
                    candidates.append(next(source))
                except StopIteration:
                    exhausted = True
            if not candidates:
                return None
            logging.info(time_budget_report(
                candidates, max_weights=max_weights, max_time=t,
                system=system,
            ))

            for limits in _run_limit_sequences():
                for i in range(len(limits)):
                    if i >= len(candidates) and not exhausted:
                        try:
                            candidates.append(next(source))
                        except StopIteration:
                            exhausted = True

                    if i >= len(candidates):
                        break
                    c = candidates[i]

                    if c.system_runs == 0 or c.total_runs < limits[i]:
                        logging.info(
                            f'Time-budget conf for {system} at pos {i}: '
                            f'{c.conf} '
                            f'(runs: {c.total_runs} total, '
                            f'{c.system_runs} on system; limit {limits[i]})'
                        )
                        return c.conf
                if exhausted and not candidates:
                    return None

            return None

    def _advance_warmup(self, system: str, rung: int, why: str) -> None:
        self._warmup_rung[system] = rung + 1
        self._warmup_selects[system] = 0
        if rung + 1 >= len(_WARMUP_LADDER):
            logging.info(
                f'Warmup finished for {system} ({why}); ordinary '
                f'selection from here')
            return
        steps, batch, length = _WARMUP_LADDER[rung + 1]
        logging.info(
            f'Warmup {system}: rung {rung + 2}/{len(_WARMUP_LADDER)} '
            f'({why}) -- steps<={steps} batch<={batch} length<={length}')

    def _warmup_top_confs(self, system: str, rung: int) -> list:
        cached = self._warmup_top.get(system)
        if cached is not None and cached[0] == rung:
            return cached[1]
        template = _warmup_template(self.template, *_WARMUP_LADDER[rung])
        top = list(self._db.top_confs_global(template, min_num_runs=1))
        self._warmup_top[system] = (rung, top)
        return top

    def _warmup_pick(
        self, system: str, rung: int) -> Optional[Configuration]:
        """First global top conf this rung admits, in random order,
        that `system` hasn't covered yet.

        The conf is taken AS IT STANDS -- the rung caps filter the
        candidates rather than rewriting them. Shrinking a top conf's
        knobs to fit would mint a fresh (spec, steps, batch, length)
        combination the fleet has never run, so its one run here would
        be the only run it ever has: no cross-system comparison, and
        below the two runs `top_confs_global` wants before showing a
        conf at all. Filtering keeps every warmup run landing on a conf
        others have already measured.
        """
        top = [c for c in self._warmup_top_confs(system, rung)
               if not _is_retired_conf(c.conf)]
        random.shuffle(top)
        for c in top:
            # No-op until this system has a timing model; once it does,
            # the ordinary budget cap applies on top of the rung.
            conf = self._cap_steps(c.conf, system)
            if self._db.has_covering_run(conf, system):
                continue
            return conf
        return None

    def _select_warmup(self, system: str) -> Optional[Configuration]:
        """Pick a deliberately bounded conf for a system still climbing
        the `_WARMUP_LADDER`, or None once it has finished the ladder
        (the caller then falls through to ordinary selection).

        Each rung admits top confs whose (steps, batch, length) are
        within its caps; the rung advances after
        `_WARMUP_SELECTS_PER_RUNG` picks, or immediately when it has no
        uncovered confs left for this system (including none at all,
        which is normal for the tightest rungs on a small DB).
        """
        with latency.timer('Search._select_warmup'):
            while True:
                rung = self._warmup_rung.get(system, 0)
                if rung >= len(_WARMUP_LADDER):
                    return None
                conf = self._warmup_pick(system, rung)
                if conf is None:
                    self._advance_warmup(
                        system, rung, 'no uncovered confs at this rung')
                    continue
                spent = self._warmup_selects.get(system, 0) + 1
                self._warmup_selects[system] = spent
                if spent >= _WARMUP_SELECTS_PER_RUNG:
                    self._advance_warmup(system, rung, 'picks spent')
                return conf

    def _select_uncovered_top(
        self, system: str, entry: TemplateEntry,
    ) -> Optional[Configuration]:
        """Walk every top conf `entry`'s template admits in random
        order, cap each for `system`, and return the first whose capped
        variant has no covering run on `system` (= no run with the
        same spec/lr/length/batch/precision/decay/cosine and steps ≥
        the capped count).

        The frontier is the one the ENTRY sees: a conf another system
        found that is Pareto-optimal for this sub-template but not
        globally still deserves a cross-system re-run here.

        Sets `self._coverage_flag[(system, entry.name)]` to True when
        `_COVERAGE_STICKY_THRESHOLD`+ uncovered confs are found, so
        the next select for that pair fires this walk unconditionally;
        resets to False otherwise. The randomized order keeps
        successive selects from picking the same conf during the
        prefetch gap window (before the previous run is written back).
        """
        with latency.timer('Search._select_uncovered_top'):
            # `top_confs_global` applies the template's max_weights
            # bound itself via `_make_template_conditions`, so no extra
            # max_weights filter here. Materialized so we can shuffle.
            # `min_num_runs=1` so even single-run top confs get picked
            # up — the whole point of this walk is to verify those
            # faster across systems.
            # Retired confs drop out before the covered/uncovered
            # accounting: counted as uncovered they could never be
            # closed, and the sticky flag would latch on forever.
            top = [c for c in self._db.top_confs_global(
                       entry.template, min_num_runs=1)
                   if not _is_retired_conf(c.conf)]
            random.shuffle(top)

            selected: Optional[Configuration] = None
            uncovered = 0
            for c in top:
                capped = self._cap_steps(c.conf, system)
                if self._db.has_covering_run(capped, system):
                    continue
                uncovered += 1
                if selected is None:
                    selected = capped

            sticky = uncovered >= _COVERAGE_STICKY_THRESHOLD
            covered = len(top) - uncovered
            logging.info(
                f'Coverage walk {system}/{entry.name}: '
                f'{covered}/{len(top)} top confs covered, sticky={sticky}'
            )
            key = (system, entry.name)
            self._coverage_flag[key] = sticky
            self._coverage_stats[key] = (covered, len(top))
            return selected

    def _select_predicted_best(
        self, t: float, max_weights: int, system: str, bfs_depth: int,
        template: Template,
    ) -> Optional[Configuration]:
        """Predictor-guided tournament.

        Take the best-known conf for (system, template, max_weights,
        max_time<=t) as a seed, BFS out to `bfs_depth` neighbors,
        adjust each candidate's steps to fit the time budget, score
        them with a compound median(predicted_loss, run_losses...),
        then walk the top 9 across the [1] / [2,1,1] / [3,2,2,1x6]
        run-limit sequences.
        """
        if not self.loss_model.is_ready():
            return None
        with latency.timer(
            f'Search._select_predicted_best.depth{bfs_depth}'
        ):
            return self._select_predicted_best_impl(
                t, max_weights, system, bfs_depth, template)

    def _select_predicted_best_impl(
        self, t: float, max_weights: int, system: str, bfs_depth: int,
        template: Template,
    ) -> Optional[Configuration]:
        try:
            seed = next(self._db.top_confs_for_system(
                system=system, template=template,
                max_weights=max_weights, max_time=t, limit=1,
            ))
        except StopIteration:
            return None
        seed_conf = seed.conf

        # BFS to `bfs_depth` from the seed, including the seed itself.
        # We dedupe on the raw (unnormalized) conf so that step and
        # batch mutations stay distinct in the walk — over-budget
        # intermediates are fine, they just won't survive the
        # post-BFS normalization. (W, T) variety from the neighbor
        # walk is preserved end-to-end this way.
        visited: set[Configuration] = {seed_conf}
        frontier: list[Configuration] = [seed_conf]
        deadline = time.monotonic() + _BFS_TIME_BUDGET_S
        clipped: str | None = None
        for _ in range(bfs_depth):
            next_frontier: list[Configuration] = []
            for c in frontier:
                if time.monotonic() >= deadline:
                    clipped = 'time budget'
                    break
                if len(visited) >= _BFS_VISITED_CAP:
                    clipped = 'visited cap'
                    break
                for n in conf_neighbors(c, template):
                    if n in visited:
                        continue
                    visited.add(n)
                    next_frontier.append(n)
            if clipped or not next_frontier:
                break
            # Frontier order decides which parents expand before the
            # visited cap trips, and the seed rarely changes between
            # selects: unrandomized, the cap would truncate the same
            # enumeration-order prefix every time (biased toward the
            # first layers' mutations) and the rest of the ball would
            # never be scored. random.sample already returns its picks
            # shuffled; sampled-out confs stay in `visited` and still
            # get scored.
            if len(next_frontier) > _BFS_FRONTIER_CAP:
                next_frontier = random.sample(
                    next_frontier, _BFS_FRONTIER_CAP)
            else:
                random.shuffle(next_frontier)
            frontier = next_frontier
        if clipped:
            logging.info(
                f'predicted-best BFS clipped by {clipped} '
                f'(depth {bfs_depth}, |visited|={len(visited)}, {system})')

        # Filter by weight budget *after* the BFS, so intermediate
        # neighbors with too many weights can still lead us to
        # smaller-but-different final candidates at depth N. Retired
        # confs are filtered here too, for the same reason: the BFS
        # still expands through them (a shift seed's hexbpe bridge
        # stays reachable), they just can't be picked.
        weight_filtered = [
            c for c in visited
            if c.num_weights <= max_weights and not _is_retired_conf(c)]
        if not weight_filtered:
            return None

        # Seed selection uses the sampled `t`; the step-fit cap uses
        # the configured maximum training time. We want to score
        # candidates at the largest budget the user has authorized,
        # not at whatever `t` happened to be sampled this iteration.
        max_t = self.train_time[1]

        # Normalize each weight-budget candidate: leave alone if its
        # predicted total time already fits max_t, cap c.steps to the
        # largest pow2 that does fit otherwise, drop the conf if even
        # the smallest training run won't fit (or the timing model
        # isn't fit for this system/precision). Multiple over-budget
        # neighbors can normalize to the same capped conf, so dedupe.
        def _norm(c: Configuration) -> Optional[Configuration]:
            predicted = self.timing_model.predict(system, c)
            if predicted is None:
                return None
            if predicted <= max_t:
                return c
            ms = self.timing_model.predict_max_steps(system, c, max_t)
            if ms is None or ms < 2:
                return None
            return c.replace(steps=ms)

        adjusted_set: set[Configuration] = set()
        for c in weight_filtered:
            nc = _norm(c)
            if nc is not None:
                adjusted_set.add(nc)
        if not adjusted_set:
            return None
        adjusted = list(adjusted_set)

        # Predict loss for each adjusted conf (log2 space).
        log_preds = self.loss_model.predict(adjusted)

        # Look up any existing runs for these (adjusted) confs.
        id_of = {c: self._db.get_conf_id(c) for c in adjusted}
        ids = [cid for cid in id_of.values() if cid is not None]
        losses_by_id = self._db.get_losses_by_conf_ids(ids)

        # Compound score in log2-space.
        candidate_data = []
        for i, c in enumerate(adjusted):
            cid = id_of[c]
            raw_losses = losses_by_id.get(cid, []) if cid else []
            log_losses = [
                np.log2(np.clip(l, 0.1, 10)) for l in raw_losses
            ]
            compound = float(np.median([float(log_preds[i])] + log_losses))
            candidate_data.append(
                (compound, len(raw_losses), c))
        candidate_data.sort(key=lambda r: r[0])

        logging.info(_predicted_best_report(
            seed, candidate_data, system, max_weights, t, bfs_depth))

        # Run-limit sequences (cap at iteration 3 / top 9 confs).
        sequences = [[1], [2, 1, 1], [3, 2, 2, 1, 1, 1, 1, 1, 1]]
        for limits in sequences:
            for i, limit in enumerate(limits):
                if i >= len(candidate_data):
                    break
                compound, total_runs, c = candidate_data[i]
                if total_runs < limit:
                    logging.info(
                        f'Predicted-best conf for {system} at pos {i}: '
                        f'{c} (runs={total_runs} < limit={limit}, '
                        f'compound={2.0 ** compound:.4f} b/B)'
                    )
                    return c
        return None

    def _result(
        self, conf: Configuration, strategy: str, system: str,
        covered: int | None = None, total: int | None = None,
        entry: Optional[TemplateEntry] = None,
    ) -> SearchResult:
        """Build the SearchResult and log it. `entry` (when the pick
        came from a drawn sub-template) also books the select against
        that entry's realized-share counter."""
        if entry is None:
            logging.info(f'Conf for {system}: {conf} ({strategy})')
        else:
            self.entry_selects[entry.name] = (
                self.entry_selects.get(entry.name, 0) + 1)
            logging.info(
                f'Conf for {system}: {conf} ({strategy}, {entry.name})')
        return SearchResult(conf, strategy, system, covered, total)

    def _select_default(
        self, system: str, entry: TemplateEntry,
    ) -> Optional[Configuration]:
        """Return the DRAWN entry's default conf, unless it no longer
        matches that entry's template or has already been explored
        enough on this system.

        This is what bootstraps an empty sub-space: without it a
        brand-new entry returns None on every select and its budget
        evaporates into the general search.
        """
        default = entry.default
        if _is_retired_conf(default):
            logging.info(
                f'No conf for {system}: default conf for {entry.name!r} '
                f'uses a retired input')
            return None
        if not entry.template.match(default):
            logging.info(
                f'No conf for {system}: default conf for {entry.name!r} '
                f'does not match the template')
            return None
        conf_id = self._db.get_conf_id(default)
        if conf_id is not None:
            counts = self._db.get_run_counts([conf_id], system=system)
            total_runs, system_runs = counts.get(conf_id, (0, 0))
            if total_runs >= 7 and system_runs >= 1:
                logging.info(
                    f'No conf for {system}: default conf for '
                    f'{entry.name!r} has {total_runs} total runs '
                    f'({system_runs} on this system)')
                return None
        return default

    def _sample_layer_capped_template(
        self, entry: TemplateEntry,
    ) -> tuple[Template, Optional[int]]:
        """Sample a per-select num_layers cap and intersect it with the
        entry's template. Caps below the entry's own minimum layer count
        are dropped from the distribution first (see
        `_layer_cap_probs`)."""
        probs = _layer_cap_probs(entry.min_layers)
        (cap, _), = random.choices(
            probs,
            weights=[w for _, w in probs],
            k=1,
        )
        return _layer_capped_template(entry.template, cap)

    def _run_strategy(
        self, name: str, t: float, max_weights: int, system: str,
        template: Template,
    ) -> Optional[Configuration]:
        match name:
            case 'predicted_2nd_neighbor':
                return self._select_predicted_best(
                    t, max_weights, system, bfs_depth=2, template=template)
            case 'predicted_3rd_neighbor':
                return self._select_predicted_best(
                    t, max_weights, system, bfs_depth=3, template=template)
            case 'time_budget':
                return self._select_time_budget(
                    t, max_weights, system, template)
            case 'neighbor':
                return self._select_top_neighbor(
                    t, max_weights, system, template)
            case _:
                raise ValueError(f"unknown strategy: {name}")

    def select_conf(self, system: str) -> Optional[SearchResult]:
        """Select a SearchResult, or None if nothing matches.

        Pick-me confs (explicit user-injected candidates with
        `pick_me = 1`) take absolute priority until each has
        `PICK_ME_MIN_RUNS` total runs. The warmup ladder comes next.
        Both are template-blind: they run against the base template.

        Then ONE share-weighted draw picks the sub-template for this
        select, and everything below it -- coverage walk, layer cap,
        weight ceiling, strategies and both fallbacks -- runs inside
        that entry. With a single entry this is exactly the
        pre-template-set path.

        The coverage walk runs first (independent of t / max_weights)
        and fires either with probability `_COVERAGE_PROB` or
        unconditionally when its sticky flag is set for this
        (system, entry).

        After that, one weighted random draw over `_STRATEGY_PROBS`
        picks the main strategy. If the picked strategy returns None
        (e.g. `predicted_*` without a fitted loss model, or
        `time_budget` with no qualifying confs), we fall back to the
        neighbor walk. If neighbor too is unavailable, fall back to
        the entry's default conf (skipped if it doesn't match the
        entry's template or has already been explored enough).
        """
        with latency.timer("Search.select_conf"):
            # Pick-me: explicit user picks bypass every other strategy
            # until they've been measured enough times.
            pm = self._db.pick_me_conf(
                self.template, min_runs=PICK_ME_MIN_RUNS)
            if pm is not None and _is_retired_conf(pm):
                logging.warning(
                    f'Ignoring pick_me conf {pm}: retired input')
                pm = None
            if pm is not None:
                return self._result(pm, 'pick_me', system)

            # Warmup ladder: a system started with --warmup-systems
            # only gets confs whose (steps, batch, length) fit the
            # current rung, so a machine/backend with no timing model
            # can't be handed an hours-long run. Returns None once the
            # ladder is done and the system joins ordinary selection.
            if system in self.warmup_systems:
                conf = self._select_warmup(system)
                if conf is not None:
                    return self._result(conf, 'warmup', system)
                self.warmup_systems.discard(system)

            # Sub-template draw: one share-weighted pick per select,
            # after which this whole call lives inside that entry.
            entry = self.templates.draw()
            if not self.templates.is_single:
                logging.info(
                    f'Sub-template for {system}: {entry.name} '
                    f'(spec={entry.spec})')

            # Coverage walk runs *before* t / max_weights selection so
            # the bulk cross-system push isn't confined to the current
            # iteration's weight bucket. Sticky on this (system, entry)
            # when the previous walk left more uncovered to do. Held
            # back until the system has a fitted timing model: without
            # one we can't cap the (often slow, recurrent) global top
            # confs this walk pulls, so a fresh system would run them at
            # full length. The neighbor walk seeds the timing model
            # first; once it's fit, this turns on and catches up on the
            # entry's top confs.
            if self._timing_ready(system) and (
                self._coverage_flag.get((system, entry.name), False)
                or random.random() < _COVERAGE_PROB
            ):
                conf = self._select_uncovered_top(system, entry)
                if conf is not None:
                    covered, total = self._coverage_stats.get(
                        (system, entry.name), (None, None))
                    return self._result(
                        conf, 'coverage_walk', system, covered, total,
                        entry=entry)

            # Layer-count diversification: this select (seed queries,
            # neighbor/BFS expansion, and hence the trained conf) may run
            # under a sampled num_layers cap on top of the entry's
            # template. The fallback chain below stays capped too; only
            # the final default fallback is uncapped.
            template, layer_cap = self._sample_layer_capped_template(entry)
            if layer_cap is not None:
                logging.info(
                    f'Layer-capped select for {system}: '
                    f'num_layers <= {layer_cap}')

            t = self._select_time()
            # Derived from the best conf WITHIN this (capped) entry, so
            # each sub-space gets a weight range matched to its own
            # population rather than to the global frontier.
            max_weights = self._select_max_weights(t, system, template)

            picked, _ = random.choices(
                _STRATEGY_PROBS,
                weights=[w for _, w in _STRATEGY_PROBS],
                k=1,
            )[0]
            conf = self._run_strategy(picked, t, max_weights, system, template)
            if conf is not None:
                return self._result(conf, picked, system, entry=entry)

            # First-line strategy was unavailable -- always fall back
            # to the neighbor walk.
            if picked != 'neighbor':
                conf = self._run_strategy(
                    'neighbor', t, max_weights, system, template)
                if conf is not None:
                    return self._result(
                        conf, 'neighbor', system, entry=entry)

            conf = self._select_default(system, entry)
            if conf is not None:
                return self._result(conf, 'default', system, entry=entry)
            return None


class SearchThread(threading.Thread):
    """Drives the search loop. Read-only — writes go straight from the
    request handler into `write_queue`, with no Search detour.

    Reads `select` / `set_template` / `stop` commands off the requests
    queue. For `select` it asks `Search` for a candidate and drops the
    result onto the per-system response queue.
    """

    def __init__(
        self,
        path: str,
        template: Template,
        train_time: tuple[float, float],
        default: Configuration,
        requests_queue: Queue,
        confs_by_system: dict,
        confs_by_system_lock: threading.Lock,
        timing_model: TrainTimingModel,
        loss_model: LossModelHolder,
        warmup_systems: Optional[Iterable[str]] = None,
        templates: Optional[TemplateSet] = None,
    ):
        # Daemon like every other server thread: this one is
        # read-only, and durability belongs to the graceful path
        # (`SearchServer.join` posts `Stop` and joins) rather than to
        # holding the interpreter open.
        super().__init__(daemon=True)
        # Search and the DbReader are built lazily on `run()` so the
        # reader is opened on the thread that uses it; that lets
        # `DbReader` enforce `check_same_thread=True`.
        self._path = path
        self._warmup_systems = set(warmup_systems or ())
        self._template = template
        self._templates = templates
        self._train_time = train_time
        self._default = default
        self._timing_model = timing_model
        self._loss_model = loss_model
        self.template = template
        self.requests_queue = requests_queue
        self.confs_by_system = confs_by_system
        self.confs_by_system_lock = confs_by_system_lock
        _, self.max_time = train_time
        self.search: Optional[Search] = None

    def run(self):
        reader = DbReader(self._path)
        self.search = Search(
            reader=reader,
            template=self._template,
            init_conf=self._default,
            train_time=self._train_time,
            timing_model=self._timing_model,
            loss_model=self._loss_model,
            warmup_systems=self._warmup_systems,
            templates=self._templates,
        )
        logging.info("Started search thread")
        if self.search.templates.is_single:
            logging.info("Single template (no sub-searches)")
        else:
            for entry in self.search.templates:
                logging.info(
                    f'Sub-search {entry.name}: '
                    f'{self.search.templates.nominal_share(entry):.0f}% '
                    f'spec={entry.spec} default={entry.default.model}')
        if self._warmup_systems:
            logging.info(
                f"Warmup mode for {sorted(self._warmup_systems)}: "
                f"{len(_WARMUP_LADDER)} rungs from "
                f"{_WARMUP_LADDER[0]} to {_WARMUP_LADDER[-1]} "
                f"(steps, batch, length)")
        try:
            while True:
                m = self.requests_queue.get()
                match m:
                    case Select(system=system):
                        age = time.monotonic() - m.enqueued_at
                        if age > _SELECT_STALE_S:
                            logging.info(
                                f'select_conf({system}) skipped: request '
                                f'{age:.0f}s old, client timed out')
                            with self.confs_by_system_lock:
                                self.confs_by_system[system].put(None)
                            continue
                        # A failure in select_conf must not kill the
                        # thread -- if it did, every client's /select
                        # would block forever on the response queue.
                        # Log it, hand back None (the client sleeps and
                        # retries), and keep serving.
                        # Start/done logs bracket the (serial) work so a
                        # slow call is visible live, with the backlog it
                        # is holding up.
                        logging.info(
                            f'select_conf({system}) start '
                            f'(queue depth {self.requests_queue.qsize()})')
                        t0 = time.monotonic()
                        try:
                            result = self.search.select_conf(system)
                        except Exception:
                            logging.exception(
                                f"select_conf failed for {system!r}; "
                                f"returning no conf")
                            result = None
                        logging.info(
                            f'select_conf({system}) done in '
                            f'{ttoa3(time.monotonic() - t0)} '
                            f'({"no conf" if result is None else result.strategy})')
                        with self.confs_by_system_lock:
                            self.confs_by_system[system].put(result)
                    case SetTemplate(
                        template=template,
                        init_conf=init_conf,
                        train_time=train_time,
                        templates=templates,
                    ):
                        self.search.template = template
                        self.search.init_conf = init_conf
                        self.search.train_time = train_time
                        self.search.set_templates(templates)
                        self.max_time = train_time[1]
                        logging.info(
                            f'Search thread: new template {template}')
                        if not templates.is_single:
                            logging.info(
                                f'Search thread: new sub-searches '
                                f'{templates}')
                    case Stop():
                        logging.info("Stopping search thread")
                        break
                    case _:
                        assert False, f"Unknown message: {m!r}"
        finally:
            reader.close()
