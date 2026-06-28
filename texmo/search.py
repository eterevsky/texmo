import logging
import random
import threading
from dataclasses import dataclass
from math import log2
from queue import Queue
from typing import Optional

import numpy as np

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration import Configuration, Template, conf_neighbors
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


@dataclass
class SetTemplate:
    """Reconfigure `Search` between two `select_conf` calls. The trio
    moves together so a partial update never lands on the search loop."""
    template: Template
    init_conf: Configuration
    train_time: tuple[float, float]


@dataclass
class Stop:
    """Stop the search thread."""


SearchMessage = Select | SetTemplate | Stop

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

# Minimum total runs a pick_me conf needs before it stops being
# picked preferentially. Two runs give a median that's robust to a
# single-run outlier, which is what `top_confs_global` already uses.
PICK_ME_MIN_RUNS = 2

# Sticky-flag threshold: keep firing the coverage walk only when at
# least this many uncovered top confs remain on the system. Set high
# enough that successive selects within the prefetch gap window
# don't pile up on the same few confs before any run finishes.
_COVERAGE_STICKY_THRESHOLD = 5


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
    ):
        assert isinstance(reader, DbReader)
        # `_db` is the read handle used by every strategy below. Search
        # is read-only; `add_run` now goes straight to the writer queue
        # from `SearchThread` without a Search detour.
        self._db = reader

        assert isinstance(template, Template)
        self.template = template

        assert isinstance(init_conf, Configuration)
        self.init_conf = init_conf

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self.train_time = train_time

        # Shared references with the Model thread. `timing_model` is
        # mutated in place (per-(system, precision) dict inserts are
        # atomic in CPython); `loss_model` is a holder whose underlying
        # `LossModel` gets swapped atomically on refit.
        self.timing_model = timing_model
        self.loss_model = loss_model

        # Sticky per-system flag set by `_select_uncovered_top` when it
        # finds 2+ uncovered top confs — keeps the coverage walk firing
        # at 100% on the next select for that system until the gap
        # closes. Reset by the same call when fewer than 2 are found.
        self._coverage_flag: dict[str, bool] = {}

        # Last (covered, total) top-conf counts from `_select_uncovered_top`,
        # per system. Attached to coverage-walk SearchResults so the
        # client can log progress.
        self._coverage_stats: dict[str, tuple[int, int]] = {}

    def _select_time(self) -> float:
        tmin, tmax = self.train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t, system: str):
        with latency.timer("Search._select_max_weights"):
            try:
                top_conf = next(
                    self._db.top_confs_for_system(
                        max_time=t, system=system, limit=1, template=self.template)
                )
            except StopIteration:
                return self.template.max_weights.min
            maxw = 8 * top_conf.conf.model.num_weights
            if self.template.max_weights.min >= maxw:
                return self.template.max_weights.min
            if self.template.max_weights.max <= maxw:
                maxw = self.template.max_weights.max
            l = random.uniform(
                log2(self.template.max_weights.min), log2(maxw))
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
        self, conf: Configuration, system: str
    ) -> Optional[tuple[Configuration, int, int]]:
        """Find the neighbor with fewest runs that matches the template.

        Returns (neighbor_conf, total_runs, system_runs) or None. The
        caller decides whether the candidate is urgent enough to run:
        a neighbor with system_runs == 0 is always eligible to be
        selected; otherwise it's compared against the min-runs threshold
        from _generate_limits.
        """
        neighbors = conf_neighbors(conf, self.template)
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
            self, t: float, max_weights: int, system: str
    ) -> Optional[Configuration]:
        with latency.timer("Search._select_top_neighbor"):
            top_confs = list(
                self._db.top_confs_for_system(
                    max_time=t, max_weights=max_weights, system=system,
                    limit=10, template=self.template
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
                                limit=end, template=self.template
                            )
                        )
                        have_confs = end

                    for j in range(start, min(end, len(top_confs))):
                        if top_confs[j].num_runs < min_self:
                            logging.info(
                                f'Getting conf {j} because it has {top_confs[j].num_runs} runs < {min_self}')
                            return top_confs[j].conf

                    if min_neighbor == 0:
                        continue

                    for j in range(start, min(end, len(top_confs))):
                        if len(min_runs_neighbor) < j + 1:
                            result = self._select_neighbor_fewest_runs(
                                top_confs[j].conf, system)
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
        self, t: float, max_weights: int, system: str
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
            source = iter(self._db.confs_under_time(
                template=self.template, system=system,
                max_weights=max_weights, max_time=t,
            ))
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

    def _select_uncovered_top(
        self, system: str
    ) -> Optional[Configuration]:
        """Walk every top conf the template admits in random order,
        cap each for `system`, and return the first whose capped
        variant has no covering run on `system` (= no run with the
        same spec/lr/length/batch/precision/decay/cosine and steps ≥
        the capped count).

        Sets `self._coverage_flag[system]` to True when
        `_COVERAGE_STICKY_THRESHOLD`+ uncovered confs are found, so
        the next select on this system fires this walk
        unconditionally; resets to False otherwise. The randomized
        order keeps successive selects from picking the same conf
        during the prefetch gap window (before the previous run is
        written back).
        """
        with latency.timer('Search._select_uncovered_top'):
            # `top_confs_global` applies the template's max_weights
            # bound itself via `_make_template_conditions`, so no extra
            # max_weights filter here. Materialized so we can shuffle.
            # `min_num_runs=1` so even single-run top confs get picked
            # up — the whole point of this walk is to verify those
            # faster across systems.
            top = list(self._db.top_confs_global(
                self.template, min_num_runs=1))
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
                f'Coverage walk {system}: {covered}/{len(top)} '
                f'top confs covered, sticky={sticky}'
            )
            self._coverage_flag[system] = sticky
            self._coverage_stats[system] = (covered, len(top))
            return selected

    def _select_predicted_best(
        self, t: float, max_weights: int, system: str, bfs_depth: int,
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
                t, max_weights, system, bfs_depth)

    def _select_predicted_best_impl(
        self, t: float, max_weights: int, system: str, bfs_depth: int,
    ) -> Optional[Configuration]:
        try:
            seed = next(self._db.top_confs_for_system(
                system=system, template=self.template,
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
        for _ in range(bfs_depth):
            next_frontier: list[Configuration] = []
            for c in frontier:
                for n in conf_neighbors(c, self.template):
                    if n in visited:
                        continue
                    visited.add(n)
                    next_frontier.append(n)
            if not next_frontier:
                break
            frontier = next_frontier

        # Filter by weight budget *after* the BFS, so intermediate
        # neighbors with too many weights can still lead us to
        # smaller-but-different final candidates at depth N.
        weight_filtered = [
            c for c in visited if c.num_weights <= max_weights]
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
    ) -> SearchResult:
        logging.info(f'Conf for {system}: {conf} ({strategy})')
        return SearchResult(conf, strategy, system, covered, total)

    def _select_default(self, system: str) -> Optional[Configuration]:
        """Return init_conf, unless it no longer matches the current
        template or has already been explored enough on this system."""
        if not self.template.match(self.init_conf):
            logging.info(
                f'No conf for {system}: init_conf does not match template')
            return None
        conf_id = self._db.get_conf_id(self.init_conf)
        if conf_id is not None:
            counts = self._db.get_run_counts([conf_id], system=system)
            total_runs, system_runs = counts.get(conf_id, (0, 0))
            if total_runs >= 7 and system_runs >= 1:
                logging.info(
                    f'No conf for {system}: init_conf has {total_runs} '
                    f'total runs ({system_runs} on this system)')
                return None
        return self.init_conf

    def _run_strategy(
        self, name: str, t: float, max_weights: int, system: str,
    ) -> Optional[Configuration]:
        match name:
            case 'predicted_2nd_neighbor':
                return self._select_predicted_best(
                    t, max_weights, system, bfs_depth=2)
            case 'predicted_3rd_neighbor':
                return self._select_predicted_best(
                    t, max_weights, system, bfs_depth=3)
            case 'time_budget':
                return self._select_time_budget(t, max_weights, system)
            case 'neighbor':
                return self._select_top_neighbor(t, max_weights, system)
            case _:
                raise ValueError(f"unknown strategy: {name}")

    def select_conf(self, system: str) -> Optional[SearchResult]:
        """Select a SearchResult, or None if nothing matches.

        Pick-me confs (explicit user-injected candidates with
        `pick_me = 1`) take absolute priority until each has
        `PICK_ME_MIN_RUNS` total runs.

        Then the coverage walk runs (independent of t / max_weights)
        and fires either with probability `_COVERAGE_PROB` or
        unconditionally when its sticky flag is set on this system.

        After that, one weighted random draw over `_STRATEGY_PROBS`
        picks the main strategy. If the picked strategy returns None
        (e.g. `predicted_*` without a fitted loss model, or
        `time_budget` with no qualifying confs), we fall back to the
        neighbor walk. If neighbor too is unavailable, fall back to
        the default conf (skipped if it doesn't match the current
        template or has already been explored enough).
        """
        with latency.timer("Search.select_conf"):
            # Pick-me: explicit user picks bypass every other strategy
            # until they've been measured enough times.
            pm = self._db.pick_me_conf(
                self.template, min_runs=PICK_ME_MIN_RUNS)
            if pm is not None:
                return self._result(pm, 'pick_me', system)

            # Coverage walk runs *before* t / max_weights selection so
            # the bulk cross-system push isn't confined to the current
            # iteration's weight bucket. Sticky on this system when the
            # previous walk left more uncovered to do. Held back until
            # the system has a fitted timing model: without one we can't
            # cap the (often slow, recurrent) global top confs this walk
            # pulls, so a fresh system would run them at full length.
            # The neighbor walk seeds the timing model first; once it's
            # fit, this turns on and catches up on the global tops.
            if self._timing_ready(system) and (
                self._coverage_flag.get(system, False)
                or random.random() < _COVERAGE_PROB
            ):
                conf = self._select_uncovered_top(system)
                if conf is not None:
                    covered, total = self._coverage_stats.get(
                        system, (None, None))
                    return self._result(
                        conf, 'coverage_walk', system, covered, total)

            t = self._select_time()
            max_weights = self._select_max_weights(t, system)

            picked, _ = random.choices(
                _STRATEGY_PROBS,
                weights=[w for _, w in _STRATEGY_PROBS],
                k=1,
            )[0]
            conf = self._run_strategy(picked, t, max_weights, system)
            if conf is not None:
                return self._result(conf, picked, system)

            # First-line strategy was unavailable -- always fall back
            # to the neighbor walk.
            if picked != 'neighbor':
                conf = self._run_strategy(
                    'neighbor', t, max_weights, system)
                if conf is not None:
                    return self._result(conf, 'neighbor', system)

            conf = self._select_default(system)
            if conf is not None:
                return self._result(conf, 'default', system)
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
    ):
        super().__init__()
        # Search and the DbReader are built lazily on `run()` so the
        # reader is opened on the thread that uses it; that lets
        # `DbReader` enforce `check_same_thread=True`.
        self._path = path
        self._template = template
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
        )
        logging.info("Started search thread")
        try:
            while True:
                m = self.requests_queue.get()
                match m:
                    case Select(system=system):
                        result = self.search.select_conf(system)
                        with self.confs_by_system_lock:
                            self.confs_by_system[system].put(result)
                    case SetTemplate(
                        template=template,
                        init_conf=init_conf,
                        train_time=train_time,
                    ):
                        self.search.template = template
                        self.search.init_conf = init_conf
                        self.search.train_time = train_time
                        self.max_time = train_time[1]
                        logging.info(
                            f'Search thread: new template {template}')
                    case Stop():
                        logging.info("Stopping search thread")
                        break
                    case _:
                        assert False, f"Unknown message: {m!r}"
        finally:
            reader.close()
