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
    label that picked it."""
    conf: Configuration
    strategy: str
    system: str

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "conf": self.conf.to_dict(),
            "strategy": self.strategy,
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

# Probability of running the predicted-best strategy at BFS depth 2
# (~100 candidates) before falling back to others. Requires both loss
# and timing models to be ready.
_PREDICTED_2ND_NEIGHBOR_PROB = 0.15

# Probability of running predicted-best at BFS depth 3 (~1000
# candidates — bigger jump from the seed). Same model requirements.
_PREDICTED_3RD_NEIGHBOR_PROB = 0.2

# Placeholder value for the `steps` field when deduplicating candidate
# configurations — we replace it with a budget-fitting value later.
_CANDIDATE_STEPS = 1024

# Probability of choosing the time-budget strategy over the top-neighbor
# one on each select_conf call. The fallback on None still kicks in.
_TIME_BUDGET_PROB = 0.3


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

    def _select_global_top(
        self, max_weights: int, system: str
    ) -> Optional[Configuration]:
        """Cross-system coverage: pick the global top conf at
        `weights ≤ max_weights`, cap its steps to fit maxt on `system`,
        and propose it if this system hasn't run it yet.

        Without this, the throughput-comparison page is misleading —
        we only know how a top conf performs on the system(s) that
        happened to discover it. Falls through (returns None) once
        every global top is covered, so the regular search strategies
        keep running.
        """
        with latency.timer("Search._select_global_top"):
            top_confs = list(
                self._db.top_confs_global(
                    self.template, max_weights=max_weights))
            if not top_confs:
                return None
            # `top_confs_global` yields a strictly-improving Pareto
            # front; the last one is the global best at this budget.
            best = top_confs[-1]
            capped = self._cap_steps(best.conf, system)
            conf_id = self._db.get_conf_id(capped)
            if conf_id is not None:
                counts = self._db.get_run_counts(
                    [conf_id], system=system)
                _, system_runs = counts.get(conf_id, (0, 0))
                if system_runs > 0:
                    return None
            return capped

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
        # Normalize steps to a constant before inserting into the set
        # — otherwise confs differing only in steps end up as separate
        # entries, then collapse to the same thing after the
        # budget-fitting step. Expand from unnormalized (= seed's
        # actual steps) so neighbor generation matches the seed's
        # parameter grid.
        def _norm(c: Configuration) -> Configuration:
            return c if c.steps == _CANDIDATE_STEPS else c.replace(
                steps=_CANDIDATE_STEPS)

        visited: set[Configuration] = {_norm(seed_conf)}
        frontier: list[Configuration] = [seed_conf]
        for _ in range(bfs_depth):
            next_frontier: list[Configuration] = []
            for c in frontier:
                for n in conf_neighbors(c, self.template):
                    nn = _norm(n)
                    if nn not in visited:
                        visited.add(nn)
                        next_frontier.append(n)
            if not next_frontier:
                break
            frontier = next_frontier

        # Filter by weight budget *after* the BFS, so intermediate
        # neighbors with too many weights can still lead us to
        # smaller-but-different final candidates at depth N.
        candidates = {c for c in visited if c.num_weights <= max_weights}
        if not candidates:
            return None

        # Pick the largest pow2 step count that fits the budget for each.
        # The timing model owns the inversion (init / scan / step
        # decomposition lives inside it). `predict_max_steps` returns
        # None iff no model is fit for (system, precision); since
        # precision is the same across `confs_in`, the first None means
        # the whole strategy is unavailable.
        confs_in = list(candidates)
        adjusted = []
        for c in confs_in:
            max_steps = self.timing_model.predict_max_steps(system, c, t)
            if max_steps is None:
                return None
            if max_steps < 2:
                continue
            if max_steps == c.steps:
                adjusted.append(c)
            else:
                adjusted.append(c.replace(steps=max_steps))
        if not adjusted:
            return None

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

    def select_conf(self, system: str) -> Optional[SearchResult]:
        """Select a SearchResult, or None if nothing matches.

        Strategies, tried in order with fallback:
        * global-top — always tried; cross-system coverage for the
          global top conf at the current weight budget. Falls through
          once every top is covered on this system.
        * predicted-best at BFS depth 2 (needs both loss and timing
          models) — _PREDICTED_2ND_NEIGHBOR_PROB of the time;
        * predicted-best at BFS depth 3 (bigger jump from the seed) —
          _PREDICTED_3RD_NEIGHBOR_PROB of the time, same model
          requirements;
        * time-budget — _TIME_BUDGET_PROB of the time;
        * neighbor-walk — the always-available fallback.
        """
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t, system)

            # Cross-system coverage first: if the global top conf at
            # this weight budget hasn't been run on `system`, do that.
            # Falls through once every global top is covered.
            conf = self._select_global_top(max_weights, system)
            if conf is not None:
                logging.info(f'Conf for {system}: {conf} (global_top)')
                return SearchResult(conf, 'global_top', system)

            if random.random() < _PREDICTED_2ND_NEIGHBOR_PROB:
                conf = self._select_predicted_best(
                    t, max_weights, system, bfs_depth=2)
                if conf is not None:
                    return SearchResult(conf, 'predicted_2nd_neighbor', system)

            if random.random() < _PREDICTED_3RD_NEIGHBOR_PROB:
                conf = self._select_predicted_best(
                    t, max_weights, system, bfs_depth=3)
                if conf is not None:
                    return SearchResult(conf, 'predicted_3rd_neighbor', system)

            if random.random() < _TIME_BUDGET_PROB:
                conf = self._select_time_budget(t, max_weights, system)
                if conf is not None:
                    return SearchResult(conf, 'time_budget', system)

            conf = self._select_top_neighbor(t, max_weights, system)
            if conf is not None:
                logging.info(f'Conf for {system}: {conf}')
                return SearchResult(conf, 'neighbor', system)

            # Template can change on the fly — re-check init_conf still
            # matches before falling back to it.
            if not self.template.match(self.init_conf):
                logging.info(
                    f'No conf for {system}: init_conf does not match template')
                return None

            # Skip init_conf if it's already been explored enough.
            conf_id = self._db.get_conf_id(self.init_conf)
            if conf_id is not None:
                counts = self._db.get_run_counts([conf_id], system=system)
                total_runs, system_runs = counts.get(conf_id, (0, 0))
                if total_runs >= 7 and system_runs >= 1:
                    logging.info(
                        f'No conf for {system}: init_conf has {total_runs} '
                        f'total runs ({system_runs} on this system)')
                    return None

            logging.info(
                f'Conf for {system}: {self.init_conf} (default)')
            return SearchResult(self.init_conf, 'default', system)


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
