import logging
import math
import random
from math import log2
from typing import Optional

import numpy as np

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration import Configuration, Template, conf_neighbors
from .model import build_model_def
from .resultdb import ConfScore, ConfWithRuns, ResultDB
from .run import Run

# Probability of running the predicted-best strategy before falling
# back to the others. Requires both loss and timing models to be ready.
_PREDICTED_BEST_PROB = 0.15

# Probability of running the hill-climb strategy. Requires both loss
# and timing models to be ready.
_HILL_CLIMB_PROB = 0.15

# Placeholder value for the `steps` field when deduplicating candidate
# configurations — we replace it with a budget-fitting value later.
_CANDIDATE_STEPS = 1024

# Probability of choosing the time-budget strategy over the top-neighbor
# one on each select_conf call. The fallback on None still kicks in.
_TIME_BUDGET_PROB = 0.3

# Hill-climb tunables and hard limits.
# Temperature is in log2(loss) units. A 0.05 gap gives a ~2.7x
# probability ratio — sharp but not greedy.
_HC_TEMPERATURE = 0.03
_HC_MAX_ITER = 40
_HC_MAX_LAYERS = 8
_HC_MAX_BATCH = 2 ** 16
_HC_MAX_LENGTH = 2 ** 16
_HC_MIN_DECAY = 2 ** -16
_HC_MIN_LR = 2 ** -16
_HC_MAX_LR = 4.0

# Inputs, ordered by size. Hill-climb picks from among those where
# `input|mgru.2` still fits the weight budget so the walk has room
# to grow before hitting the cap.
_HC_INPUTS = ('bits.1+bp', 'bits.2.oh+bp', 'bits.4.oh+bp', 'bytes')

# Plausible first hidden layers to randomize the trivial start.
_HC_FIRST_LAYERS = (
    'dense.1.tanh', 'rnn.1.tanh',
    'gru.1', 'mgru.1', 'mingru.1', 'lstm.1',
)


def top_confs_report(
    confs: list[ConfScore], max_weights: int, max_time: Optional[float], system: Optional[str]
) -> str:
    sys = "" if system is None else f" ({system})"
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    lines = [f"Top confs W ≤ {itoa3(max_weights)}{t}{sys}:"]
    for c in confs:
        t = '   ?    ' if c.median_time is None else f'{c.median_time:7.3f}s'
        score_runs = f'{c.median_score:.4f} ({c.num_runs})'
        lines.append(f'    {score_runs:<11} {t}  {c.conf.aligned_str()}')
    return "\n".join(lines)


def _predicted_best_report(
    seed: ConfScore,
    candidate_data: list[tuple[float, int, Configuration]],
    system: str,
    max_weights: int,
    max_time: float,
) -> str:
    """Log the seed (with its known median loss for comparison) and
    the top 9 candidates — we only look at the top 9 during the
    run-limit walk."""
    seed_score = f'{seed.median_score:.4f} ({seed.num_runs})'
    lines = [
        f"Predicted-best confs W <= {itoa3(max_weights)} "
        f"T <= {ttoa3(max_time)} ({system}):",
        f'    {seed_score:<12}  {seed.conf.aligned_str()}  [seed]',
    ]
    for compound, total_runs, c in candidate_data[:9]:
        score = f'{2.0 ** compound:.4f} ({total_runs})'
        lines.append(f'    {score:<12}  {c.aligned_str()}')
    return "\n".join(lines)


def _hill_climb_report(
    path: list[tuple[float, Configuration]],
    system: str,
    max_weights: int,
    max_time: float,
) -> str:
    """Log the hill-climb path: start -> intermediate steps -> end."""
    lines = [
        f"Hill-climb path W <= {itoa3(max_weights)} "
        f"T <= {ttoa3(max_time)} ({system}, {len(path) - 1} steps):"
    ]
    for i, (pred, c) in enumerate(path):
        score = f'{2.0 ** pred:.4f}'
        lines.append(f'    [{i}] {score:<8}  {c.aligned_str()}')
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
        db: ResultDB,
        template: Template,
        init_conf: Configuration,
        train_time: tuple[float, float],
    ):
        assert isinstance(db, ResultDB)
        self._db = db

        assert isinstance(template, Template)
        self.template = template

        assert isinstance(init_conf, Configuration)
        self._init_conf = init_conf

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self.train_time = train_time

        # Models populated by ModelTrainingThread via the search queue.
        # None until first training completes.
        self.loss_model = None
        self.timing_model = None

    def add_run(
        self,
        conf: Configuration,
        run: Run,
    ):
        assert isinstance(conf, Configuration)
        assert isinstance(run, Run)

        self._db.add_run(conf, run)

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

    def _adjust_steps_to_budget(
        self, conf: Configuration, per_step: float, t: float,
    ) -> Optional[Configuration]:
        """Return `conf` with steps = largest pow2 fitting budget, or None."""
        # (S - 1) * per_step <= t -> S <= t/per_step + 1
        max_steps = int(t / per_step) + 1
        if max_steps < 2:
            return None
        adjusted = 1 << int(math.floor(math.log2(max_steps)))
        if adjusted < 2:
            return None
        if adjusted == conf.steps:
            return conf
        return conf.replace(steps=adjusted)

    def _select_predicted_best(
        self, t: float, max_weights: int, system: str
    ) -> Optional[Configuration]:
        """Predictor-guided tournament.

        Take the best-known conf for (system, template, max_weights,
        max_time<=t) as a seed, expand its neighbors-of-neighbors,
        adjust each candidate's steps to fit the time budget, score
        them with a compound median(predicted_loss, run_losses...),
        then walk the top 9 across the [1] / [2,1,1] / [3,2,2,1x6]
        run-limit sequences.
        """
        if self.loss_model is None or self.timing_model is None:
            return None
        with latency.timer('Search._select_predicted_best'):
            return self._select_predicted_best_impl(t, max_weights, system)

    def _select_predicted_best_impl(
        self, t: float, max_weights: int, system: str
    ) -> Optional[Configuration]:
        try:
            seed = next(self._db.top_confs_for_system(
                system=system, template=self.template,
                max_weights=max_weights, max_time=t, limit=1,
            ))
        except StopIteration:
            return None
        seed_conf = seed.conf

        # BFS-2 from the seed, including seed itself. Normalize steps
        # to a constant before inserting into the set — otherwise
        # confs that differ only in steps end up as different entries,
        # then collapse to the same thing after _adjust_steps_to_budget.
        def _norm(c: Configuration) -> Configuration:
            return c if c.steps == _CANDIDATE_STEPS else c.replace(
                steps=_CANDIDATE_STEPS)

        candidates: set[Configuration] = {_norm(seed_conf)}
        n1 = conf_neighbors(seed_conf, self.template)
        candidates.update(_norm(n) for n in n1)
        for n in n1:
            for nn in conf_neighbors(n, self.template):
                candidates.add(_norm(nn))

        # Filter by weight budget *after* the BFS, so intermediate
        # neighbors with too many weights can still lead us to
        # smaller-but-different final candidates at depth 2.
        candidates = {c for c in candidates if c.num_weights <= max_weights}
        if not candidates:
            return None

        # Predict per-step time for each and adjust steps.
        confs_in = list(candidates)
        per_steps = self.timing_model.predict_batch(system, confs_in)
        if per_steps is None:
            return None  # no timing model for (system, precision)
        adjusted = []
        for c, ps in zip(confs_in, per_steps):
            ac = self._adjust_steps_to_budget(c, float(ps), t)
            if ac is not None:
                adjusted.append(ac)
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
            seed, candidate_data, system, max_weights, t))

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
                        f'compound={compound:.4f})'
                    )
                    return c
        return None

    def _pick_trivial_start(
        self, max_weights: int,
    ) -> Optional[Configuration]:
        """Pick a trivial starting conf for the hill-climb.

        Randomize over: inputs where `input|mgru.2` still fits the
        weight budget (so there's room to grow), and a small set of
        plausible first hidden layers. Metaparameters are hardcoded
        to values that train reasonably across shapes; steps are
        filled in by the caller based on the time budget.
        """
        if not self.template.precision:
            return None
        precision = self.template.precision[0]
        fits = [
            inp for inp in _HC_INPUTS
            if build_model_def(
                f'{inp}|mgru.2', precision=precision
            ).num_weights <= max_weights
        ]
        if not fits:
            fits = ['bits.1+bp']
        inp = random.choice(fits)
        layer = random.choice(_HC_FIRST_LAYERS)
        model = build_model_def(f'{inp}|{layer}', precision=precision)
        return Configuration(
            model=model,
            batch=16, length=256, lr=1.0 / 16, decay=0.5, steps=256,
        )

    def _within_hc_limits(self, conf: Configuration) -> bool:
        if len(conf.model.layers) > _HC_MAX_LAYERS:
            return False
        if conf.batch > _HC_MAX_BATCH or conf.length > _HC_MAX_LENGTH:
            return False
        if conf.decay < _HC_MIN_DECAY:
            return False
        if conf.lr < _HC_MIN_LR or conf.lr > _HC_MAX_LR:
            return False
        return True

    def _select_hill_climb(
        self, t: float, max_weights: int, system: str,
    ) -> Optional[Configuration]:
        """Predictor-guided stochastic hill-climb from a trivial seed.

        Start from a randomly-chosen trivial conf (input + one hidden
        layer + hardcoded metaparams), then step to a predicted-better
        neighbor each iteration. Next step is sampled via softmax over
        the predicted losses with temperature `_HC_TEMPERATURE` — gives
        some diversity without ignoring the predictor. Termination:
        best neighbor's predicted loss >= current's, or `_HC_MAX_ITER`
        steps.
        """
        if self.loss_model is None or self.timing_model is None:
            return None
        with latency.timer('Search._select_hill_climb'):
            return self._select_hill_climb_impl(t, max_weights, system)

    def _select_hill_climb_impl(
        self, t: float, max_weights: int, system: str,
    ) -> Optional[Configuration]:
        start = self._pick_trivial_start(max_weights)
        if start is None:
            return None
        per_step = self.timing_model.predict(system, start)
        if per_step is None:
            return None
        current = self._adjust_steps_to_budget(start, per_step, t)
        if current is None:
            return None

        path: list[tuple[float, Configuration]] = [
            (float(self.loss_model.predict([current])[0]), current),
        ]
        for _ in range(_HC_MAX_ITER):
            raw_neighbors = conf_neighbors(current, self.template)
            # Filter: max_weights, hill-climb global limits.
            valid = [
                n for n in raw_neighbors
                if n.num_weights <= max_weights and self._within_hc_limits(n)
            ]
            if not valid:
                break
            per_steps = self.timing_model.predict_batch(system, valid)
            if per_steps is None:
                break
            adjusted_n = []
            for n, ps in zip(valid, per_steps):
                a = self._adjust_steps_to_budget(n, float(ps), t)
                # Skip neighbors that collapse onto `current` after the
                # steps snap — otherwise the walk can stall stepping in
                # place.
                if a is not None and a != current:
                    adjusted_n.append(a)
            if not adjusted_n:
                break
            preds = np.asarray(self.loss_model.predict(adjusted_n))
            current_pred = float(self.loss_model.predict([current])[0])
            best = float(preds.min())
            if best >= current_pred:
                break  # local minimum in predicted loss
            # Softmax sample (numerically stable).
            shifted = preds - preds.min()
            probs = np.exp(-shifted / _HC_TEMPERATURE)
            probs /= probs.sum()
            choice = np.random.choice(len(adjusted_n), p=probs)
            current = adjusted_n[int(choice)]
            path.append((float(preds[int(choice)]), current))

        logging.info(_hill_climb_report(path, system, max_weights, t))
        return current

    def select_conf(self, system: str) -> Optional[Configuration]:
        """Select a configuration to run, or None if nothing matches.

        Four strategies, tried in order of preference with fallback:
        * predicted-best (needs both loss and timing models) —
          _PREDICTED_BEST_PROB of the time;
        * hill-climb from a trivial start, predictor-guided stochastic
          walk — _HILL_CLIMB_PROB of the time, same model requirements;
        * time-budget — _TIME_BUDGET_PROB of the time;
        * neighbor-walk — the always-available fallback.
        """
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t, system)

            if random.random() < _PREDICTED_BEST_PROB:
                conf = self._select_predicted_best(t, max_weights, system)
                if conf is not None:
                    return conf

            if random.random() < _HILL_CLIMB_PROB:
                conf = self._select_hill_climb(t, max_weights, system)
                if conf is not None:
                    return conf

            if random.random() < _TIME_BUDGET_PROB:
                conf = self._select_time_budget(t, max_weights, system)
                if conf is not None:
                    return conf

            conf = self._select_top_neighbor(t, max_weights, system)
            if conf is not None:
                logging.info(f'Conf for {system}: {conf}')
                return conf

            # Template can change on the fly — re-check init_conf still
            # matches before falling back to it.
            if not self.template.match(self._init_conf):
                logging.info(
                    f'No conf for {system}: init_conf does not match template')
                return None

            # Skip init_conf if it's already been explored enough.
            conf_id = self._db.get_conf_id(self._init_conf)
            if conf_id is not None:
                counts = self._db.get_run_counts([conf_id], system=system)
                total_runs, system_runs = counts.get(conf_id, (0, 0))
                if total_runs >= 7 and system_runs >= 1:
                    logging.info(
                        f'No conf for {system}: init_conf has {total_runs} '
                        f'total runs ({system_runs} on this system)')
                    return None

            logging.info(
                f'Conf for {system}: {self._init_conf} (default)')
            return self._init_conf
