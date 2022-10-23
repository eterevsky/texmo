import math
import random

from texmo import latency
from texmo.configuration import conf_neighbors
from texmo.results import ResultSet
from texmo.predict import Predictor


# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6
INF = float("inf")


class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(self, db, template, init_conf, max_weights, min_max_weights):
        self._db = db
        self._template = template
        self._init_conf = init_conf
        self._max_weights = max_weights
        self._min_max_weights = min_max_weights

        print(f"Creaing ResultSet.")
        self._result_set = ResultSet(db, template)
        self._last_predictor_runs = 0

        print("Creating Predictor")
        self._predictor = Predictor(self._result_set.all_conf_runs())
        if self._result_set.total_runs_count() > 0:
            self.train_predictor()

    def train_predictor(self):
        self._predictor.train()
        print("Generating predicted losses for all confs")
        all_confs = list(self._result_set.all_confs())
        pred_losses = self._predictor.predict(all_confs)
        print("Populating predicted losses")
        self._result_set.update_pred_scores(all_confs, pred_losses)
        print("Pred scores ready")
        self._last_predictor_runs = self._result_set.total_runs_count()

    def add_record(self, record):
        with latency.timer("Search.add_record"):
            conf, loss = self._result_set.add_record(record)
            conf = conf._replace(id=None)
            affected_confs = set()
            for neighbor in self._predictor.add_sample(conf, loss):
                if self._template.match_conf(neighbor):
                    affected_confs.add(neighbor)

            for neighbor in conf_neighbors(conf, self._template):
                affected_confs.add(neighbor)

            affected_confs = list(affected_confs)

            if self._result_set.total_runs_count() >= max(
                1, 1.01 * self._last_predictor_runs
            ):
                self.train_predictor()

            pred_losses = self._predictor.predict(affected_confs)
            self._result_set.update_pred_scores(affected_confs, pred_losses)

    def _select_time(self):
        with latency.timer("Search._select_time"):
            lo, hi = self._template.t
            if lo == hi:
                return lo

            runs_count = self._result_set.runs_count_per_t()
            runs = runs_count.get(lo, 0)
            print(f"t = {lo:3}  complete = {runs:5}")

            if runs == 0:
                return lo

            min_ratio = RUNS_EXP
            best_t = lo
            prev_runs = runs

            t = 2 * lo
            while t <= hi:
                runs = runs_count.get(t, 0)
                ratio = runs / prev_runs
                print(f"t = {t:3}  complete = {runs:5}  ratio = {ratio:.2f}")
                if runs == 0:
                    return t
                if ratio < min_ratio:
                    min_ratio = ratio
                    best_t = t
                prev_runs = runs
                t *= 2

            return best_t

    def _select_max_weights(self, t):
        with latency.timer("Search._select_max_weights"):
            top_conf = self._result_set.top_conf(t)
            if top_conf is None:
                return self._min_max_weights
            maxw = top_conf.spec.weights()
            if self._min_max_weights >= maxw * 4:
                return self._min_max_weights
            l = random.uniform(
                math.log2(self._min_max_weights), math.log2(maxw) + 2
            )
            return int(2**l)

    def _select_by_pred_score(self, t: int, max_weights: int):
        with latency.timer("Search._select_by_pred_score"):
            top_weights = max_weights
            for top_conf, _ in self._result_set.top_pred_confs(
                t, max_weights, limit=1
            ):
                top_weights = top_conf.spec.weights()

            with latency.timer("Search._select_by_pred_score.runs_count"):
                nruns = self._result_set.runs_count(
                    t, max_weights, min_weights=top_weights // 2
                )

            # The number of runs per conf depends on the order by pred score.
            # The number of confs with >= n+1 runs is 1/3 of all the confs with
            # >= n runs.
            # This means that confs with the order <= 2 * enr / 3^k should have
            # >= k runs.

            best_conf = None
            best_gap = -1

            k = 16
            for i, (conf, results) in enumerate(
                self._result_set.top_pred_confs(t, max_weights)
            ):
                while i + 1 > 2 * nruns / 3 ** (k - 1) and k > 1:
                    k -= 1
                gap = k - results.num_runs
                if gap > best_gap:
                    best_conf = conf
                    best_gap = gap
                if gap > k:
                    break

            return best_conf

    def print_top_confs(self, t, max_weights):
        with latency.timer("print_top_confs"):
            print(f"\nT = {t}  W ≤ {max_weights}\n")

            for i, (conf, results) in enumerate(
                self._result_set.top_pred_confs(t, max_weights, limit=20)
            ):
                if i >= 20:
                    break
                score = f"{results.score:.4f}" if results.score else "      "
                print(
                    f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
                    + f"B{conf.batch:4}  R{conf.regularization:4}  "
                    + f"I{conf.init_scale:4}  {score} ({results.num_runs})"
                )
            print()

    def select_conf(self):
        with latency.timer("select_conf"):
            t = self._select_time()
            if self._max_weights is None:
                max_weights = self._select_max_weights(t)
            else:
                max_weights = self._max_weights

            self.print_top_confs(t, max_weights)

            conf = self._select_by_pred_score(t, max_weights)
            if conf is not None:
                return conf

            return self._init_conf._replace(t=t)
