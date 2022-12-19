import logging
import math
from math import log2
import random

from . import latency
from .common import INF
from .configuration import conf_neighbors, conf_to_string
from .results import ResultSet
from .predict import Predictor


# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6


class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(self, db, template, init_conf, min_max_weights):
        self._db = db
        self._template = template
        self._init_conf = init_conf
        self._min_max_weights = min_max_weights

        logging.info("Creaing ResultSet.")
        self._result_set = ResultSet(db, template)
        self._last_predictor_update = 0

        logging.info("Creating Predictor")
        self._predictor = Predictor(self._result_set)
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
        self._last_predictor_update = self._result_set.total_runs_count()

    def add_record(self, record, step_loss):
        with latency.timer("Search.add_record"):
            if abs(log2(record.planned_time_s) - log2(record.train_time_s)) > 0.1:
                print("Bad training time, skipping")
                record.loss = INF

            conf_results, loss = self._result_set.add_record(record, step_loss)
            affected_confs = set()
            for neighbor in self._predictor.add_sample(conf_results, loss):
                if self._template.match_conf(neighbor):
                    affected_confs.add(neighbor)

            for neighbor in conf_neighbors(conf_results.conf, self._template):
                affected_confs.add(neighbor)

            affected_confs = list(affected_confs)

            total_runs = self._result_set.total_runs_count()
            if (
                total_runs > 0
                and total_runs
                > self._last_predictor_update
                + self._last_predictor_update ** (1 / 3)
            ):
                self.train_predictor()

            pred_losses = self._predictor.predict(affected_confs)
            self._result_set.update_pred_scores(affected_confs, pred_losses)

    def _select_time(self):
        with latency.timer("Search._select_time"):
            lo, hi = self._template.t
            assert 1 <= lo <= hi
            if lo == hi:
                return lo

            times = [lo]
            weights = [1]

            runs_count = self._result_set.runs_count_per_t()
            runs = runs_count.get(lo, 0)
            print(f"t = {lo:3}  complete = {runs:5}")

            prev_runs = runs

            t = 2 * lo
            while t <= hi:
                times.append(t)
                weights.append(0.6 * weights[-1])

                runs = runs_count.get(t, 0)
                ratio = runs / (prev_runs + 1)
                print(f"t = {t:3}  complete = {runs:5}  ratio = {ratio:.2f}")
                prev_runs = runs
                t *= 2

            return random.choices(times, weights)[0]

    def _select_time_deterministic(self):
        with latency.timer("Search._select_time"):
            lo, hi = self._template.t
            assert 1 <= lo <= hi
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
            top_conf_results = self._result_set.top_conf(t)
            if top_conf_results is None:
                return self._min_max_weights
            maxw = top_conf_results.conf.model.weights
            if self._min_max_weights >= maxw * 8:
                return self._min_max_weights
            l = random.uniform(
                math.log2(self._min_max_weights), math.log2(maxw) + 3
            )
            return int(2**l)

    def _select_by_pred_score(self, t: int, max_weights: int):
        with latency.timer("Search._select_by_pred_score"):
            confs = [(None, 0)]
            for conf_results in self._result_set.top_pred_confs(t, max_weights):
                i = len(confs)
                confs.append((conf_results.conf, len(conf_results.runs)))
                expected_runs = 1
                while i > 0:
                    if confs[i][1] < expected_runs:
                        return i - 1, confs[i][0]
                    i //= 3
                    expected_runs += 1
        return None, None

    def _select_neighbor(self, t: int, max_weights: int):
        """Select a neighbor of a top-scoring conf."""
        with latency.timer("Search._select_neighbor"):
            for i, conf_results in enumerate(
                self._result_set.top_confs(t, max_weights)
            ):
                conf = conf_results.conf
                neighbors = list(conf_neighbors(conf, self._template))
                if neighbors is not None:
                    random.shuffle(neighbors)
                    for neighbor in neighbors:
                        assert neighbor is not None
                        assert neighbor != conf
                        if not self._result_set.has_runs(neighbor):
                            return i, neighbor, conf
        return None, None, None

    def print_top_confs(self, t, max_weights):
        with latency.timer("Search.print_top_confs"):
            print(f"\nT = {t}  W ≤ {max_weights}\n")

            for i, conf_results in enumerate(
                self._result_set.top_pred_confs(t, max_weights, limit=20)
            ):
                if i >= 20:
                    break

                num_runs = len(conf_results.runs)
                score = (
                    f"{conf_results.median_score:.4f} ({num_runs})"
                    if conf_results.median_score
                    else "          "
                )
                if num_runs < 10:
                    score += " "
                extras = ""
                conf = conf_results.conf
                if conf.sample_len != 128:
                    extras += f"  LEN {conf.sample_len}"
                print(
                    f"{score} LR{conf.lr:6.3f}  B{conf.batch:4}  {conf.model}{extras}"
                )
            print()

    def select_conf(self):
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            if self._template.max_weights is None:
                max_weights = self._select_max_weights(t)
            else:
                max_weights = self._template.max_weights

            self.print_top_confs(t, max_weights)

            ipred, pred_conf = self._select_by_pred_score(t, max_weights)
            ineighbor, neighbor_conf, parent_conf = self._select_neighbor(t, max_weights)

            if ipred is not None and (ineighbor is None or ineighbor >= ipred):
                logging.info(f"Selecting conf top #{ipred} by predicted score.")
                return pred_conf

            if ineighbor is not None and (ipred is None or random.randrange(2)):
                c = conf_to_string(parent_conf)
                logging.info(f"Selecting a neighbor or conf #{ineighbor} by median score: {c}")
                return neighbor_conf
            elif ipred is not None:
                logging.info(f"Selecting conf top #{ipred} by predicted score.")
                return pred_conf

            logging.info("Selecting default configuration")
            return self._init_conf._replace(t=t)
