import logging
import random
from math import log2, sqrt
from typing import Optional

from . import latency
from .common import INF, ttoa3, itoa3
from .configuration2 import Configuration2, conf_neighbors, Template
from .model2 import Weights
from .predict import Predictor2
from .pretrained import Checkpoint
from .record import TrainingRecord
from .results import ResultSet
from .resultdb import ResultDB
from .run import Run

# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6


class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(
        self,
        system: str,
        db: ResultDB,
        template: Template,
        init_conf: Configuration2,
        min_max_weights: int,
        predictor: Predictor2,
        train_time: tuple[float, float],
        checkpoints_path: str = None,
    ):
        assert isinstance(system, str)
        self._system = system

        assert isinstance(db, ResultDB)
        self._db = db

        assert isinstance(template, Template)
        self._template = template

        assert isinstance(init_conf, Configuration2)
        self._init_conf = init_conf

        self._min_max_weights = min_max_weights
        self._checkpoints_path = checkpoints_path

        logging.info("Creating ResultSet.")
        self._result_set = ResultSet(
            system=system, result_db=db, template=template, populate_neighbors=True
        )
        # self._last_predictor_update = 0

        # self._predictor = LossPredictorV1(self._result_set, split_test_set=True)
        self._predictor = predictor
        if self._predictor is not None and self._predictor.maybe_train():
            self._update_pred_scores()

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self._train_time = train_time

    def _update_pred_scores(self):
        logging.info("Generating predicted losses for all confs")
        all_confs = list(self._result_set.get_confs())
        pred_losses = self._predictor.predict(all_confs, verbose=True)
        logging.info("Populating predicted losses")
        self._result_set.update_pred_scores(all_confs, pred_losses)

    def add_run(
        self,
        conf: Configuration2,
        run: Run,
        weights: Weights,
        parent_checkpoint: Optional[Checkpoint] = None,
    ):
        assert isinstance(conf, Configuration2)
        assert isinstance(run, Run)
        assert self._template.match(conf)

        with latency.timer("Search.add_run"):
            # if parent_checkpoint is None:
            #     checkpoint = Checkpoint(conf, run)
            # else:
            #     checkpoint = parent_checkpoint.clone()
            #     checkpoint.add_run(conf, run)

            # if (
            #     self._checkpoints_path
            #     and run.loss < 4.5
            #     and run.loss < self._db.best_checkpoint_loss(conf.model)
            # ):
            #     checkpoint.save(weights, self._checkpoints_path)

            if self._predictor:
                self._predictor.add_run(conf, run)
            self._result_set.add_run(conf, run)

            # self._predictor.update_conf_results(conf_results)
            # affected_confs = set()
            # for neighbor in self._predictor.update_conf_results(conf_results):
            #     if self._template.match_conf(neighbor):
            #         affected_confs.add(neighbor)

            # for neighbor in conf_neighbors(conf_results.conf, self._template):
            #     affected_confs.add(neighbor)

            # affected_confs = list(affected_confs)

            if self._predictor and self._predictor.maybe_train():
                self._update_pred_scores()

            # total_runs = self._result_set.total_runs_count()
            # if (
            #     total_runs > 0
            #     and total_runs
            #     > self._last_predictor_update
            #     + self._last_predictor_update ** (1 / 3)
            # ):
            #     self.train_predictor()

            # pred_losses = self._predictor.predict(affected_confs)
            # self._result_set.update_pred_scores(affected_confs, pred_losses)

    def _select_time(self):
        tmin, tmax = self._train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    # def _select_time(self):
    #     with latency.timer("Search._select_time"):
    #         lo, hi = self._template.t
    #         assert 1 <= lo <= hi
    #         if lo == hi:
    #             return lo

    #         times = [lo]
    #         weights = [1]

    #         runs_count = self._result_set.runs_count_per_t()
    #         runs = runs_count.get(lo, 0)
    #         logging.info(f"t = {lo:3}  complete = {runs:5}")

    #         prev_runs = runs

    #         t = 2 * lo
    #         while t <= hi:
    #             times.append(t)
    #             weights.append(0.6 * weights[-1])

    #             runs = runs_count.get(t, 0)
    #             ratio = runs / (prev_runs + 1)
    #             logging.info(f"t = {t:3}  complete = {runs:5}  ratio = {ratio:.2f}")
    #             prev_runs = runs
    #             t *= 2

    #         return random.choices(times, weights)[0]

    # def _select_time_deterministic(self):
    #     with latency.timer("Search._select_time"):
    #         lo, hi = self._template.t
    #         assert 1 <= lo <= hi
    #         if lo == hi:
    #             return lo

    #         runs_count = self._result_set.runs_count_per_t()
    #         runs = runs_count.get(lo, 0)
    #         logging.info(f"t = {lo:3}  complete = {runs:5}")

    #         if runs == 0:
    #             return lo

    #         min_ratio = RUNS_EXP
    #         best_t = lo
    #         prev_runs = runs

    #         t = 2 * lo
    #         while t <= hi:
    #             runs = runs_count.get(t, 0)
    #             ratio = runs / prev_runs
    #             logging.info(f"t = {t:3}  complete = {runs:5}  ratio = {ratio:.2f}")
    #             if runs == 0:
    #                 return t
    #             if ratio < min_ratio:
    #                 min_ratio = ratio
    #                 best_t = t
    #             prev_runs = runs
    #             t *= 2

    #         return best_t

    def _select_max_weights(self, t):
        with latency.timer("Search._select_max_weights"):
            top_conf_results = self._result_set.top_conf(t)
            if top_conf_results is None:
                return self._min_max_weights
            maxw = 8 * top_conf_results.conf.model.weights
            if self._min_max_weights >= maxw:
                return self._min_max_weights
            l = random.uniform(log2(self._min_max_weights), log2(maxw))
            return int(2**l)

    def _select_by_neighbors_score(self, t: float, max_weights: int):
        with latency.timer("Search._select_by_neighbors_score"):
            confs = [(None, 0)]
            for conf_results in self._result_set.top_by_neighbors_score(t, max_weights):
                i = len(confs)
                confs.append((conf_results.conf, len(conf_results.runs)))
                expected_runs = 1
                while i > 0:
                    if confs[i][1] < expected_runs:
                        assert self._template.match(confs[i][0])
                        logging.info(f"Selecting conf top #{i-1} by neighbors score.")
                        return confs[i][0]
                    i //= 3
                    expected_runs += 1
        return None

    # def _select_by_pred_score(self, t: int, max_weights: int):
    #     with latency.timer("Search._select_by_pred_score"):
    #         confs = [(None, 0)]
    #         for conf_results in self._result_set.top_pred_confs(t, max_weights):
    #             i = len(confs)
    #             confs.append((conf_results.conf, len(conf_results.runs)))
    #             expected_runs = 1
    #             while i > 0:
    #                 if confs[i][1] < expected_runs:
    #                     assert self._template.match_conf(confs[i][0])
    #                     return i - 1, confs[i][0]
    #                 i //= 3
    #                 expected_runs += 1
    #     return None, None

    # def _select_neighbor(self, t: int, max_weights: int):
    #     """Select a neighbor of a top-scoring conf."""
    #     with latency.timer("Search._select_neighbor"):
    #         for i, conf_results in enumerate(
    #             self._result_set.top_confs(t, max_weights)
    #         ):
    #             conf = conf_results.conf
    #             neighbors = list(conf_neighbors(conf, self._template))
    #             random.shuffle(neighbors)
    #             for neighbor in neighbors:
    #                 assert neighbor is not None
    #                 assert neighbor != conf
    #                 if not self._result_set.has_runs(neighbor):
    #                     return i, neighbor, conf
    #     return None, None, None

    def print_top_confs(self, t, max_weights):
        with latency.timer("Search.print_top_confs"):
            report = [f"T ≤ {ttoa3(t)}  W ≤ {itoa3(max_weights)}"]

            for i, conf_results in enumerate(
                self._result_set.top_by_neighbors_score(t, max_weights, limit=20)
            ):
                num_runs = len(conf_results.runs)
                if conf_results.median_score:
                    score = f" {conf_results.median_score:.4f} ({num_runs})"
                    if num_runs < 10:
                        score += " "
                elif conf_results.neighbors_score:
                    score = f"({conf_results.neighbors_score:.4f})    "
                conf = conf_results.conf
                report.append(
                    f"{score}  LEN{conf.length:4}  B{conf.batch:4}  LR{conf.lr:7.4f}  S{conf.steps:4}  {conf.model}"
                )
            report.append("")
            logging.info("\n".join(report))

    # def _select_checkpoint(self, t):
    #     pass

    def select_conf(self) -> Configuration2:
        with latency.timer("Search.select_conf"):
            t = self._select_time()

            if False:
                return self._select_checkpoint(t)

            if self._template.max_weights == INF:
                max_weights = self._select_max_weights(t)
            else:
                max_weights = self._template.max_weights

            self.print_top_confs(t, max_weights)

            conf = self._select_by_neighbors_score(t, max_weights)

            if conf is not None:
                return conf

            # ipred, pred_conf = None, None
            # ipred, pred_conf = self._select_by_pred_score(t, max_weights)
            # ineighbor, neighbor_conf, parent_conf = self._select_neighbor(
            #     t, max_weights
            # )

            # if ipred is not None and (ineighbor is None or ineighbor >= ipred):
            #     logging.info(f"Selecting conf top #{ipred} by predicted score.")
            #     return pred_conf, None

            # if ineighbor is not None and (ipred is None or random.randrange(2)):
            #     c = conf_to_string(parent_conf)
            #     logging.info(
            #         f"Selecting a neighbor or conf #{ineighbor} by median score: {c}"
            #     )
            #     return neighbor_conf, None
            # elif ipred is not None:
            #     logging.info(f"Selecting conf top #{ipred} by predicted score.")
            #     return pred_conf, None

            logging.info("Selecting default configuration")
            return self._init_conf
