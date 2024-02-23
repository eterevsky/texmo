import logging
import random
from math import log2, sqrt
from typing import Optional

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration2 import Configuration2, Template, conf_neighbors
from .model2 import Weights
from .resultdb import ResultDB
from .results import ResultSet
from .run import Run

class Search(object):
    """Keep track of the configurations and selects what to try next."""

    def __init__(
        self,
        system: str,
        db: ResultDB,
        template: Template,
        init_conf: Configuration2,
        train_time: tuple[float, float],
    ):
        assert isinstance(system, str)
        self._system = system

        assert isinstance(db, ResultDB)
        self._db = db

        assert isinstance(template, Template)
        self._template = template

        assert isinstance(init_conf, Configuration2)
        self._init_conf = init_conf

        assert isinstance(train_time[0], float)
        assert isinstance(train_time[1], float)
        self._train_time = train_time

    def add_run(
        self,
        conf: Configuration2,
        run: Run,
    ):
        assert isinstance(conf, Configuration2)
        assert isinstance(run, Run)
        assert self._template.match(conf)

        self._db.add_run(conf, run)

    def _select_time(self) -> float:
        tmin, tmax = self._train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t):
        with latency.timer("Search._select_max_weights"):
            top_conf_results = self._result_set.top_conf(t)
            if top_conf_results is None:
                return self._template.max_weights.min
            maxw = 8 * top_conf_results.conf.model.weights
            if self._template.max_weights.min >= maxw:
                return self._template.max_weights.min
            if self._template.max_weights.max <= maxw:
                maxw = self._template.max_weights.max
            l = random.uniform(log2(self._template.max_weights.min), log2(maxw))
            return int(2**l)

    def _select_by_neighbors_score(self, t: float, max_weights: int):
        with latency.timer("Search._select_by_neighbors_score"):
            confs = [(None, 0)]
            for conf_results in self._result_set.top_by_neighbors_score(t, max_weights):
                i = len(confs)
                nruns = min(len(conf_results.runs), conf_results.ntimes(self._system))
                confs.append((conf_results.conf, nruns))
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

    def print_top_confs(self, t, max_weights):
        with latency.timer("Search.print_top_confs"):
            report = [f"T ≤ {ttoa3(t)}  W ≤ {itoa3(max_weights)}  by neighbors score"]

            for conf_results in self._result_set.top_by_neighbors_score(t, max_weights, limit=20):
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

    def print_top_median(self, t, max_weights):
        with latency.timer("Search.print_top_median"):
            report = [f"T ≤ {ttoa3(t)}  W ≤ {itoa3(max_weights)}  by median score"]

            for conf_results in self._result_set.top_confs(max_time=t, max_weights=max_weights, limit=20):
                num_runs = len(conf_results.runs)
                assert conf_results.median_score is not None
                score = f" {conf_results.median_score:.4f} ({num_runs})"
                if num_runs < 10:
                    score += " "
                conf = conf_results.conf
                report.append(
                    f"{score}  LEN{conf.length:4}  B{conf.batch:4}  LR{conf.lr:7.4f}  S{conf.steps:4}  {conf.model}"
                )
            report.append("")
            logging.info("\n".join(report))

    def _select_untimed(self, max_weights: int):
        confs = list(self._result_set.top_confs_any_t(max_weights, 10))
        if all(c.median_time(self._system) is not None for c in confs):
            return None
        print(f"Top confs W ≤ {itoa3(max_weights)}:")
        for c in confs:
            t = c.median_time(self._system)
            if t is not None:
                t = ttoa3(t)
            else:
                t = "?     "
            print(f"{c.median_score:.4f}  {t}  {c.conf}")
        for c in confs:
            if c.median_time(self._system) is None:
                return c.conf

    def _select_median_neighbor(self, t: float, max_weights: int) -> Optional[Configuration2]:
        with latency.timer("Search._select_median_neighbor"):
            confs = []
            for conf_results in self._result_set.top_confs(max_time=t, max_weights=max_weights, limit=100):
                
                i = len(confs)
                nruns = len(conf_results.runs)
                confs.append((conf_results.conf, nruns))
                expected_runs = 1
                while i > 0:
                    if confs[i][1] < expected_runs:
                        assert self._template.match(confs[i][0])
                        logging.info(f"Selecting conf top #{i-1} by median score.")
                        return confs[i][0]
                    i //= 3
                    expected_runs += 1

                for neighbor in conf_neighbors(conf_results.conf, self._template):
                    neighbor_results = self._result_set.get_conf_results(neighbor)
                    if neighbor_results.median_score is not None:
                        continue

                    logging.info(f"Selected {neighbor} (neighbor of {conf_results.conf})")
                    return neighbor

    def select_conf(self) -> Configuration2:
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t)

            conf = self._select_untimed(max_weights)
            if conf is not None:
                logging.info(f"Selecting untimed conf")
                return conf

            if random.choice([True, False]):
                self.print_top_median(t, max_weights)
                conf = self._select_median_neighbor(t, max_weights)
                if conf is not None:
                    return conf

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
