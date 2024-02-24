import logging
import random
from math import log2, sqrt
from typing import Optional

from . import latency
from .common import INF, itoa3, ttoa3
from .configuration2 import Configuration2, Template, conf_neighbors
from .resultdb import ResultDB, ConfScore
from .run import Run


def top_confs_report(
    confs: list[ConfScore], max_weights: int, max_time: float | None
) -> str:
    t = "" if max_time is None else f" T ≤ {ttoa3(max_time)}"
    lines = [f"Top confs W ≤ {itoa3(max_weights)}{t}:"]
    for c in confs:
        t = "?       " if c.median_time is None else "{:8}".format(ttoa3(c.median_time))
        lines.append(
            f"{c.median_score:.4f} ({c.num_runs})  {t}  {c.conf.aligned_str()}"
        )
    return "\n".join(lines)


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

        self._db.add_run(conf, run, update_neighbors=False)

    def _select_time(self) -> float:
        tmin, tmax = self._train_time
        if tmin == tmax:
            return tmin
        assert 0 < tmin < tmax
        return 2 ** random.uniform(log2(tmin), log2(tmax))

    def _select_max_weights(self, t):
        with latency.timer("Search._select_max_weights"):
            try:
                top_conf = next(
                    self._db.top_confs(max_time=t, system=self._system, limit=1)
                )
            except StopIteration:
                return self._template.max_weights.min
            maxw = 8 * top_conf.conf.model.weights
            if self._template.max_weights.min >= maxw:
                return self._template.max_weights.min
            if self._template.max_weights.max <= maxw:
                maxw = self._template.max_weights.max
            l = random.uniform(log2(self._template.max_weights.min), log2(maxw))
            return round(2**l)

    # def print_top_median(self, t, max_weights):
    # with latency.timer("Search.print_top_median"):
    #     report = [f"T ≤ {ttoa3(t)}  W ≤ {itoa3(max_weights)}  by median score"]

    #     for res in self._db.top_confs_with_time(max_time=t, max_weights=max_weights, system=self._system, limit=20):
    #         num_runs = res["num_runs"]
    #         median_score = res["median_score"]
    #         assert median_score is not None
    #         score = f" {median_score:.4f} ({num_runs})"
    #         if num_runs < 10:
    #             score += " "
    #         conf = res["conf"]
    #         report.append(
    #             f"{score}  LEN{conf.length:4}  B{conf.batch:4}  LR{conf.lr:7.4f}  S{conf.steps:4}  {conf.model}"
    #         )
    #     report.append("")
    #     logging.info("\n".join(report))

    def _select_untimed(self, max_weights: int):
        with latency.timer("Search._select_untimed"):
            confs = list(
                self._db.top_confs(
                    max_weights=max_weights, system=self._system, limit=10
                )
            )
            if all(c.median_time is not None for c in confs):
                return None
            logging.info(top_confs_report(confs=confs, max_weights=max_weights, max_time=None))
            for i, c in enumerate(confs):
                if c.median_time is None:
                    logging.info(f"Selecting conf #{i}: {c.conf}")
                    return c.conf

    def _select_neighbor_fewest_runs(self, conf_id: int) -> Optional[ConfScore]:
        for neighbor_cs in self._db.get_neighbors_by_runs(conf_id, self._system):
            if self._template.match(neighbor_cs.conf):
                return neighbor_cs
        return None

    def _select_median_neighbor(
        self, t: float, max_weights: int
    ) -> Optional[Configuration2]:
        """
        Expected number of runs based on position (configuration, neighbors):

        (2, 0)
        (2, 1)
        (2, 1) (2, 0) (2, 0)
        (2, 1) (2, 1) (2, 1)
        (3, 1) (2, 1) (2, 1)
        (3, 2) (2, 1) (2, 1)
        """
        with latency.timer("Search._select_median_neighbor"):
            confs = list(
                self._db.top_confs(
                    max_time=t, max_weights=max_weights, system=self._system, limit=10
                )
            )
            logging.info(
                top_confs_report(confs=confs, max_weights=max_weights, max_time=t)
            )

            top_cs = confs[0]
            if top_cs.num_runs < 2:
                logging.info(f"Selecting conf #0: {top_cs.conf}")
                return top_cs.conf

            neighbor_cs = self._select_neighbor_fewest_runs(top_cs.conf_id)
            if neighbor_cs is not None and neighbor_cs.num_runs == 0:
                logging.info(f"Selecting neighbor of top conf: {neighbor_cs.conf}")
                return neighbor_cs.conf

            for i in range(1, 3):
                cs = confs[i]
                if cs.num_runs < 2:
                    logging.info(f"Selecting conf #{i}: {cs.conf}")
                    return cs.conf

            for i in range(1, 3):
                cs = confs[i]
                neighbor_cs = self._select_neighbor_fewest_runs(cs.conf_id)
                if neighbor_cs is not None and neighbor_cs.num_runs == 0:
                    logging.info(f"Selecting neighbor of conf #{i}: {neighbor_cs.conf}")
                    return neighbor_cs.conf

            top_cs = confs[0]
            if top_cs.num_runs < 3:
                logging.info(f"Selecting conf #0: {top_cs.conf}")
                return top_cs.conf

            neighbor_cs = self._select_neighbor_fewest_runs(top_cs.conf_id)
            if neighbor_cs is not None and neighbor_cs.num_runs < 2:
                logging.info(f"Selecting neighbor of top conf: {neighbor_cs.conf}")
                return neighbor_cs.conf

            # raise NotImplementedError()
            return None

    def select_conf(self) -> Configuration2:
        with latency.timer("Search.select_conf"):
            t = self._select_time()
            max_weights = self._select_max_weights(t)

            conf = self._select_untimed(max_weights)
            if conf is not None:
                return conf

            conf = self._select_median_neighbor(t, max_weights)
            if conf is not None:
                return conf

            logging.info("Selecting default configuration")
            return self._init_conf
