import logging
import os
import sqlite3
from collections.abc import Iterable
from typing import Optional
from datetime import datetime

from . import latency
from .common import INF
from .configuration2 import Configuration, Template, conf_neighbors
from .confresults import ConfResults
from .resultdb import ResultDB
from .run import Run


class ResultSet(object):
    def __init__(
        self,
        result_db: Optional[ResultDB],
        template: Template,
        system: str,
        generate_neighbors: bool = True,
    ):
        assert isinstance(template, Template)
        self._template: Template = template

        assert isinstance(system, str)
        self._system: str = system

        # Configurations matching the template
        self._conf_results_by_id: dict[int, ConfResults] = {}
        self._conf_results_by_conf: dict[Configuration, ConfResults] = {}

        self._init_db()

        if result_db is None:
            self._result_db = ResultDB()
        else:
            self._result_db = result_db
            self._import_from_result_db()

        if generate_neighbors:
            logging.info("Generating all neighbors")
            self._update_all_neighbors()
        self._populate_db()

    def _init_db(self):
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        schema_path = os.path.join(os.path.dirname(__file__), "runtime-db.sql")
        with open(schema_path) as schema:
            self._db.executescript(schema.read())

    def _populate_db(self):
        logging.info("Populating runtime DB")
        for conf_results in self._conf_results_by_id.values():
            cur = self._db.execute(
                """
                UPDATE conf
                SET weights = :weights,
                    matches_template = :matches_template,
                    median_time = :median_time,
                    estimated_time = :estimated_time,
                    median_score = :median_score,
                    neighbors_score = :neighbors_score
                WHERE id = :id
                """,
                {
                    "id": conf_results.id,
                    "weights": conf_results.conf.model.weights,
                    "matches_template": self._template.match(conf_results.conf),
                    "median_time": conf_results.median_time(self._system),
                    "estimated_time": conf_results.estimated_time(self._system),
                    "median_score": conf_results.median_score,
                    "neighbors_score": conf_results.neighbors_score,
                },
            )
            assert cur.rowcount == 1
        logging.info(f"Populated runtime DB with {len(self._conf_results_by_id)} confs")

    def _import_from_result_db(self):
        """Import from the persistent DB all matching confs.

        Only the confs that are matching `self._template` are being selected.
        """
        with latency.timer("ResultSet._import_from_result_db"):
            logging.info("Importing relevant configurations and results from ResultDB")
            n = 0
            n_w_template = 0

            for id, conf, run in self._result_db.get_confs_runs():
                n += 1
                if self._template.match(conf):
                    n_w_template += 1
                conf_results = self._find_or_add_conf(conf, id)
                conf_results.add_run(run)

            logging.info(f"Imported {n} runs ({n_w_template} matching template)")
            # logging.info("Populating scores from run results")
            # self.update_all_scores()

    def _find_or_add_conf(self, conf: Configuration, id=None) -> ConfResults:
        """Finds the conf in the db and returns the copy with populated id."""
        with latency.timer("ResultSet._find_or_add_conf"):
            assert isinstance(conf, Configuration)
            conf_results = self._conf_results_by_conf.get(conf)
            if conf_results is not None:
                assert id is None or conf_results.id == id
                return conf_results
            if id is None:
                id = self._result_db.find_or_add_conf(conf, init_neighbors=False)
            conf_results = ConfResults(id, conf)
            self._conf_results_by_id[id] = conf_results
            self._conf_results_by_conf[conf] = conf_results

            matches_template = self._template.match(conf)

            self._db.execute(
                """
                INSERT INTO conf(id, weights, matches_template)
                VALUES(:id, :weights, :matches_template)
                """,
                {
                    "id": id,
                    "weights": conf.model.weights,
                    "matches_template": matches_template,
                },
            )

            return conf_results
    
    def get_conf_results(self, conf: Configuration) -> ConfResults:
        return self._find_or_add_conf(conf)

    def _update_all_neighbors(self):
        with latency.timer("ResultSet._update_all_neighbors"):
            count = 0
            for conf_results in list(self._conf_results_by_id.values()):
                if conf_results.median_score is None or not self._template.match(
                    conf_results.conf
                ):
                    continue
                count += 1
                for neighbor in conf_neighbors(conf_results.conf, self._template):
                    neighbor_results = self._find_or_add_conf(neighbor)
                    for run in conf_results.runs:
                        neighbor_results.add_neighbor_run(conf_results, run)
            logging.info(f"Generated neighbors for {count} configurations")

    def _update_conf_scores(self, conf_results: ConfResults):
        with latency.timer("ResultSet._update_conf_scores"):
            cur = self._db.execute(
                """
                UPDATE conf
                SET median_score = :median_score,
                    neighbors_score = :neighbors_score,
                    median_time = :median_time,
                    estimated_time = :estimated_time
                WHERE id = :id
                """,
                {
                    "id": conf_results.id,
                    "median_score": conf_results.median_score,
                    "neighbors_score": conf_results.neighbors_score,
                    "median_time": conf_results.median_time(self._system),
                    "estimated_time": conf_results.estimated_time(self._system),
                },
            )
            assert cur.rowcount == 1

    def _check_consistency(self):
        """Checks the consistency of ConfResults with the values in the DB."""
        found_ids = set()

        for row in self._db.execute(
            "SELECT id, weights, matches_template, median_score, neighbors_score, median_time, estimated_time FROM conf"
        ):
            id = row[0]
            found_ids.add(id)
            conf_results = self._conf_results_by_id[id]

            assert conf_results.conf.model.weights == row[1]
            assert self._template.match(conf_results.conf) == row[2]
            assert conf_results.median_score == row[3]
            assert conf_results.neighbors_score == row[4]
            assert conf_results.median_time(self._system) == row[5]
            assert conf_results.estimated_time(self._system) == row[6]

        assert len(found_ids) == len(self._conf_results_by_id)

    def get_conf_results(self, conf: Configuration) -> Optional[ConfResults]:
        return self._conf_results_by_conf.get(conf)

    def top_confs_any_t(self, max_weights: int = INF, limit: int = INF) -> Iterable[ConfResults]:
        cur = self._db.execute(
            """
            SELECT id
            FROM conf
            WHERE weights <= ?
              AND matches_template
              AND median_score IS NOT NULL
            ORDER BY median_score
            LIMIT ?
            """,
            (max_weights, limit),
        )
        for row in cur:
            yield self._conf_results_by_id[row[0]]

    def top_by_neighbors_score(
        self, max_time: float, max_weights: float = INF, limit: Optional[int] = None
    ) -> Iterable[ConfResults]:
        with latency.timer("ResultSet.top_by_median_score"):
            limit_str = "" if limit is None else "\nLIMIT :limit"
            cur = self._db.execute(
                """
                SELECT id
                FROM conf
                WHERE neighbors_score IS NOT NULL
                  AND estimated_time <= :max_time
                  AND weights <= :max_weights
                  AND matches_template
                ORDER BY neighbors_score
                """
                + limit_str,
                {"max_time": max_time, "max_weights": max_weights, "limit": limit},
            )
            for row in cur:
                yield self._conf_results_by_id[row[0]]

    def top_conf(self, max_time: float, max_weights: int = INF) -> ConfResults:
        """A configuration with the highest (self) score with time = t."""

        cur = self._db.execute(
            """
            SELECT id FROM conf
            WHERE matches_template
              AND median_time IS NOT NULL
              AND median_time <= ?
              AND median_score IS NOT NULL
              AND weights <= ?
            ORDER BY median_score LIMIT 1
            """,
            (max_time, max_weights),
        )
        row = cur.fetchone()
        return None if row is None else self._conf_results_by_id[row[0]]

    def top_confs(self, max_time: float, limit: int, max_weights: int | None = None) -> Iterable[ConfResults]:
        """A configuration with the highest (self) score with time = t."""

        params = {
            "max_time": max_time,
            "limit": limit,
        }
        if max_weights is None:
            weights_condition = ""
        else:
            weights_condition = "AND weights <= :max_weights"
            params["max_weights"] = max_weights

        cur = self._db.execute(
            f"""
            SELECT id FROM conf
            WHERE matches_template
              AND median_time IS NOT NULL
              AND median_time <= :max_time
              {weights_condition}
              AND median_score IS NOT NULL
            ORDER BY median_score LIMIT :limit
            """,
            params
        )
        for row in cur:
            yield self._conf_results_by_id[row[0]]
    
    def add_run(self, conf: Configuration, run: Run):
        with latency.timer("ResultSet.add_run"):
            conf_results = self._find_or_add_conf(conf)

            self._result_db.add_run(
                conf, run, timestamp=datetime.now(), conf_id=conf_results.id
            )

            conf_results.add_run(run)

            self._update_conf_scores(conf_results)
            count = 1
            for neighbor in conf_neighbors(conf_results.conf, self._template):
                neighbor_results = self._find_or_add_conf(neighbor)
                neighbor_results.add_neighbor_run(conf_results, run)
                self._update_conf_scores(neighbor_results)
                count += 1

            logging.info(f"Updated scores for {count} configurations")

    def get_results_by_weights(self, max_time=None):
        if max_time is None:
            return sorted(
                self._conf_results_by_id.values(), key=lambda cr: cr.conf.model.weights
            )
        else:
            valid_crs = []
            for cr in self._conf_results_by_id.values():
                if not self._template.match(cr.conf):
                    continue
                median_time = cr.median_time(self._system)
                if median_time is not None and median_time <= max_time:
                    valid_crs.append(cr)
            valid_crs.sort(key=lambda cr: cr.conf.model.weights)
            return valid_crs


    def get_results_by_time(self, max_weights: int):
            cur = self._db.execute(
                """
                SELECT id
                FROM conf
                WHERE median_score IS NOT NULL
                  AND median_time IS NOT NULL
                  AND weights <= :max_weights
                  AND matches_template
                ORDER BY median_time
                """,
                {"max_weights": max_weights},
            )
            for row in cur:
                yield self._conf_results_by_id[row[0]]

    # def train_test_split(self) -> tuple:
    #     train_set = ResultSet(
    #         result_db=None, template=self._template, populate_neighbors=False
    #     )
    #     test_set = ResultSet(
    #         result_db=None, template=self._template, populate_neighbors=False
    #     )

    #     for conf_results in self._conf_results_by_id.values():
    #         for run in conf_results.runs:
    #             target_set = train_set if random.random() < 0.9 else test_set
    #             target_set.add_run_conf(
    #                 conf_results.conf,
    #                 run,
    #                 update_scores=False,
    #             )

    #     return train_set, test_set

    # def find_conf_id(self, conf: Configuration) -> Optional[int]:
    #     conf_results = self._all_conf_results_by_conf.get(conf)
    #     return None if conf_results is None else conf_results.id

    # def get_conf_results(self, conf: Configuration) -> Optional[ConfResults]:
    #     return self._all_conf_results_by_conf.get(conf)

    # def all_conf_results(self) -> Iterable[ConfResults]:
    #     return self._all_conf_results_by_conf.values()

    # def _update_neighbors(self, conf=None):
    #     with latency.timer("ResultSet._update_neighbors"):
    #         for neighbor in conf_neighbors(conf, self._template):
    #             self._find_or_add_conf(neighbor)

    # def _update_scores(self, conf_results: ConfResults):
    #     self._db.execute(
    #         "UPDATE conf SET score = ? WHERE id = ?",
    #         (conf_results.median_score, conf_results.id),
    #     )

    # def add_run(
    #     self,
    #     conf_results: ConfResults,
    #     run: Run,
    #     update_scores: bool = True,
    # ):
    #     conf_results.add_run(run)

    #     if update_scores and self._template.match_conf(conf_results.conf):
    #         self._update_scores(conf_results)
    #         # self._update_neighbors(conf_results.conf)

    # def add_run_conf(
    #     self,
    #     conf: Configuration,
    #     run: Run,
    #     update_scores=False,
    # ):
    #     assert isinstance(run, Run)
    #     conf_results = self._find_or_add_conf(conf)
    #     self.add_run(conf_results, run, update_scores)

    # def add_record(
    #     self, record: TrainingRecord, run: Run, update_scores=True
    # ) -> ConfResults:
    #     conf = record.conf
    #     assert conf_is_valid(conf)
    #     assert self._template.match_conf(conf)

    #     self._result_db.add_record(record, run)
    #     conf_results = self._find_or_add_conf(conf)
    #     self.add_run(conf_results, run, update_scores=update_scores)

    #     return conf_results

    # def all_results_for_t(self, t: int) -> Iterable[ConfResults]:
    #     cur = self._db.execute(
    #         "SELECT id FROM conf WHERE t = ? AND score IS NOT NULL", (t,)
    #     )
    #     for row in cur:
    #         yield self._conf_results_by_id[row[0]]

    # def total_runs_count(self):
    #     with latency.timer("ResultSet.total_runs_count"):
    #         return sum(len(cr.runs) for cr in self._conf_results_by_id.values())
    #         # cur = self._db.execute("SELECT COUNT(*) FROM run")
    #         # return cur.fetchone()[0]

    # def num_runs_by_id(self, conf_id):
    #     conf_runs = self._conf_results_by_id.get(conf_id)
    #     return 0 if conf_runs is None else len(conf_runs.runs)

    # def runs_count(self, t, max_weights=INF, min_weights=512):
    #     cur = self._db.execute(
    #         """
    #         SELECT id
    #         FROM conf
    #         WHERE conf.t = ?
    #           AND conf.weights <= ?
    #           AND conf.weights >= ?
    #         """,
    #         (t, max_weights, min_weights),
    #     )
    #     return sum(self.num_runs_by_id(row[0]) for row in cur)
    #     # return cur.fetchone()[0]

    # def runs_count_per_t(self):
    #     runs_per_t = {}
    #     for i in range(11):
    #         runs_per_t[2**i] = 0

    #     for conf_results in self._conf_results_by_id.values():
    #         runs_per_t[conf_results.conf.t] += len(conf_results.runs)

    #     return runs_per_t

    # def confs_count(self, t, max_weights=None):
    #     if max_weights is None:
    #         cur = self._db.execute(
    #             "SELECT COUNT(*) FROM conf WHERE t = ? AND score NOT NULL",
    #             (t,),
    #         )
    #     else:
    #         cur = self._db.execute(
    #             "SELECT COUNT(*) FROM conf WHERE t = ? AND weights < ? AND score NOT NULL",
    #             (t, max_weights),
    #         )
    #     return cur.fetchone()[0]

    # def top_conf_all_t(self, t_lo, t_hi):
    #     """A configuration with the highest (self) score with time = t."""
    #     cur = self._db.execute(
    #         "SELECT id FROM conf "
    #         + "WHERE t >= ? AND T <= ? AND score IS NOT NULL "
    #         + "ORDER BY score LIMIT 1",
    #         (t_lo, t_hi),
    #     )
    #     row = cur.fetchone()
    #     return None if row is None else self._conf_results_by_id[row[0]]

    # def top_conf_for_tokenset(self, t, tokenset):
    #     best_conf = None
    #     for conf_results in self._conf_results_by_id.values():
    #         if (conf_results.conf.t == t and
    #             conf_tokens_name(conf_results.conf) == tokenset and
    #             conf_results.median_score is not None and
    #             (best_conf is None or
    #              conf_results.median_score < best_conf.median_score)):
    #             best_conf = conf_results
    #     return best_conf

    # def top_confs(self, t, max_weights) -> Iterable[ConfResults]:
    #     with latency.timer("ResultSet.top_confs"):
    #         cur = self._db.execute(
    #             f"SELECT id FROM conf "
    #             + "WHERE t = ? AND weights <= ? AND score IS NOT NULL "
    #             + "ORDER BY score",
    #             (t, max_weights),
    #         )
    #         for row in cur:
    #             yield self._conf_results_by_id[row[0]]

    # def top_pred_confs(self, t, max_weights, limit=None):
    #     limit_str = "" if limit is None else f"-{limit}"
    #     with latency.timer(f"ResultSet.top_pred_confs{limit_str}"):
    #         limit_clause = f"LIMIT {limit}" if limit else ""
    #         with latency.timer(f"ResultSet.top_pred_confs-cur"):
    #             cur = self._db.execute(
    #                 f"""
    #                 SELECT id
    #                 FROM conf
    #                 WHERE t = ? AND weights <= ? AND pred_score IS NOT NULL
    #                 ORDER BY pred_score
    #                 """
    #                 + limit_clause,
    #                 (t, max_weights),
    #             )
    #         for row in cur:
    #             yield self._conf_results_by_id[row[0]]

    # def top_confs_by_score(self, t, limit=10):
    #     with latency.timer(f"ResultSet.top_pred_confs-cur"):
    #         cur = self._db.execute(
    #             f"""
    #             SELECT id
    #             FROM conf
    #             WHERE t = ? AND score IS NOT NULL
    #             ORDER BY score
    #             LIMIT ?
    #             """,
    #             (t, limit),
    #         )
    #     for row in cur:
    #         yield self._conf_results_by_id[row[0]]

    # def get_confs(self):
    #     for conf_results in self._conf_results_by_id.values():
    #         yield conf_results.conf

    # def update_pred_scores(self, confs: Iterable[Configuration], scores: Iterable[float]):
    #     logging.info("Updating predicted scores in ResultSet")
    #     for conf, score in zip(confs, scores):
    #         conf_results = self._find_or_add_conf(conf)

    #         if len(conf_results.runs) > 0:
    #             scores = [r.loss for r in conf_results.runs]
    #             scores.append(score)
    #             score = median(scores)

    #         conf_results.pred_score = score
    #         self._db.execute(
    #             "UPDATE conf SET pred_score = ? WHERE id = ?",
    #             (score, conf_results.id),
    #         )

    # def all_conf_runs(self):
    #     for conf_results in self._all_conf_results_by_conf.values():
    #         for run in conf_results.runs:
    #             yield conf_results.conf, run

    # def has_runs(self, conf):
    #     conf_results = self._all_conf_results_by_conf.get(conf)
    #     return conf_results is not None and conf_results.runs
