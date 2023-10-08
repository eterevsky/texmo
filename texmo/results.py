import logging
import math
import os
import random
import sqlite3
from collections.abc import Iterable
from statistics import median
from typing import Optional

from . import latency
from .common import INF
from .configuration import (Configuration, Template,
                            conf_is_valid, conf_neighbors)
from .record import TrainingRecord
from .resultdb import ResultDB
from .run import Run


class ConfResults(object):
    def __init__(self, id: int, conf: Configuration):
        self.id: int = id
        self.conf: Configuration = conf
        self.runs: list[Run] = []
        self.pred_score: Optional[float] = None

    @property
    def median_score(self) -> Optional[float]:
        if self.runs:
            return median(r.loss for r in self.runs)
        else:
            return None

    def add_run(self, run: Run):
        self.runs.append(run)


class ResultSet(object):
    def __init__(
        self,
        result_db: Optional[ResultDB],
        template: Template,
        populate_neighbors: bool = True,
        verbose: bool = True,
    ):
        self._template: Template = template

        # Configurations, matching the template
        self._conf_results_by_id: dict[int, ConfResults] = {}
        # All configurations from the DB
        self._all_conf_results_by_conf: dict[Configuration, ConfResults] = {}

        self._db = sqlite3.connect(":memory:")
        schema_path = os.path.join(os.path.dirname(__file__), "runtime-db.sql")
        with open(schema_path) as schema:
            self._db.executescript(schema.read())

        if result_db is None:
            self._result_db = ResultDB()
        else:
            self._result_db = result_db
            self._import_from_result_db()

        if populate_neighbors:
            logging.info("Generating all neighbors")
            self.update_all_neighbors()

    def train_test_split(self) -> tuple:
        train_set = ResultSet(
            result_db=None, template=self._template, populate_neighbors=False
        )
        test_set = ResultSet(
            result_db=None, template=self._template, populate_neighbors=False
        )

        for conf_results in self._conf_results_by_id.values():
            for run in conf_results.runs:
                target_set = train_set if random.random() < 0.9 else test_set
                target_set.add_run_conf(
                    conf_results.conf,
                    run,
                    update_scores=False,
                )

        return train_set, test_set

    def _find_or_add_conf(self, conf: Configuration, id=None) -> ConfResults:
        """Finds the conf in the db and returns the copy with populated id."""
        with latency.timer("ResultSet._find_or_add_conf"):
            assert isinstance(conf, Configuration)
            conf_results = self._all_conf_results_by_conf.get(conf)
            if conf_results is not None:
                assert id is None or conf_results.id == id
                return conf_results
            if id is None:
                id = self._result_db.find_or_add_conf(conf)
            conf_results = ConfResults(id, conf)
            if self._template.match_conf(conf):
                self._conf_results_by_id[id] = conf_results
            self._all_conf_results_by_conf[conf] = conf_results

            conf_dict = {
                "id": id,
                "spec": str(conf.model),
                "lr": conf.lr,
                "sample_len": conf.sample_len,
                "batch": conf.batch,
                "t": conf.t,
                "weights": conf.model.weights,
            }
            self._db.execute(
                """
                INSERT INTO conf(id, spec, lr, sample_len, batch, t, weights)
                VALUES(:id, :spec, :lr, :sample_len, :batch, :t, :weights)
                """,
                conf_dict,
            )

            return conf_results

    def _import_from_result_db(self):
        """Import from the persistent DB all matching confs.

        Only the confs that are matching `self._template` are being selected.
        """
        with latency.timer("ResultSet._import_from_result_db"):
            logging.info(
                "Importing relevant configurations and results from ResultDB"
            )
            n = 0
            n_w_template = 0

            for id, conf, run in self._result_db.get_confs_runs():
                n += 1
                if self._template.match_conf(conf):
                    n_w_template += 1
                conf_results = self._find_or_add_conf(conf, id)
                conf_results.add_run(run)

            logging.info(f"Imported {n} runs ({n_w_template} matching template)")
            logging.info("Populating scores from run results")
            self.update_all_scores()

    def find_conf_id(self, conf: Configuration) -> Optional[int]:
        conf_results = self._all_conf_results_by_conf.get(conf)
        return None if conf_results is None else conf_results.id

    def get_conf_results(self, conf: Configuration) -> Optional[ConfResults]:
        return self._all_conf_results_by_conf.get(conf)

    def all_conf_results(self) -> Iterable[ConfResults]:
        return self._all_conf_results_by_conf.values()

    def _update_neighbors(self, conf=None):
        with latency.timer("ResultSet._update_neighbors"):
            for neighbor in conf_neighbors(conf, self._template):
                self._find_or_add_conf(neighbor)

    def update_all_neighbors(self):
        with latency.timer("ResultSet.update_all_neighbors"):
            confs = []
            for conf_results in self._conf_results_by_id.values():
                if conf_results.median_score is not None:
                    confs.append(conf_results.conf)

            for conf in confs:
                for neighbor in conf_neighbors(conf, self._template):
                    self._find_or_add_conf(neighbor)
            n = len(confs)
            logging.info(f"Generated neighbors for {n} confs")

    def update_all_scores(self):
        with latency.timer("ResultSet.update_all_scores"):
            for conf_results in self._conf_results_by_id.values():
                score = conf_results.median_score
                if score is not None:
                    self._db.execute(
                        "UPDATE conf SET score = ? WHERE id = ?",
                        (score, conf_results.id),
                    )

    def _update_scores(self, conf_results: ConfResults):
        self._db.execute(
            "UPDATE conf SET score = ? WHERE id = ?",
            (conf_results.median_score, conf_results.id),
        )

    def add_run(
        self,
        conf_results: ConfResults,
        run: Run,
        update_scores: bool = True,
    ):
        conf_results.add_run(run)

        if update_scores and self._template.match_conf(conf_results.conf):
            self._update_scores(conf_results)
            # self._update_neighbors(conf_results.conf)

    def add_run_conf(
        self,
        conf: Configuration,
        run: Run,
        update_scores=False,
    ):
        assert isinstance(run, Run)
        conf_results = self._find_or_add_conf(conf)
        self.add_run(conf_results, run, update_scores)

    def add_record(
        self, record: TrainingRecord, run: Run, update_scores=True
    ) -> ConfResults:
        conf = record.conf
        assert conf_is_valid(conf)
        assert self._template.match_conf(conf)

        self._result_db.add_record(record, run)
        conf_results = self._find_or_add_conf(conf)
        self.add_run(conf_results, run, update_scores=update_scores)

        return conf_results

    def get_results_by_weights(self):
        return sorted(
            self._conf_results_by_id.values(), key=lambda cr: cr.conf.model.weights
        )

    def all_results_for_t(self, t: int) -> Iterable[ConfResults]:
        cur = self._db.execute(
            "SELECT id FROM conf WHERE t = ? AND score IS NOT NULL", (t,)
        )
        for row in cur:
            yield self._conf_results_by_id[row[0]]

    def total_runs_count(self):
        with latency.timer("ResultSet.total_runs_count"):
            return sum(len(cr.runs) for cr in self._conf_results_by_id.values())
            # cur = self._db.execute("SELECT COUNT(*) FROM run")
            # return cur.fetchone()[0]

    def num_runs_by_id(self, conf_id):
        conf_runs = self._conf_results_by_id.get(conf_id)
        return 0 if conf_runs is None else len(conf_runs.runs)

    def runs_count(self, t, max_weights=INF, min_weights=512):
        cur = self._db.execute(
            """
            SELECT id
            FROM conf
            WHERE conf.t = ?
              AND conf.weights <= ?
              AND conf.weights >= ?
            """,
            (t, max_weights, min_weights),
        )
        return sum(self.num_runs_by_id(row[0]) for row in cur)
        # return cur.fetchone()[0]

    def runs_count_per_t(self):
        runs_per_t = {}
        for i in range(11):
            runs_per_t[2**i] = 0

        for conf_results in self._conf_results_by_id.values():
            runs_per_t[conf_results.conf.t] += len(conf_results.runs)

        return runs_per_t

    def confs_count(self, t, max_weights=None):
        if max_weights is None:
            cur = self._db.execute(
                "SELECT COUNT(*) FROM conf WHERE t = ? AND score NOT NULL",
                (t,),
            )
        else:
            cur = self._db.execute(
                "SELECT COUNT(*) FROM conf WHERE t = ? AND weights < ? AND score NOT NULL",
                (t, max_weights),
            )
        return cur.fetchone()[0]

    def top_conf(self, t) -> ConfResults:
        """A configuration with the highest (self) score with time = t."""
        cur = self._db.execute(
            "SELECT id FROM conf "
            + "WHERE t = ? AND score IS NOT NULL "
            + "ORDER BY score LIMIT 1",
            (t,),
        )
        row = cur.fetchone()
        return None if row is None else self._conf_results_by_id[row[0]]

    def top_conf_all_t(self, t_lo, t_hi):
        """A configuration with the highest (self) score with time = t."""
        cur = self._db.execute(
            "SELECT id FROM conf "
            + "WHERE t >= ? AND T <= ? AND score IS NOT NULL "
            + "ORDER BY score LIMIT 1",
            (t_lo, t_hi),
        )
        row = cur.fetchone()
        return None if row is None else self._conf_results_by_id[row[0]]

    def top_confs(self, t, max_weights) -> Iterable[ConfResults]:
        with latency.timer("ResultSet.top_confs"):
            cur = self._db.execute(
                f"SELECT id FROM conf "
                + "WHERE t = ? AND weights <= ? AND score IS NOT NULL "
                + "ORDER BY score",
                (t, max_weights),
            )
            for row in cur:
                yield self._conf_results_by_id[row[0]]

    def top_pred_confs(self, t, max_weights, limit=None):
        limit_str = "" if limit is None else f"-{limit}"
        with latency.timer(f"ResultSet.top_pred_confs{limit_str}"):
            limit_clause = f"LIMIT {limit}" if limit else ""
            with latency.timer(f"ResultSet.top_pred_confs-cur"):
                cur = self._db.execute(
                    f"""
                    SELECT id
                    FROM conf
                    WHERE t = ? AND weights <= ? AND pred_score IS NOT NULL
                    ORDER BY pred_score
                    """
                    + limit_clause,
                    (t, max_weights),
                )
            for row in cur:
                yield self._conf_results_by_id[row[0]]

    def top_confs_by_score(self, t, limit=10):
        with latency.timer(f"ResultSet.top_pred_confs-cur"):
            cur = self._db.execute(
                f"""
                SELECT id
                FROM conf
                WHERE t = ? AND score IS NOT NULL
                ORDER BY score
                LIMIT ?
                """,
                (t, limit),
            )
        for row in cur:
            yield self._conf_results_by_id[row[0]]

    def get_confs(self):
        for conf_results in self._conf_results_by_id.values():
            yield conf_results.conf

    def update_pred_scores(self, confs: Iterable[Configuration], scores: Iterable[float]):
        logging.info("Updating predicted scores in ResultSet")
        for conf, score in zip(confs, scores):
            conf_results = self._find_or_add_conf(conf)

            if len(conf_results.runs) > 0:
                scores = [r.loss for r in conf_results.runs]
                scores.append(score)
                score = median(scores)

            conf_results.pred_score = score
            self._db.execute(
                "UPDATE conf SET pred_score = ? WHERE id = ?",
                (score, conf_results.id),
            )

    def all_conf_runs(self):
        for conf_results in self._all_conf_results_by_conf.values():
            for run in conf_results.runs:
                yield conf_results.conf, run

    def has_runs(self, conf):
        conf_results = self._all_conf_results_by_conf.get(conf)
        return conf_results is not None and conf_results.runs


def open_db(path):
    exists = os.path.exists(path)
    db = sqlite3.connect(path)
    if not exists:
        with open("results.sql") as schema:
            db.executescript(schema.read())
            db.commit()
    return db
