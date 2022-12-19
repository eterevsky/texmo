from collections import namedtuple
from collections.abc import Iterable
import logging
import math
import os
import random
import sqlite3
from statistics import median
from typing import Optional

from .common import INF
from .configuration import (
    Configuration,
    Template,
    conf_neighbors,
    conf_from_record,
    conf_is_valid,
)
from .confresults import ConfResults, Run
from . import latency
from .resultdb import ResultDB


class ResultSet(object):
    def __init__(
        self,
        result_db: Optional[ResultDB],
        template: Template,
        populate_neighbors: bool = True,
    ):
        self._template: Template = template
        self._confs: dict[int, ConfResults] = {}
        self._conf_to_results: dict[Configuration, ConfResults] = {}

        self._db = sqlite3.connect(":memory:")
        schema_path = os.path.join(os.path.dirname(__file__), "runtime-db.sql")
        with open(schema_path) as schema:
            self._db.executescript(schema.read())

        if result_db is None:
            result_db = ResultDB()

        self._result_db: ResultDB = result_db
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

        for conf_results in self._confs.values():
            for run in conf_results.runs:
                target_set = train_set if random.random() < 0.8 else test_set
                target_set.add_run_conf(
                    conf_results.conf,
                    run.loss,
                    run.step_loss,
                    update_scores=False,
                )

        return train_set, test_set

    def _find_or_add_conf(self, conf: Configuration, id=None) -> ConfResults:
        """Finds the conf in the db and returns the copy with populated id."""
        assert isinstance(conf, Configuration)
        conf_results = self._conf_to_results.get(conf)
        if conf_results is not None:
            assert id is None or conf_results.id == id
            return conf_results
        if id is None:
            id = self._result_db.find_or_add_conf(conf)
        conf_results = ConfResults(id, conf)
        self._confs[id] = conf_results
        self._conf_to_results[conf] = conf_results

        conf_dict = {
            "id": id,
            "spec": str(conf.model),
            "lr": conf.lr,
            "sample_len": conf.sample_len,
            "batch": conf.batch,
            "regularization": conf.regularization,
            "init_scale": conf.init_scale,
            "t": conf.t,
            "weights": conf.model.weights,
        }
        self._db.execute(
            """
            INSERT INTO conf(id, spec, lr, sample_len, batch, regularization, init_scale, t, weights)
            VALUES(:id, :spec, :lr, :sample_len, :batch, :regularization, :init_scale, :t, :weights)
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

            template_wo_t = self._template.clone()
            template_wo_t.t = None

            for id, conf, run in self._result_db.get_confs_runs(template_wo_t):
                n += 1
                conf_results = self._find_or_add_conf(conf, id)
                conf_results.add_run(run)

            logging.info(f"Imported {n} runs")
            logging.info("Populating scores from run results")
            self.update_all_scores()

    def find_conf_id(self, conf: Configuration) -> Optional[int]:
        conf_results = self._conf_to_results.get(conf)
        return None if conf_results is None else conf_results.id

    def get_conf_results(self, conf: Configuration) -> Optional[ConfResults]:
        return self._conf_to_results.get(conf)

    def _update_neighbors(self, conf=None):
        with latency.timer("ResultSet._update_neighbors"):
            for neighbor in conf_neighbors(conf, self._template):
                self._find_or_add_conf(neighbor)

    def update_all_neighbors(self):
        with latency.timer("ResultSet.update_all_neighbors"):
            confs = []
            for conf_results in self._conf_to_results.values():
                if conf_results.median_score is not None:
                    confs.append(conf_results.conf)

            for conf in confs:
                for neighbor in conf_neighbors(conf, self._template):
                    self._find_or_add_conf(neighbor)
            n = len(confs)
            logging.info(f"Generated neighbors for {n} confs")

    def update_all_scores(self):
        with latency.timer("ResultSet.update_all_scores"):
            for conf_results in self._confs.values():
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
        loss: float,
        step_loss: Optional[list[float]] = None,
        update_scores: bool = True,
    ):
        if math.isnan(loss) or loss is None:
            loss = INF

        conf_results.add_run(Run(loss, step_loss))

        if update_scores:
            self._update_scores(conf_results)
            self._update_neighbors(conf_results.conf)

    def add_run_conf(
        self,
        conf: Configuration,
        loss: float,
        step_loss: Optional[list[float]] = None,
        update_scores=False,
    ):
        conf_results = self._find_or_add_conf(conf)
        self.add_run(conf_results, loss, step_loss, update_scores)

    def add_record(
        self, record, step_loss: Iterable[float], update_scores=True
    ) -> tuple[ConfResults, float]:
        conf = conf_from_record(record)
        assert conf_is_valid(conf)

        if self._result_db is not None:
            self._result_db.add_record(record, step_loss)

        conf_results = self._find_or_add_conf(conf)
        self.add_run(conf_results, record.loss, step_loss, update_scores)

        return conf_results, record.loss

    def all_results_by_weights(self):
        return sorted(
            self._conf_to_results.values(), key=lambda cr: cr.conf.model.weights
        )

    def all_results_for_t(self, t: int) -> Iterable[ConfResults]:
        cur = self._db.execute(
            "SELECT id FROM conf WHERE t = ? AND score IS NOT NULL", (t,)
        )
        for row in cur:
            yield self._confs[row[0]]

    def total_runs_count(self):
        with latency.timer("ResultSet.total_runs_count"):
            return sum(len(cr.runs) for cr in self._confs.values())
            # cur = self._db.execute("SELECT COUNT(*) FROM run")
            # return cur.fetchone()[0]

    def num_runs_by_id(self, conf_id):
        conf_runs = self._confs.get(conf_id)
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

        for conf_results in self._confs.values():
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
        return None if row is None else self._confs[row[0]]

    def top_conf_all_t(self, t_lo, t_hi):
        """A configuration with the highest (self) score with time = t."""
        cur = self._db.execute(
            "SELECT id FROM conf "
            + "WHERE t >= ? AND T <= ? AND score IS NOT NULL "
            + "ORDER BY score LIMIT 1",
            (t_lo, t_hi),
        )
        row = cur.fetchone()
        return None if row is None else self._confs[row[0]]

    def top_confs(self, t, max_weights) -> Iterable[ConfResults]:
        with latency.timer("ResultSet.top_confs"):
            cur = self._db.execute(
                f"SELECT id FROM conf "
                + "WHERE t = ? AND weights <= ? AND score IS NOT NULL "
                + "ORDER BY score",
                (t, max_weights),
            )
            for row in cur:
                yield self._confs[row[0]]
            # return map(conf_from_row, cur)

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
                yield self._confs[row[0]]

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
            yield self._confs[row[0]]

    def all_confs(self):
        for conf_results in self._conf_to_results.values():
            yield conf_results.conf

    def update_pred_scores(self, confs: Iterable[Configuration], scores):
        for conf, score in zip(confs, scores):
            conf_results = self._find_or_add_conf(conf)
            conf_results.pred_score = score
            self._db.execute(
                "UPDATE conf SET pred_score = ? WHERE id = ?",
                (score, conf_results.id),
            )

    def all_conf_runs(self):
        for conf_results in self._conf_to_results.values():
            for run in conf_results.runs:
                yield conf_results.conf, run.loss

    def has_runs(self, conf):
        conf_results = self._conf_to_results.get(conf)
        return conf_results is not None and conf_results.runs


def open_db(path):
    exists = os.path.exists(path)
    db = sqlite3.connect(path)
    if not exists:
        with open("results.sql") as schema:
            db.executescript(schema.read())
            db.commit()
    return db
