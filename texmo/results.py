from collections import namedtuple
import logging
import math
import os
import sqlite3
from statistics import median

from .configuration import (
    CONF_FIELDS,
    INF,
    conf_from_row,
    conf_neighbors,
    Configuration,
    conf_from_record,
    conf_is_valid,
)
from . import latency
from .record import TrainingRecord
from .spec import ModelSpec, ObsoleteSpec


Results = namedtuple(
    "Results",
    [
        "score",
        "cluster_score",
        "num_runs",
    ],
)


class ResultSet(object):
    def __init__(self, result_db, template, populate_neighbors=True):
        self._result_db = result_db
        self._template = template
        self._confs = {}  # conf id -> Configuration
        self._runs = {}  # conf id -> [run losses]
        self._db = sqlite3.connect(":memory:")
        with open("runtime-db.sql") as schema:
            self._db.executescript(schema.read())

        if self._result_db is not None:
            self._import_from_result_db()
        if populate_neighbors:
            logging.info("Generating all neighbors")
            self.update_all_neighbors()

    def _import_from_result_db(self):
        """Import from the persistent DB all matching confs.

        Only the confs that are matching `self._template` are being selected.
        """
        with latency.timer("ResultSet._import_from_result_db"):
            print("Importing relevant configurations and results from ResultDB")
            n = 0
            for conf, loss in self._result_db.get_confs_runs(self._template):
                conf = conf._replace(id=None)
                conf = self._find_or_add_conf(conf)

                n += 1
                conf_runs = self._runs.get(conf.id)
                if conf_runs is None:
                    conf_runs = []
                    self._runs[conf.id] = conf_runs
                conf_runs.append(loss)
            print(f"Imported {n} runs")

            print("Populating scores from run results")
            self.update_all_scores()

    def find_conf_id(self, conf):
        cur = self._db.execute(
            """
            SELECT id FROM conf
            WHERE spec = ?
              AND lr = ?
              AND sample_len = ?
              AND batch = ?
              AND regularization = ?
              AND init_scale = ?
              AND t = ?
            """,
            (
                str(conf.model),
                conf.lr,
                conf.sample_len,
                conf.batch,
                conf.regularization,
                conf.init_scale,
                conf.t,
            ),
        )
        rows = cur.fetchall()
        assert len(rows) <= 1
        return rows[0][0] if rows else None

    def _find_or_add_conf(self, conf):
        """Finds the conf in the db and returns the copy with populated id."""
        id = self.find_conf_id(conf)
        assert conf.id == None or id == conf.id
        if id is not None:
            return self._confs[id]

        conf_dict = {
            "spec": str(conf.model),
            "lr": conf.lr,
            "sample_len": conf.sample_len,
            "batch": conf.batch,
            "regularization": conf.regularization,
            "init_scale": conf.init_scale,
            "t": conf.t,
            "weights": conf.model.weights,
        }
        cursor = self._db.execute(
            """
            INSERT INTO conf(spec, lr, sample_len, batch, regularization, init_scale, t, weights)
            VALUES(:spec, :lr, :sample_len, :batch, :regularization, :init_scale, :t, :weights)
            """,
            conf_dict,
        )
        id = cursor.lastrowid
        conf = conf._replace(id=id)
        self._confs[id] = conf
        return conf

    def _update_neighbors(self, conf=None):
        with latency.timer("ResultSet._update_neighbors"):
            for neighbor in conf_neighbors(conf, self._template):
                neighbor = neighbor._replace(id=None)
                neighbor = self._find_or_add_conf(neighbor)

    def update_all_neighbors(self):
        with latency.timer("ResultSet.update_all_neighbors"):
            cur = self._db.execute(
                "SELECT id FROM conf WHERE score IS NOT NULL"
            )
            i = 0
            for i, row in enumerate(cur):
                conf_id = row[0]
                conf = self._confs[conf_id]
                for neighbor in conf_neighbors(conf, self._template):
                    self._find_or_add_conf(neighbor)
            logging.info(f"Generated neighbors for {i} confs")

            self._db.commit()

    def update_all_scores(self):
        with latency.timer("ResultSet.update_all_scores"):
            for conf_id, conf_runs in self._runs.items():
                score = median(conf_runs)
                self._db.execute(
                    "UPDATE conf SET score = ? WHERE id = ?", [score, conf_id]
                )
            self._db.commit()

    def _update_scores(self, conf):
        with latency.timer("ResultSet._update_scores"):
            score = median(self._runs[conf.id])
            self._db.execute(
                "UPDATE conf SET score = ? WHERE id = ?", (score, conf.id)
            )


    def add_run(self, conf: Configuration, loss: float, update_scores=True):
        conf = self._find_or_add_conf(conf)
        if math.isnan(loss) or loss is None:
            loss = INF

        conf_runs = self._runs.get(conf.id)
        if conf_runs is None:
            conf_runs = []
            self._runs[conf.id] = conf_runs
        conf_runs.append(loss)

        if update_scores:
            self._update_scores(conf)
            self._update_neighbors(conf)

    def add_record(self, record, update_scores=True):
        conf = conf_from_record(record)
        assert conf_is_valid(conf)

        if self._result_db is not None:
            self._result_db.add_record(record)

        conf = self._find_or_add_conf(conf)
        self.add_run(conf, record.loss, update_scores)

        return conf, record.loss

    def all_results_by_weights(self):
        # cur = self._db.execute(
        #     "SELECT conf_id, COUNT(*) FROM run GROUP BY conf_id"
        # )
        # num_runs = dict(cur)

        cur = self._db.execute(
            f"""
            SELECT id, score, pred_score
            FROM conf
            WHERE score IS NOT NULL
            ORDER BY weights
            """
        )
        for id, score, pred_score in cur:
            conf = self._confs[id]
            results = Results(score, pred_score, self.num_runs_by_id(id))
            yield conf, results

    def total_runs_count(self):
        with latency.timer("ResultSet.total_runs_count"):
            return sum(len(s) for s in self._runs.values())
            # cur = self._db.execute("SELECT COUNT(*) FROM run")
            # return cur.fetchone()[0]

    def num_runs_by_id(self, conf_id):
        conf_runs = self._runs.get(conf_id)
        return 0 if conf_runs is None else len(conf_runs)

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

        for conf_id, conf_runs in self._runs.items():
            conf = self._confs[conf_id]
            runs_per_t[conf.t] += len(conf_runs)

        # cur = self._db.execute("SELECT id, t FROM conf")
        # for id, t in cur:
        #     runs_per_t[t] += self.num_runs_by_id(id)
        return runs_per_t

        # cur = self._db.execute(
        #     """
        #     SELECT conf.t, COUNT(*)
        #     FROM conf, run
        #     WHERE conf.id = run.conf_id
        #     GROUP BY conf.t
        #     """
        # )
        # return dict(cur)

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

    def top_conf(self, t):
        """A configuration with the highest (self) score with time = t."""
        cur = self._db.execute(
            "SELECT id FROM conf "
            + "WHERE t = ? AND score IS NOT NULL "
            + "ORDER BY score LIMIT 1",
            (t,),
        )
        row = cur.fetchone()
        return None if row is None else self._confs[row[0]]

    def top_confs(self, t, max_weights):
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

    def find(self, conf):
        """Find the configuration by its fields.

        Returns an instance with populated id, and Results
        """
        conf = self._find_or_add_conf(conf)
        cur = self._db.execute("SELECT score FROM conf WHERE id = ?", [conf.id])
        return conf, cur.fetchone()[0]

    def find_by_id(self, conf_id):
        # cur = self._db.execute(
        #     f"SELECT {CONF_FIELDS} FROM conf WHERE id = ?", (conf_id,)
        # )
        # return conf_from_row(cur.fetchone())
        return self._confs.get(conf_id)

    def top_pred_confs(self, t, max_weights, limit=None):
        limit_str = "" if limit is None else f"-{limit}"
        with latency.timer(f"ResultSet.top_pred_confs{limit_str}"):
            limit_clause = f"LIMIT {limit}" if limit else ""
            with latency.timer(f"ResultSet.top_pred_confs-cur"):
                cur = self._db.execute(
                    f"""
                    SELECT id, score, pred_score
                    FROM conf
                    WHERE t = ?
                    AND weights <= ?
                    AND pred_score IS NOT NULL
                    ORDER BY pred_score
                    """
                    + limit_clause,
                    (t, max_weights),
                )
            for id, score, pred_score in cur:
                # conf = conf_from_row(row[:8])
                conf = self._confs[id]
                results = Results(score, pred_score, self.num_runs_by_id(conf.id))
                yield conf, results

    def all_confs(self):
        return self._confs.values()
        # cur = self._db.execute(f"SELECT {CONF_FIELDS} FROM conf")
        # for row in cur:
        #     yield conf_from_row(row)

    def update_pred_scores(self, confs, scores):
        for conf, score in zip(confs, scores):
            if conf.id is None:
                conf = self._find_or_add_conf(conf)
            self._db.execute(
                "UPDATE conf SET pred_score = ? WHERE id = ?", (score, conf.id)
            )

    def all_conf_runs(self):
        for conf_id, conf_runs in self._runs.items():
            conf = self._confs[conf_id]
            for loss in conf_runs:
                yield conf, loss
        # cur = self._db.execute(f"SELECT {CONF_FIELDS} FROM conf")
        # for row in cur:
        #     conf = conf_from_row(row[:8])
        #     conf_runs = self._runs.get(conf.id)
        #     if conf_runs:
        #         for loss in conf_runs:
        #             yield conf, loss

    def has_runs(self, conf):
        id = self.find_conf_id(conf)
        # if id is None:
        #     return False
        return id in self._runs
        # cur = self._db.execute(
        #     "SELECT 1 FROM run WHERE conf_id = ? LIMIT 1", (id,)
        # )
        # return cur.fetchone() is not None


def open_db(path):
    exists = os.path.exists(path)
    db = sqlite3.connect(path)
    if not exists:
        with open("results.sql") as schema:
            db.executescript(schema.read())
            db.commit()
    return db
