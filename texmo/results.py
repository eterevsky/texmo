import argparse
import csv
from collections import namedtuple
from itertools import chain
import math
import os
import sqlite3
from statistics import median, StatisticsError

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
from .spec import ModelSpec, ObsoleteSpec, is_reachable_spec


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
        self._db = sqlite3.connect(":memory:")
        with open("runtime-db.sql") as schema:
            self._db.executescript(schema.read())

        if self._result_db is not None:
            self._import_from_result_db(populate_neighbors)

    def _import_from_result_db(self, populate_neighbors):
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
                self._db.execute(
                    "INSERT INTO run(conf_id, loss) VALUES(?, ?)",
                    (conf.id, loss),
                )

            print(f"Imported {n} runs")

            print("Populating scores from run results")
            self.update_all_scores()
            if populate_neighbors:
                print("Populating neighbors")
                self.update_all_neighbors()
                # print("Populating cluster scores")
                # self.update_all_cluster_scores()

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
                str(conf.spec),
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
            return conf._replace(id=id)

        conf_dict = {
            "spec": str(conf.spec),
            "lr": conf.lr,
            "sample_len": conf.sample_len,
            "batch": conf.batch,
            "regularization": conf.regularization,
            "init_scale": conf.init_scale,
            "t": conf.t,
            "weights": conf.spec.weights(),
        }
        cursor = self._db.execute(
            """
            INSERT INTO conf(spec, lr, sample_len, batch, regularization, init_scale, t, weights)
            VALUES(:spec, :lr, :sample_len, :batch, :regularization, :init_scale, :t, :weights)
            """,
            conf_dict,
        )
        id = cursor.lastrowid
        return conf._replace(id=id)

    def _update_neighbors(self, conf=None, conf_id=None):
        with latency.timer("ResultSet._update_neighbors"):
            if conf is not None:
                if conf_id is not None:
                    assert conf.id == conf_id
                else:
                    conf_id = conf.id

            cur = self._db.execute(
                f"""
                SELECT 1 FROM neighbor
                WHERE conf1_id = :id
                LIMIT 1
                """,
                {"id": conf_id},
            )
            if cur.fetchone():
                return

            if conf is None:
                conf = self.find_by_id(conf_id)
                assert conf is not None
                assert conf.id == conf_id

            for neighbor in conf_neighbors(conf, self._template):
                neighbor = neighbor._replace(id=None)
                neighbor = self._find_or_add_conf(neighbor)
                self._db.execute(
                    "INSERT INTO neighbor(conf1_id, conf2_id) VALUES (?, ?)",
                    (conf.id, neighbor.id),
                )

    def update_all_neighbors(self):
        with latency.timer("ResultSet.update_all_neighbors"):
            self._db.execute("DELETE FROM neighbor")
            cur = self._db.execute(
                f"SELECT {CONF_FIELDS} FROM conf WHERE score IS NOT NULL"
            )
            i = 0
            for i, row in enumerate(cur):
                conf = conf_from_row(row)
                conf_id = conf.id
                conf = conf._replace(id=None)
                for neighbor in conf_neighbors(conf, self._template):
                    neighbor = self._find_or_add_conf(neighbor)
                    self._db.execute(
                        "INSERT INTO neighbor(conf1_id, conf2_id) VALUES (?, ?)",
                        (conf_id, neighbor.id),
                    )
            print(f"Updated neighbors for {i} confs")

            self._db.commit()

    def update_all_scores(self):
        with latency.timer("ResultSet.update_all_scores"):
            cur = self._db.execute("SELECT conf_id, loss FROM run")
            scores = {}
            for conf_id, loss in cur:
                if conf_id not in scores:
                    scores[conf_id] = []
                scores[conf_id].append(loss)

            for conf_id, conf_scores in scores.items():
                score = median(conf_scores)
                self._db.execute(
                    "UPDATE conf SET score = ? WHERE id = ?", [score, conf_id]
                )
            self._db.commit()

    def _update_cluster_score(self, conf_id):
        with latency.timer("ResultSet._update_cluster_score"):
            self._update_neighbors(conf=None, conf_id=conf_id)
            cur_runs = self._db.execute(
                "SELECT loss FROM run WHERE conf_id = ?", (conf_id,)
            )
            run_scores = [row[0] for row in cur_runs]
            self_score = median(run_scores) if run_scores else None
            cur_neighbors = self._db.execute(
                f"""
                SELECT conf.score
                FROM conf, neighbor
                WHERE neighbor.conf2_id = ?
                AND conf.id = neighbor.conf1_id
                AND conf.score IS NOT NULL
                """,
                (conf_id,),
            )
            neighbor_scores = [row[0] for row in cur_neighbors]
            try:
                cluster_score = median(chain(run_scores, neighbor_scores))
            except StatisticsError:
                # If we don't have any scores
                cluster_score = None
            if self_score is not None and self_score < cluster_score:
                cluster_score = self_score
            self._db.execute(
                "UPDATE conf SET cluster_score = ? WHERE id = ?",
                (cluster_score, conf_id),
            )

    def update_all_cluster_scores(self):
        neighbors = {}
        cur = self._db.execute("SELECT conf1_id, conf2_id FROM neighbor")
        for conf1_id, conf2_id in cur:
            conf2_neighbors = neighbors.get(conf2_id)
            if conf2_neighbors:
                conf2_neighbors.append(conf1_id)
            else:
                conf2_neighbors = [conf1_id]
                neighbors[conf2_id] = conf2_neighbors

        runs = {}
        cur = self._db.execute("SELECT conf_id, loss FROM run")
        for conf_id, loss in cur:
            conf_runs = runs.get(conf_id)
            if conf_runs:
                conf_runs.append(loss)
            else:
                conf_runs = [loss]
                runs[conf_id] = conf_runs

        self_scores = {}
        for conf_id, r in runs.items():
            self_scores[conf_id] = median(r)

        for conf_id, conf_neighbors in neighbors.items():
            neighbor_scores = (
                self_scores[neighbor]
                for neighbor in conf_neighbors
                if neighbor in self_scores
            )
            try:
                if conf_id in runs:
                    cluster_score = median(
                        chain(neighbor_scores, runs[conf_id])
                    )
                    self_score = self_scores[conf_id]
                    if self_score < cluster_score:
                        cluster_score = self_score
                else:
                    cluster_score = median(neighbor_scores)
            except StatisticsError:
                cluster_score = None

            self._db.execute(
                """
                UPDATE conf SET score = ?, cluster_score = ?
                WHERE id = ?
                """,
                (self_scores.get(conf_id), cluster_score, conf_id),
            )

        self._db.commit()

    def _update_scores(self, conf):
        with latency.timer("ResultSet._update_scores"):
            cur = self._db.execute(
                "SELECT loss FROM run WHERE conf_id = ?", (conf.id,)
            )
            score = median(row[0] for row in cur)
            self._db.execute(
                "UPDATE conf SET score = ? WHERE id = ?", (score, conf.id)
            )

            self._update_neighbors(conf=conf)
            # self._update_cluster_score(conf.id)

            # neighbors = self._db.execute(
            #     "SELECT conf2_id FROM neighbor WHERE conf1_id = ?",
            #     (conf.id,),
            # )
            # for (neighbor_id,) in neighbors:
            #     self._update_neighbors(conf_id=neighbor_id)
            #     self._update_cluster_score(neighbor_id)

    def add_run(self, conf: Configuration, loss: float, update_scores=True):
        conf = self._find_or_add_conf(conf)
        if math.isnan(loss) or loss is None:
            loss = INF

        self._db.execute(
            "INSERT INTO run (conf_id, loss) VALUES (?, ?)", (conf.id, loss)
        )

        if update_scores:
            self._update_scores(conf)

    def add_record(self, record, update_scores=True):
        conf = conf_from_record(record)
        assert conf_is_valid(conf)

        if self._result_db is not None:
            self._result_db.add_record(record)

        conf = self._find_or_add_conf(conf)
        self.add_run(conf, record.loss, update_scores)

        return conf, record.loss

    def all_results_by_weights(self):
        cur = self._db.execute(
            "SELECT conf_id, COUNT(*) FROM run GROUP BY conf_id"
        )
        num_runs = dict(cur)

        cur = self._db.execute(
            f"""
            SELECT {CONF_FIELDS}, score, cluster_score
            FROM conf
            WHERE score IS NOT NULL
            ORDER BY weights
            """
        )
        for row in cur:
            conf = conf_from_row(row[:8])
            results = Results(row[8], row[9], num_runs[row[0]])
            yield conf, results

    def total_runs_count(self):
        with latency.timer("ResultSet.total_runs_count"):
            cur = self._db.execute("SELECT COUNT(*) FROM run")
            return cur.fetchone()[0]

    def runs_count(self, t, max_weights=INF, min_weights=512):
        cur = self._db.execute(
            """
            SELECT COUNT(*)
            FROM conf, run
            WHERE conf.id = run.conf_id
              AND conf.t = ?
              AND conf.weights <= ?
              AND conf.weights >= ?
            """,
            (t, max_weights, min_weights),
        )
        return cur.fetchone()[0]

    def runs_count_per_t(self):
        cur = self._db.execute(
            """
            SELECT conf.t, COUNT(*)
            FROM conf, run
            WHERE conf.id = run.conf_id
            GROUP BY conf.t
            """
        )
        return dict(cur)

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
            f"SELECT {CONF_FIELDS} FROM conf "
            + "WHERE t = ? AND score IS NOT NULL "
            + "ORDER BY score LIMIT 1",
            (t,),
        )
        return conf_from_row(cur.fetchone())

    def top_confs(self, t, max_weights):
        with latency.timer("ResultSet.top_confs"):
            cur = self._db.execute(
                f"SELECT {CONF_FIELDS} FROM conf "
                + "WHERE t = ? AND weights <= ? AND score IS NOT NULL "
                + "ORDER BY score",
                (t, max_weights),
            )
            return map(conf_from_row, cur)

    def find(self, conf):
        """Find the configuration by its fields.

        Returns an instance with populated id, and Results
        """
        conf = self._find_or_add_conf(conf)
        cur = self._db.execute("SELECT score FROM conf WHERE id = ?", [conf.id])
        return conf, cur.fetchone()[0]

    def find_by_id(self, conf_id):
        cur = self._db.execute(
            f"SELECT {CONF_FIELDS} FROM conf WHERE id = ?", (conf_id,)
        )
        return conf_from_row(cur.fetchone())

    def top_pred_confs(self, t, max_weights, limit=None):
        with latency.timer("ResultSet.top_pred_confs"):
            limit_clause = f"LIMIT {limit}" if limit else ""
            cur = self._db.execute(
                f"""
                SELECT {CONF_FIELDS},
                    score,
                    pred_score,
                    (SELECT COUNT(*) FROM run WHERE conf_id = conf.id)
                FROM conf
                WHERE t = ?
                AND weights <= ?
                AND pred_score IS NOT NULL
                ORDER BY pred_score
                """
                + limit_clause,
                (t, max_weights),
            )
            for row in cur:
                conf = conf_from_row(row[:8])
                results = Results(*row[8:])
                yield conf, results

    def all_confs(self):
        cur = self._db.execute(f"SELECT {CONF_FIELDS} FROM conf")
        for row in cur:
            yield conf_from_row(row)

    def update_pred_scores(self, confs, scores):
        for conf, score in zip(confs, scores):
            if conf.id is None:
                conf = self._find_or_add_conf(conf)
            self._db.execute(
                "UPDATE conf SET pred_score = ? WHERE id = ?", (score, conf.id)
            )

    def all_conf_runs(self):
        cur = self._db.execute(
            f"SELECT {CONF_FIELDS}, loss FROM conf, run WHERE conf.id = run.conf_id"
        )
        for row in cur:
            conf = conf_from_row(row[:8])
            loss = row[8]
            yield conf, loss

    def has_runs(self, conf):
        id = self.find_conf_id(conf)
        if id is None:
            return False
        cur = self._db.execute(
            "SELECT 1 FROM run WHERE conf_id = ? LIMIT 1", (id,)
        )
        return cur.fetchone() is not None


def open_db(path):
    exists = os.path.exists(path)
    db = sqlite3.connect(path)
    if not exists:
        with open("results.sql") as schema:
            db.executescript(schema.read())
            db.commit()
    return db
