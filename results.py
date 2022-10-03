import argparse
from collections import namedtuple
import csv
from itertools import chain
import math
import os
import sqlite3
from statistics import median, StatisticsError

import latency
from record import TrainingRecord
from spec import ModelSpec, ObsoleteSpec


INF = float("inf")
LRS = [
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
]


def neighbor_numbers(x):
    """Produce neighbor numbers from a close to exponent but readable table"""
    i = LRS.index(x)
    if i > 0:
        yield LRS[i - 1]
    if i < len(LRS) - 1:
        yield LRS[i + 1]


Configuration = namedtuple(
    "Configuration",
    [
        "id",
        "spec",
        "lr",
        "sample_len",
        "batch",
        "regularization",
        "init_scale",
        "t",
    ],
)


Results = namedtuple(
    "Results",
    [
        "score",
        "cluster_score",
        "num_runs",
    ],
)


def conf_from_record(record):
    spec = ModelSpec.parse(record.model_spec)
    return Configuration(
        None,
        spec,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.regularization,
        record.init_scale,
        record.time_round,
    )


# Database fields required to create a Configuration instance using
# conf_from_row
CONF_FIELDS = "id, spec, lr, sample_len, batch, regularization, init_scale, t"


def conf_from_row(row):
    """Create Configuration from a database row."""
    if row is None: return None
    spec = ModelSpec.parse(row[1])
    return Configuration(row[0], spec, *row[2:])


def conf_neighbors(conf, vary=()):
    for spec in conf.spec.neighbors(vary):
        yield conf._replace(spec=spec)

    if "batch" in vary:
        yield conf._replace(batch=conf.batch * 2)
        if conf.batch > 1:
            yield conf._replace(batch=conf.batch // 2)

    if "lr" in vary:
        for x in neighbor_numbers(conf.lr):
            yield conf._replace(lr=x)

    if "len" in vary:
        if conf.sample_len >= 4:
            yield conf._replace(sample_len=conf.sample_len // 2)
        yield conf._replace(sample_len=conf.sample_len * 2)

    if "regularization" in vary:
        for x in neighbor_numbers(conf.regularization):
            yield conf._replace(regularization=x)

    if "init_scale" in vary:
        for x in neighbor_numbers(conf.init_scale):
            yield conf._replace(init_scale=x)


VARY = {
    # Add an extra layer
    "layer": 1,
    # Change the type of a layer (dense, rec, gru, lstm)
    "type": 2,
    # Change the size of a layer (dense, rec, gru, lstm)
    "size": 3,
    # Introduce a suffix.2 layer or change the size of a suffix
    # or attention layer
    "suffix": 4,
    # Change layer type between suffix, attention, attention.pos
    "attn": 5,
    # Change batch size
    "batch": 6,
    # Learning rate
    "lr": 7,
    # Regualarization
    "reg": 8,
    # init_scale
    "init": 9,
    # Sample length
    "len": 10,
    # Training time
    "time": 11,
}


spec_neighbors = {}


def all_conf_neighbors(conf):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    if conf.spec in spec_neighbors:
        for neighbor_spec, vary in spec_neighbors[conf.spec]:
            yield conf._replace(spec=neighbor_spec), vary
    else:
        cache = []
        spec_neighbors[conf.spec] = cache
        for neighbor_spec, vary in conf.spec.all_neighbors():
            cache.append((neighbor_spec, VARY[vary]))
            yield conf._replace(spec=neighbor_spec), VARY[vary]

    yield conf._replace(batch=conf.batch * 2), VARY["batch"]
    if conf.batch > 1:
        yield conf._replace(batch=conf.batch // 2), VARY["batch"]

    for x in neighbor_numbers(conf.lr):
        yield conf._replace(lr=x), VARY["lr"]

    if conf.sample_len >= 4:
        yield conf._replace(sample_len=conf.sample_len // 2), VARY["len"]
    yield conf._replace(sample_len=conf.sample_len * 2), VARY["len"]

    for x in neighbor_numbers(conf.regularization):
        yield conf._replace(regularization=x), VARY["reg"]

    for x in neighbor_numbers(conf.init_scale):
        yield conf._replace(init_scale=x), VARY["init"]

    if conf.t > 1:
        yield conf._replace(t=conf.t // 2), VARY["time"]
    yield conf._replace(t=conf.t * 2), VARY["time"]


class ResultSet(object):
    def __init__(self, db=None, t=None, vary=()):
        self._t = t
        # t -> conf -> conf results
        self._vary = vary
        self._db = db
        self._vary_str = ", ".join(str(VARY[v]) for v in vary)

    @staticmethod
    def from_csv(filename, db=None, t=None, vary=()):
        result_set = ResultSet(db=db, t=t, vary=vary)
        with open(filename) as csvfile:
            for row in csv.reader(csvfile):
                record = TrainingRecord.from_csv_tuple(row)
                result_set.add_record(record, update_scores=False, commit=False)
        result_set._db.commit()
        return result_set

    def find_conf(self, conf_dict):
        res = self._db.execute(
            """
            SELECT id
            FROM conf
            WHERE spec = :spec
            AND lr = :lr
            AND sample_len = :sample_len
            AND batch = :batch
            AND regularization = :regularization
            AND init_scale = :init_scale
            AND t = :t
            """,
            conf_dict,
        )
        rows = res.fetchall()
        assert len(rows) <= 1
        return rows[0][0] if rows else None

    def _find_or_add_conf(self, conf):
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

        id = self.find_conf(conf_dict)
        if id is not None:
            return conf._replace(id=id)

        cursor = self._db.execute(
            """
            INSERT INTO conf(spec, lr, sample_len, batch, regularization, init_scale, t, weights)
            VALUES(:spec, :lr, :sample_len, :batch, :regularization, :init_scale, :t, :weights)
            """,
            conf_dict,
        )
        id = cursor.lastrowid
        return conf._replace(id=id)

    def _update_neighbors(self, conf, conf_id=None):
        if conf is not None:
            if conf_id is not None:
                assert conf.id == conf_id
            else:
                conf_id = conf.id

        cur = self._db.execute(
            f"""
            SELECT 1 FROM neighbor
            WHERE conf1_id = :id
              AND vary IN ({self._vary_str})
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

        for neighbor, vary in all_conf_neighbors(conf):
            neighbor = self._find_or_add_conf(neighbor)
            self._db.execute(
                """
                INSERT INTO neighbor(conf1_id, conf2_id, vary)
                VALUES (:conf1_id, :conf2_id, :vary)
                """,
                {"conf1_id": conf.id, "conf2_id": neighbor.id, "vary": vary},
            )

    def update_all_neighbors(self):
        self._db.execute("DELETE FROM neighbor")
        cur = self._db.execute(f"SELECT {CONF_FIELDS} FROM conf " +
                                "WHERE score IS NOT NULL OR cluster_score IS NOT NULL")
        for i, row in enumerate(cur):
            if i % 1000 == 0: print(i)
            conf = conf_from_row(row)
            self._update_neighbors(conf)
        self._db.commit()

    def update_all_scores(self):
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
        with latency.timer("update_cluster_score"):
            self._update_neighbors(conf=None, conf_id=conf_id)
            cur_runs = self._db.execute(
                "SELECT loss FROM run WHERE conf_id = ?", (conf_id,)
            )
            run_scores = [row[0] for row in cur_runs]
            self_score = median(run_scores) if run_scores else None
            cur_neighbors = self._db.execute(
                f"""
                SELECT run.loss
                FROM run, neighbor
                WHERE neighbor.conf1_id = ?
                AND neighbor.vary in ({self._vary_str})
                AND run.conf_id = neighbor.conf2_id
                AND run.loss IS NOT NULL
                """,
                [conf_id],
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
        cur = self._db.execute(
            f"""
            SELECT conf1_id, conf2_id
            FROM neighbor
            WHERE vary IN ({self._vary_str})
            """
        )
        for conf1_id, conf2_id in cur:
            conf1_neighbors = neighbors.get(conf1_id)
            if conf1_neighbors:
                conf1_neighbors.append(conf2_id)
            else:
                conf1_neighbors = [conf2_id]
                neighbors[conf1_id] = conf1_neighbors

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
        for conf_id, runs in runs.items():
            self_scores[conf_id] = median(runs)

        for conf_id, conf_neighbors in neighbors.items():
            neighbor_scores = (self_scores[neighbor]
                               for neighbor in conf_neighbors
                               if neighbor in self_scores)
            try:
                if conf_id in runs:
                    cluster_score = median(chain(neighbor_scores, runs[conf_id]))
                else:
                    cluster_score = median(neighbor_scores)
            except StatisticsError:
                cluster_score = None

            self._db.execute(
                """
                UPDATE conf SET score = ?, cluster_score = ?
                WHERE id = ?
                """, (self_scores.get(conf_id), cluster_score, conf_id))

        self._db.commit()

    def _update_scores(self, conf_id):
        with latency.timer("_update_scores"):
            cur = self._db.execute(
                "SELECT loss FROM run WHERE conf_id = ?", [conf_id]
            )
            score = median(row[0] for row in cur)
            self._db.execute(
                "UPDATE conf SET score = ? WHERE id = ?", [score, conf_id]
            )

            self._update_cluster_score(conf_id)

            # TODO: make neighbors symmetric
            neighbors = self._db.execute(
                "SELECT conf2_id FROM neighbor " +
                f"WHERE conf1_id = ? AND neighbor.vary in ({self._vary_str})",
                (conf_id,),
            )
            for (neighbor_id,) in neighbors:
                self._update_cluster_score(neighbor_id)

    def add_record(self, record, update_scores=True, commit=True):
        if self._t is not None and record.time_round != self._t:
            return

        try:
            conf = conf_from_record(record)
        except ObsoleteSpec:
            return
        if not conf.spec.is_valid() or conf.t is None or conf.t < 1:
            # print("invalid")
            return

        conf = self._find_or_add_conf(conf)
        self._update_neighbors(conf)
        row = {
            "conf_id": conf.id,
            "timestamp": record.timestamp,
            "test_sample_len": record.test_sample_len,
            "test_batch": record.test_batch,
            "loss": record.loss,
        }
        if math.isnan(record.loss) or record.loss is None:
            row["loss"] = INF
        self._db.execute(
            """
            INSERT INTO run(conf_id, timestamp, test_sample_len, test_batch, loss)
            VALUES(:conf_id, :timestamp, :test_sample_len, :test_batch, :loss)
            """,
            row,
        )
        if update_scores:
            self._update_scores(conf.id)
        if commit:
            with latency.timer("add_record-commit"):
                self._db.commit()

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

    def all_results(self):
        for results_for_t in self._conf_results.values():
            for cr in results_for_t.values():
                yield cr

    def results_for_t(self, t):
        results = self._conf_results.get(t, None)
        return () if results is None else results.values()

    def runs_count(self, t, max_weights=INF):
        cur = self._db.execute(
            """
            SELECT COUNT(*)
            FROM conf, run
            WHERE conf.id = run.conf_id
              AND conf.t = ?
              AND conf.weights <= ?
            """,
            [t, max_weights],
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
            f"SELECT {CONF_FIELDS} FROM conf " +
            "WHERE t = ? AND score IS NOT NULL " +
            "ORDER BY score LIMIT 1",
            (t,)
        )
        return conf_from_row(cur.fetchone())

    def top_confs(self, t, max_weights):
        with latency.timer("top_confs"):
            cur = self._db.execute(
                f"SELECT {CONF_FIELDS} FROM conf " +
                "WHERE t = ? AND weights <= ? AND score IS NOT NULL " +
                "ORDER BY score",
                (t, max_weights)
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
        cur = self._db.execute(f"SELECT {CONF_FIELDS} FROM conf WHERE id = ?", (conf_id,))
        return conf_from_row(cur.fetchone())

    def cluster_score(self, t, cr):
        conf_results = self._conf_results.get(t, {})

        if cr.cluster_score is not None:
            return cr.cluster_score

        self_score = cr.score if cr.scores else INF

        def iter_neighbors():
            for neighbor_conf in cr.neighbors:
                neighbor_results = conf_results.get(neighbor_conf)
                if neighbor_results is not None and neighbor_results.scores:
                    yield neighbor_results.score

        try:
            neighbor_score = median(
                chain(cr.scores, iter_neighbors())
            )
        except StatisticsError:
            neighbor_score = INF

        cluster_score = min(self_score, neighbor_score)
        cr.cluster_score = cluster_score
        return cluster_score

    def top_cluster_confs(self, t, max_weights, limit=None):
        limit_clause = f"LIMIT {limit}" if limit else "";
        cur = self._db.execute(
            f"""
            SELECT {CONF_FIELDS},
                   score,
                   cluster_score,
                   (SELECT COUNT(*) FROM run WHERE conf_id = conf.id)
            FROM conf
            WHERE t = ?
              AND weights <= ?
              AND cluster_score IS NOT NULL
            ORDER BY cluster_score
            """ + limit_clause,
            (t, max_weights)
        )
        for row in cur:
            conf = conf_from_row(row[:8])
            results = Results(*row[8:])
            yield conf, results

    @property
    def total_confs(self):
        return sum(len(r) for r in self._conf_results.values())

    @property
    def total_runs(self):
        return sum(len(cr.scores) for cr in self.all_results())


def open_db(path):
    exists = os.path.exists(path)
    db = sqlite3.connect(path)
    if not exists:
        with open("results.sql") as schema:
            db.executescript(schema.read())
            db.commit()
    return db


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database with configurations and results",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="LOG",
        help="a CSV log file with previous runs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Opening the database", args.db)
    db = open_db(args.db)
    # print("Importing the runs from", args.load)
    # result_set = ResultSet.from_csv(args.load, db=db)
    # print("Updating scores")
    # result_set.update_all_scores()
    print("Creating ResultSet")
    result_set = ResultSet(db=db)
    print("Updating neighbors")
    result_set.update_all_neighbors()

