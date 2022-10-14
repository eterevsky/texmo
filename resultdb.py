import csv
import math
import os
import sqlite3

from configuration import (
    CONF_FIELDS,
    conf_from_record,
    conf_from_row,
    conf_is_valid,
    INF,
)
import latency
from record import TrainingRecord
from spec import ObsoleteSpec


class ResultDB(object):
    def __init__(self, path=None):
        exists = path != ":memory:" and os.path.exists(path)
        self._db = sqlite3.connect(path)
        if not exists:
            with open("persistent-db.sql") as schema:
                self._db.executescript(schema.read())
                self._db.commit()

    def _find_or_add_conf(self, conf):
        """Finds the conf in the db and returns the copy with populated id."""
        spec_str = str(conf.spec)

        conf_tuple = (
            spec_str,
            conf.lr,
            conf.sample_len,
            conf.batch,
            conf.regularization,
            conf.init_scale,
            conf.t,
            conf.spec.weights(),
        )

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
              AND weights = ?
            """,
            conf_tuple,
        )
        rows = cur.fetchall()
        assert len(rows) <= 1
        if rows:
            id = rows[0][0]
            assert conf.id is None or id == conf.ids
            return conf._replace(id=id)

        cur = self._db.execute(
            """
            INSERT INTO conf (spec, lr, sample_len, batch, regularization,
                              init_scale, t, weights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            conf_tuple,
        )
        id = cur.lastrowid
        return conf._replace(id=id)

    def add_record(self, record, commit=True, skip_invalid=False):
        if skip_invalid:
            try:
                conf = conf_from_record(record)
            except ObsoleteSpec:
                return
            if not conf_is_valid(conf):
                return
        else:
            conf = conf_from_record(record)
            try:
                assert conf_is_valid(conf)
            except AssertionError:
                import sys
                print('conf:', conf, file=sys.stderr)
                raise

        conf = self._find_or_add_conf(conf)
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
        if commit:
            with latency.timer("add_record-commit"):
                self._db.commit()

    def get_confs_runs(self, init_conf, vary):
        """Iterates through all confs that match init_conf+vary."""
        conditions = []
        bindings = []

        if "lr" not in vary:
            conditions.append("lr = ?")
            bindings.append(init_conf.lr)

        if "len" not in vary:
            conditions.append("sample_len = ?")
            bindings.append(init_conf.sample_len)

        if "batch" not in vary:
            conditions.append("batch = ?")
            bindings.append(init_conf.batch)

        if "reg" not in vary:
            conditions.append("regularization = ?")
            bindings.append(init_conf.regularization)

        if "init" not in vary:
            conditions.append("init_scale = ?")
            bindings.append(init_conf.init_scale)

        if "time" not in vary:
            conditions.append("t = ?")
            bindings.append(init_conf.t)

        condition = " AND ".join(conditions)
        if condition != "":
            condition = "AND " + condition

        cur = self._db.execute(
            f"""SELECT {CONF_FIELDS}, run.loss FROM conf, run
                WHERE conf.id = run.conf_id
                      {condition}
            """,
            bindings,
        )

        for row in cur:
            yield conf_from_row(row[:8]), row[8]


def import_from_csv(result_db, filename):
    with open(filename) as csvfile:
        for row in csv.reader(csvfile):
            record = TrainingRecord.from_csv_tuple(row)
            result_db.add_record(record, commit=False, skip_invalid=True)
    result_db._db.commit()


if __name__ == "__main__":
    record_db = ResultDB("results/results-3090.sqlite")
    import_from_csv(record_db, "log-3090.csv")