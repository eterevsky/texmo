from collections.abc import Iterable
import csv
import math
import numpy as np
import os
import sqlite3

from .configuration import (
    CONF_FIELDS,
    Configuration,
    conf_from_record,
    conf_from_row,
    conf_is_valid,
    INF,
    Template,
)
from .confresults import Run
from . import latency
from .record import TrainingRecord
from .model2 import build_model


def _pack_step_loss(step_loss):
    if step_loss is None:
        return None
    return np.array(step_loss, dtype=np.float32).tobytes()


def _unpack_step_loss(blob):
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


class ResultDB(object):
    def __init__(self, path=None):
        if path is None:
            path = ":memory:"
        exists = path != ":memory:" and os.path.exists(path)
        self._db = sqlite3.connect(path)
        if not exists:
            schema_path = os.path.join(
                os.path.dirname(__file__), "persistent-db.sql"
            )
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()

    def find_or_add_conf(self, conf: Configuration) -> int:
        """Finds the conf in the db and returns the configuration id."""
        spec = str(conf.model)

        conf_tuple = (
            spec,
            conf.lr,
            conf.sample_len,
            conf.batch,
            conf.regularization,
            conf.init_scale,
            conf.t,
            conf.model.weights,
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
            return rows[0][0]
        else:
            cur = self._db.execute(
                """
                INSERT INTO conf (spec, lr, sample_len, batch, regularization,
                                init_scale, t, weights)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                conf_tuple,
            )
            return cur.lastrowid

    def add_record(
        self,
        record: TrainingRecord,
        step_loss: Iterable[float] = None,
        commit: bool = True,
        skip_invalid: bool = False,
    ):
        conf = conf_from_record(record)
        if not conf_is_valid(conf):
            if skip_invalid:
                return
            raise Exception(f"Invalid configuration: {conf}")

        id = self.find_or_add_conf(conf)
        row = {
            "conf_id": id,
            "timestamp": record.timestamp,
            "test_sample_len": record.test_sample_len,
            "test_batch": record.test_batch,
            "loss": record.loss,
            "step_loss": _pack_step_loss(step_loss),
        }
        if math.isnan(record.loss) or record.loss is None:
            row["loss"] = INF
        self._db.execute(
            """
            INSERT INTO run(conf_id, timestamp, test_sample_len, test_batch, loss, step_loss)
            VALUES(:conf_id, :timestamp, :test_sample_len, :test_batch, :loss, :step_loss)
            """,
            row,
        )
        if commit:
            with latency.timer("ResultDB.add_record-commit"):
                self._db.commit()

    def get_confs_runs(self, template=None, load_step_loss=False):
        """Iterates through all confs that match template."""
        if template is None:
            template = Template()

        conditions = []
        bindings = []

        def _add_constraint(name, bounds):
            if bounds is None:
                return
            lo, hi = bounds
            if lo == hi:
                conditions.append(f"{name} = ?")
                bindings.append(lo)
            else:
                assert lo < hi
                conditions.append(f"{name} >= ?")
                conditions.append(f"{name} <= ?")
                bindings.append(lo)
                bindings.append(hi)

        _add_constraint("lr", template.lr)
        _add_constraint("sample_len", template.sample_len)
        _add_constraint("batch", template.batch)
        _add_constraint("regularization", template.regularization)
        _add_constraint("init_scale", template.init_scale)
        _add_constraint("t", template.t)

        condition = " AND ".join(conditions)
        if condition != "":
            condition = " AND " + condition

        query = (
            f"SELECT conf.id, {CONF_FIELDS}, run.loss, run.step_loss "
            + "FROM conf, run "
            + f"WHERE conf.id = run.conf_id{condition}"
        )

        cur = self._db.execute(query, bindings)

        for row in cur:
            try:
                model = build_model(row[1])
            except KeyError:
                continue
            if template.match_model(model):
                conf = conf_from_row(row[1:8])
                run = Run(row[8], _unpack_step_loss(row[9]))
                if conf_is_valid(conf):
                    yield row[0], conf, run

    def get_runs_with_step_loss(self):
        cur = self._db.execute(
            "SELECT loss, step_loss FROM run WHERE step_loss IS NOT NULL"
        )

        for row in cur:
            yield Run(row[0], _unpack_step_loss(row[1]))


def import_from_csv(result_db, filename):
    with open(filename) as csvfile:
        for row in csv.reader(csvfile):
            record = TrainingRecord.from_csv_tuple(row)
            result_db.add_record(record, commit=False, skip_invalid=True)
    result_db._db.commit()
