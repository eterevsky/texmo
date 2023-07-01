import csv
import logging
import math
import os
import sqlite3
from collections.abc import Iterable
from typing import Optional

import numpy as np

from . import latency
from .common import INF
from .configuration import (
    Configuration,
    Template,
    conf_from_dict,
    conf_is_valid,
    conf_to_dict,
)
from .model2 import Model2, build_model
from .record import TrainingRecord
from .run import Run


def _pack_ndarray(step_loss):
    step_loss = np.array(step_loss, dtype=np.float32)
    return step_loss.tobytes()


def _unpack_ndarray(blob):
    if blob is None:
        return None
    if len(blob) == 24:
        return np.frombuffer(blob, dtype=np.float64)
    else:
        return np.frombuffer(blob, dtype=np.float32)


def _dict_row_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def _build_loss_trend(step_loss, model_version, params):
    from .predict import build_loss_trend

    return build_loss_trend(step_loss, model_version, params)


class ResultDB(object):
    def __init__(self, path=None):
        if path is None:
            path = ":memory:"
        exists = path != ":memory:" and os.path.exists(path)
        logging.info(f"Connecting to results DB {path}")
        self._db = sqlite3.connect(path)
        if not exists:
            schema_path = os.path.join(
                os.path.dirname(__file__), "persistent-db.sql"
            )
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()
        self._db.row_factory = _dict_row_factory

    def find_or_add_conf(self, conf: Configuration) -> int:
        """Finds the conf in the db and returns the configuration id."""

        conf_dict = conf_to_dict(conf)
        conf_dict["weights"] = conf.model.weights

        cur = self._db.execute(
            """
            SELECT id FROM conf
            WHERE spec = :spec
              AND ntokens = :ntokens
              AND token_type = :token_type
              AND token_processing = :token_processing
              AND lr = :lr
              AND sample_len = :sample_len
              AND batch = :batch
              AND t = :t
              AND weights = :weights
            """,
            conf_dict,
        )
        rows = cur.fetchall()
        assert len(rows) <= 1
        if rows:
            return rows[0]["id"]
        else:
            cur = self._db.execute(
                """
                INSERT INTO conf (ntokens, token_type, token_processing, spec, lr, sample_len, batch, t, weights)
                VALUES (:ntokens, :token_type, :token_processing, :spec, :lr, :sample_len, :batch, :t, :weights)
                """,
                conf_dict,
            )
            return cur.lastrowid

    def add_record(
        self,
        record: TrainingRecord,
        run: Run,
        commit: bool = True,
        skip_invalid: bool = False,
    ):
        conf = record.conf
        if not conf_is_valid(conf):
            if skip_invalid:
                return
            raise Exception(f"Invalid configuration: {conf}")

        conf_id = self.find_or_add_conf(conf)
        row = {
            "conf_id": conf_id,
            "timestamp": record.timestamp,
            "test_sample_len": record.test_sample_len,
            "test_batch": record.test_batch,
            "loss": run.loss,
            "step_loss": _pack_ndarray(run.step_loss),
            "loss_model_v": run.loss_trend.version
            if run.loss_trend is not None
            else None,
            "loss_model": _pack_ndarray(run.loss_trend.params())
            if run.loss_trend is not None
            else None,
            "checkpoint": run.checkpoint if run.checkpoint else None,
        }
        if math.isnan(record.loss) or record.loss is None:
            row["loss"] = INF
        self._db.execute(
            """
            INSERT INTO run(conf_id, timestamp, test_sample_len, test_batch,
                            loss, step_loss, loss_model_v, loss_model, checkpoint)
            VALUES(:conf_id, :timestamp, :test_sample_len, :test_batch,
                   :loss, :step_loss, :loss_model_v, :loss_model, :checkpoint)
            """,
            row,
        )
        if commit:
            with latency.timer("ResultDB.add_record-commit"):
                self._db.commit()

    def get_confs_runs(self, template=None):
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
        _add_constraint("t", template.t)

        condition = " AND ".join(conditions)
        if condition != "":
            condition = "AND " + condition

        query = f"""
            SELECT conf.id AS conf_id,
                   ntokens,
                   token_type,
                   token_processing,
                   spec,
                   lr,
                   sample_len,
                   batch,
                   t,
                   run.id AS run_id,
                   run.loss AS loss,
                   run.step_loss AS step_loss,
                   run.loss_model_v AS loss_model_v,
                   run.loss_model AS loss_model
            FROM conf, run
            WHERE conf.id = run.conf_id {condition}
        """

        cur = self._db.execute(query, bindings)

        for row in cur:
            try:
                model = build_model(row["ntokens"], row["spec"])
            except KeyError:
                continue
            if template.match_model(model):
                conf = conf_from_dict(row)
                step_loss = _unpack_ndarray(row["step_loss"])

                loss_trend = _build_loss_trend(
                    step_loss,
                    row["loss_model_v"],
                    _unpack_ndarray(row["loss_model"]),
                )

                run = Run(
                    id=row["run_id"],
                    step_loss=step_loss,
                    loss=row["loss"],
                    loss_trend=loss_trend,
                )
                assert run.loss > 0.1
                if conf_is_valid(conf):
                    yield row["conf_id"], conf, run

    def get_runs_with_step_loss(self) -> Iterable[Run]:
        cur = self._db.execute(
            """
            SELECT loss, step_loss FROM run
            WHERE step_loss IS NOT NULL
              AND length(step_loss) > 0
            """
        )

        for row in cur:
            yield Run(row[0], _unpack_ndarray(row[1]))

    def best_checkpoint_loss(self, model: Model2) -> float:
        cur = self._db.execute(
            """
            SELECT MIN(run.loss) AS loss FROM conf, run
            WHERE conf.spec = :spec
              AND run.conf_id = conf.id
              AND run.checkpoint IS NOT NULL
            """,
            {"spec": str(model)},
        )

        res = cur.fetchall()[0]["loss"]
        return INF if res is None else res

    # def get_checkpoints(self) -> Iterable[tuple[Configuration, Run]]:
    #     cur = self._db.execute(
    #         """
    #         SELECT
    #         """
    #     )


def import_from_csv(result_db, filename):
    with open(filename) as csvfile:
        for row in csv.reader(csvfile):
            record = TrainingRecord.from_csv_tuple(row)
            result_db.add_record(record, commit=False, skip_invalid=True)
    result_db._db.commit()
