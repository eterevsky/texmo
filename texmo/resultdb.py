import csv
import logging
import math
from datetime import datetime
import os
import sqlite3
from collections.abc import Iterable
from typing import Optional

import numpy as np

from . import latency
from .common import INF
from .configuration2 import Configuration2
from .model3 import build_model
from .run import Run


def _pack_ndarray(step_loss):
    step_loss = np.array(step_loss, dtype=np.float32)
    assert len(step_loss.shape) == 1
    return step_loss.tobytes()


def _unpack_ndarray(blob):
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _dict_row_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}


def _build_loss_trend(step_loss, model_version, params):
    from .predict import build_loss_trend

    return build_loss_trend(step_loss, model_version, params)


class ResultDB(object):
    def __init__(self, path: Optional[str], system_name: str):
        assert isinstance(system_name, str)
        self._system_name = system_name

        if path is None:
            path = ":memory:"
        exists = path != ":memory:" and os.path.exists(path)
        if path != ":memory:":
            logging.info(f"Connecting to results DB {path}")
        self._db = sqlite3.connect(path)

        if not exists:
            schema_path = os.path.join(os.path.dirname(__file__), "persistent-db.sql")
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()

        self._db.row_factory = _dict_row_factory

    def find_or_add_conf(self, conf: Configuration2) -> int:
        """Finds the conf in the db and returns the configuration id."""

        conf_dict = conf.to_dict()
        conf_dict["weights"] = conf.model.weights

        cur = self._db.execute(
            """
            SELECT id FROM conf
            WHERE spec = :spec
              AND lr = :lr
              AND length = :length
              AND batch = :batch
              AND steps = :steps
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
                INSERT INTO conf (spec, weights, lr, length, batch, steps)
                VALUES (:spec, :weights, :lr, :length, :batch, :steps)
                """,
                conf_dict,
            )
            return cur.lastrowid

    def add_run(
        self,
        conf: Configuration2,
        run: Run,
        conf_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        commit: bool = True,
    ):
        # TODO: Checkpoints are temporarily not supported
        assert run.checkpoint is None

        if conf_id is None:
            conf_id = self.find_or_add_conf(conf)

        if run.loss_trend is None:
            loss_model_v = None
            loss_model = None
        else:
            loss_model_v = run.loss_trend.version
            loss_model = _pack_ndarray(run.loss_trend.params())

        timestamp = timestamp.isoformat() if timestamp else None

        run_dict = {
            "conf_id": conf_id,
            "system": self._system_name,
            "train_time": run.train_time,
            "timestamp": timestamp,
            "loss": INF if math.isnan(run.loss) or run.loss is None else run.loss,
            "step_loss": _pack_ndarray(run.step_loss),
            "loss_model_v": loss_model_v,
            "loss_model": loss_model,
        }
        with latency.timer("ResultDB.add_run-execute"):
            self._db.execute(
                """
                INSERT INTO run(conf_id, system, train_time, timestamp, loss, step_loss,
                                loss_model_v, loss_model)
                VALUES (:conf_id, :system, :train_time, :timestamp, :loss, :step_loss,
                        :loss_model_v, :loss_model)
                """,
                run_dict,
            )
        if commit:
            with latency.timer("ResultDB.add_run-commit"):
                self._db.commit()

    def total_runs(self) -> int:
        cur = self._db.execute("SELECT COUNT(*) AS count FROM run")
        total = cur.fetchone()["count"]
        assert isinstance(total, int)
        return total
    
    def get_confs_runs(self) -> Iterable[tuple[int, Configuration2, Run]]:
        cur = self._db.execute(
            """
            SELECT conf.id AS conf_id,
                   spec,
                   lr,
                   length,
                   batch,
                   steps,
                   run.id AS run_id,
                   system,
                   train_time,
                   timestamp,
                   loss,
                   step_loss,
                   loss_model_v,
                   loss_model,
                   checkpoint
            FROM conf, run
            WHERE conf.id = run.conf_id 
            """
        )

        for row in cur:
            with latency.timer("ResultDB.get_confs_runs-row"):
                conf_id = row["conf_id"]

                model = build_model(row["spec"])
                conf = Configuration2(model, row["lr"], row["length"], row["batch"], row["steps"])

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
                    train_time=row["train_time"],
                    checkpoint=row["checkpoint"],
                )

                yield conf_id, conf, run
