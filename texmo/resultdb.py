import logging
import math
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from statistics import StatisticsError, median
from typing import Optional
from urllib.parse import urlparse

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


class ConfScore(object):
    def __init__(
        self,
        conf_id: int,
        conf: Configuration2,
        median_score: float | None,
        system: str,
        median_time: float | None,
    ):
        self.conf_id: int = conf_id
        self.conf: Configuration2 = conf
        self.median_score: float | None = median_score
        self.system: str = system
        self.median_time: float | None = median_time


class ResultDB(object):
    @staticmethod
    def from_args(db: Optional[str]) -> "ResultDB":
        return ResultDB(db)

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = ":memory:"
        exists = path != ":memory:" and os.path.exists(path)
        if path != ":memory:":
            logging.info(f"Connecting to results DB {path}")
        self._db = sqlite3.connect(path, check_same_thread=False)

        if not exists:
            schema_path = os.path.join(os.path.dirname(__file__), "persistent-db.sql")
            with open(schema_path) as schema:
                self._db.executescript(schema.read())
                self._db.commit()

    def commit(self):
        self._db.commit()

    def _init_neighbors(self, cur: sqlite3.Cursor, conf: Configuration2, conf_id: int):
        with latency.timer("ResultDB._init_neighbors"):
            for neighbor in conf.neighbors():
                neighbor_id = self._find_or_add_conf(
                    cur, neighbor, init_neighbors=False
                )
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO neighbor (conf1_id, conf2_id)
                    VALUES (:conf1_id, :conf2_id)
                    """,
                    [
                        {"conf1_id": conf_id, "conf2_id": neighbor_id},
                        {"conf1_id": neighbor_id, "conf2_id": conf_id},
                    ],
                )
            cur.execute(
                """
                UPDATE conf
                SET has_neighbors = 1
                WHERE id = :conf_id
                """,
                {"conf_id": conf_id},
            )

    def _find_or_add_conf(
        self, cur: sqlite3.Cursor, conf: Configuration2, init_neighbors: bool
    ) -> int:
        conf_dict = conf.to_dict()
        conf_dict["weights"] = conf.model.weights
        cur.execute(
            """
            SELECT id, has_neighbors FROM conf
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
            conf_id, has_neighbors = rows[0]
            if init_neighbors and not has_neighbors:
                self._init_neighbors(cur, conf, conf_id)
            return conf_id
        else:
            cur = self._db.cursor()
            cur.execute(
                """
                INSERT INTO conf (spec, weights, lr, length, batch, steps, has_neighbors)
                VALUES (:spec, :weights, :lr, :length, :batch, :steps, 0)
                """,
                conf_dict,
            )
            conf_id = cur.lastrowid
            if init_neighbors:
                self._init_neighbors(cur, conf, conf_id)
            return conf_id

    def find_or_add_conf(self, conf: Configuration2, init_neighbors: bool) -> int:
        """Finds the conf in the db and returns the configuration id."""
        with latency.timer("ResultDB.find_or_add_conf"):
            cur = self._db.cursor()
            cur.execute("BEGIN TRANSACTION")
            conf_id = self._find_or_add_conf(cur, conf, init_neighbors)
            cur.execute("COMMIT")
            return conf_id

    def _add_run_execute(self, cur: sqlite3.Cursor, run_dict: dict):
        with latency.timer("ResultDB._add_run_execute"):
            cur.execute(
                """
                INSERT INTO run(conf_id, system, train_time, timestamp, loss, step_loss,
                                loss_model_v, loss_model)
                VALUES (:conf_id, :system, :train_time, :timestamp, :loss, :step_loss,
                        :loss_model_v, :loss_model)
                """,
                run_dict,
            )

    def _update_median_score(self, cur: sqlite3.Cursor, conf_id: int):
        with latency.timer("ResultDB._update_median_score"):
            cur.execute(
                """
                SELECT loss FROM run WHERE conf_id = :conf_id
                """,
                {"conf_id": conf_id},
            )
            try:
                median_score = median(row[0] for row in cur)
            except StatisticsError:
                median_score = None
            cur.execute(
                """
                UPDATE conf
                SET median_score = :median_score
                WHERE id = :conf_id
                """,
                {"median_score": median_score, "conf_id": conf_id},
            )

    def _update_median_time(self, cur: sqlite3.Cursor, conf_id: int, system: str):
        with latency.timer("ResultDB._update_median_time"):
            cur.execute(
                """
                SELECT train_time
                FROM run
                WHERE conf_id = :conf_id AND system = :system
                """,
                {"conf_id": conf_id, "system": system},
            )
            try:
                median_time = median(row[0] for row in cur)
            except StatisticsError:
                median_time = None
            if median_time is not None:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO conf_time(conf_id, system, median_time)
                    VALUES (:conf_id, :system, :median_time)
                    """,
                    {"median_time": median_time, "conf_id": conf_id, "system": system},
                )

    def _update_scores(self, cur: sqlite3.Cursor, conf_id: int, system: str):
        # cur.execute(
        #     """SELECT neighbor_id FROM neighbors WHERE conf_id = :conf_id""",
        #     {"conf_id": conf_id}
        # )
        # neighbors = [row[0] for row in cur]
        self._update_median_score(cur, conf_id)
        self._update_median_time(cur, conf_id, system)
        # self._update_neighbor_score(cur, conf_id)
        # for neighbor_id in neighbors:
        #     self._update_neighbor_score(cur, neighbor_id)

    def update_all_scores(self):
        with latency.timer("ResultDB.update_scores"):
            cur = self._db.cursor()

            cur.execute(
                "SELECT DISTINCT conf.id, system FROM conf, run WHERE conf.id = run.conf_id"
            )

            for row in cur.fetchall():
                conf_id = row[0]
                system = row[1]
                logging.info(f"{conf_id} {system}")
                self._update_scores(cur, conf_id, system)
            self.commit()

    def _add_run(
        self,
        conf: Configuration2,
        run: Run,
        conf_id: Optional[int],
        timestamp: Optional[datetime],
        update_neighbors: bool,
    ):
        assert run.checkpoint is None

        cur = self._db.cursor()
        cur.execute("BEGIN TRANSACTION")

        if conf_id is None:
            conf_id = self._find_or_add_conf(cur, conf, update_neighbors)

        if run.loss_trend is None:
            loss_model_v = None
            loss_model = None
        else:
            loss_model_v = run.loss_trend.version
            loss_model = _pack_ndarray(run.loss_trend.params())

        timestamp = timestamp.isoformat() if timestamp else None

        run_dict = {
            "conf_id": conf_id,
            "system": run.system,
            "train_time": run.train_time,
            "timestamp": timestamp,
            "loss": INF if math.isnan(run.loss) or run.loss is None else run.loss,
            "step_loss": _pack_ndarray(run.step_loss),
            "loss_model_v": loss_model_v,
            "loss_model": loss_model,
        }

        self._add_run_execute(cur, run_dict)
        self._update_scores(cur, conf_id, run.system)
        cur.execute("COMMIT")

    def add_run(
        self,
        conf: Configuration2,
        run: Run,
        conf_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        update_neighbors: bool = True,
    ):
        with latency.timer("ResultDB.add_run"):
            self._add_run(conf, run, conf_id, timestamp, update_neighbors)

    def top_confs(
        self,
        system: str,
        max_weights: float | None = None,
        max_time: float | None = None,
        limit: int = None,
        sorted: bool = True,
    ) -> Iterable[ConfScore]:
        conditions = ["median_score IS NOT NULL"]
        params = {"system": system}

        if max_weights is not None:
            conditions.append("weights <= :max_weight")
            params["max_weight"] = max_weights

        if max_time is not None:
            conditions.append("median_time <= :max_time")
            params["max_time"] = max_time

        where = "WHERE " + " AND ".join(conditions)

        if limit is None:
            limit = ""
        else:
            params["limit"] = limit
            limit = "LIMIT :limit"

        order = "ORDER BY median_score ASC" if sorted else ""

        cur = self._db.execute(
            f"""
            SELECT conf.id, spec, lr, length, batch, steps, median_score,
                (SELECT median_time FROM conf_time
                 WHERE conf_id = conf.id AND system = :system) AS median_time
            FROM conf {where} {order} {limit}
            """,
            params,
        )

        for row in cur:
            conf_id = row[0]
            model = build_model(row[1])
            conf = Configuration2(
                model, lr=row[2], length=row[3], batch=row[4], steps=row[5]
            )
            yield ConfScore(
                conf_id, conf, median_score=row[6], system=system, median_time=row[7]
            )

    def total_runs(self) -> int:
        cur = self._db.execute("SELECT COUNT(*) AS count FROM run")
        total = cur.fetchone()["count"]
        assert isinstance(total, int)
        return total

    def get_top_by_median_all_systems(
        self, max_weights: int, limit: int
    ) -> Iterable[tuple[int, Configuration2, float]]:
        cur = self._db.execute(
            """
            SELECT conf.id, spec, lr, length, batch, steps, median_score
            FROM conf
            WHERE weights <= :max_weights
            ORDER BY median_score ASC
            LIMIT :limit
            """,
            {"max_weights": max_weights, "limit": limit},
        )

        for row in cur:
            conf_id = row[0]
            model = build_model(row[1])
            conf = Configuration2(model, row[2], row[3], row[4], row[5])
            yield conf_id, conf, row[6]

    def get_confs_runs(
        self, with_timestamps: bool = False
    ) -> Iterable[tuple[int, Configuration2, Run]]:
        cur = self._db.cursor()
        cur.execute(
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
                conf_id = row[0]

                model = build_model(row[1])
                conf = Configuration2(model, row[2], row[3], row[4], row[5])

                step_loss = _unpack_ndarray(row[11])
                loss_trend = _build_loss_trend(
                    step_loss,
                    row[12],
                    _unpack_ndarray(row[13]),
                )

                run = Run(
                    id=row[6],
                    step_loss=step_loss,
                    loss=row[10],
                    loss_trend=loss_trend,
                    train_time=row[8],
                    checkpoint=row[14],
                    system=row[7],
                )

                if with_timestamps:
                    if row[9] is None:
                        timestamp = None
                    else:
                        timestamp = datetime.fromisoformat(row[9])
                    yield conf_id, conf, run, timestamp
                else:
                    yield conf_id, conf, run

    def check_run_exists(
        self, conf: Configuration2, run: Run, timestamp: datetime
    ) -> bool:
        """Check if a run with the same configuration and timestamp exists."""
        conf_id = self.find_or_add_conf(conf, init_neighbors=False)
        timestamp = timestamp.isoformat()
        cur = self._db.cursor()
        cur.execute(
            """
            SELECT 1
            FROM run
            WHERE conf_id = :conf_id
            AND timestamp = :timestamp
            AND system = :system
            """,
            {"conf_id": conf_id, "timestamp": timestamp, "system": run.system},
        )
        return cur.fetchone() is not None
