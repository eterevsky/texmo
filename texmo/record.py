import json
import math
from datetime import datetime
from typing import Optional

import numpy as np

from .configuration import (
    Configuration,
    conf_from_dict,
    conf_to_dict,
    conf_to_string,
)
from .common import INF


def _round_time(t):
    log = math.log2(t)
    ilog = round(log)
    if abs(log - ilog) < 0.1:
        return 2**ilog
    else:
        return None


class TrainingRecord(object):
    def __init__(
        self,
        timestamp: datetime,
        conf: Configuration,
        steps: int,
        train_time_s: float,
        total_data: int,
        loss: float,
        test_sample_len: int,
        test_batch: int,
        test_poisoned: bool,
        planned_time_s: int,
        final_time_s: int,
        loss_model_v: int,
        loss_model_params: np.ndarray,
        regularization: float,
        checkpoint_id: Optional[int] = None,
        avg_sample_time: float = None,
        first_step_time: float = None,
        avg_step_time: float = None,
    ):
        assert isinstance(timestamp, datetime)

        self._dict = {
            "timestamp": timestamp.isoformat(),
            "conf": conf_to_dict(conf),
            "weights": conf.model.weights,
            "regularization": regularization,
            # Time in seconds that the training was supposed to take. Usually
            # this is close to train_time_s, but if a single optimization step
            # takes a long time, it could diverge.
            "planned_time_s": planned_time_s,
            # In case this report is for a checkpoint in the longer optimization,
            # this is an eventual planned optimization time. Otherwise
            # == planned_time_s.
            "final_time_s": final_time_s,
            "train_time_s": train_time_s,
            "steps": steps,
            # Total available training data.
            "total_data": total_data,
            "loss": loss,
            "loss_model_v": loss_model_v,
            "loss_model_params": list(map(float, loss_model_params)),
            "test_sample_len": test_sample_len,
            "test_batch": test_batch,
            "test_poisoned": test_poisoned,
            "checkpoint_id": checkpoint_id,
            "avg_sample_time": avg_sample_time,
            "first_step_time": first_step_time,
            "avg_step_time": avg_step_time,
        }

    @property
    def train_data(self):
        return (
            self._dict["steps"]
            * self._dict["conf"]["sample_len"]
            * self._dict["conf"]["batch"]
        )

    @property
    def conf(self):
        return conf_from_dict(self._dict["conf"])

    @property
    def planned_time_s(self):
        return self._dict["planned_time_s"]

    @property
    def train_time_s(self):
        return self._dict["train_time_s"]

    @property
    def timestamp(self) -> datetime:
        return datetime.fromisoformat(self._dict["timestamp"])

    @property
    def loss(self) -> float:
        return self._dict["loss"]

    @property
    def test_sample_len(self) -> int:
        return self._dict["test_sample_len"]

    @property
    def test_batch(self) -> int:
        return self._dict["test_batch"]

    @property
    def avg_sample_time(self) -> float:
        return self._dict["avg_sample_time"]

    @property
    def first_step_time(self) -> float:
        return self._dict["first_step_time"]

    @property
    def avg_step_time(self) -> float:
        return self._dict["avg_step_time"]

    @property
    def invalid_time(self) -> bool:
        return self._dict.get("invalid_time", False)

    def invalidate(self):
        self._dict["invalid_time"] = True

    def __str__(self) -> str:
        planned_time_s = self._dict["planned_time_s"]
        final_time_s = self._dict["final_time_s"]
        if planned_time_s == final_time_s:
            planned_time = f"{planned_time_s} s"
        else:
            planned_time = f"{planned_time_s} s / {final_time_s} s"

        if self._dict["loss_model_v"] == 1:
            c = 2 ** self._dict["loss_model_params"][0]
            expected_loss = f" (expected -> {c:.4f})"
        else:
            expected_loss = ""

        train_data = self.train_data / 1e6
        total_data = self._dict["total_data"] / 1e6

        conf_str = conf_to_string(self.conf)

        loss = self._dict["loss"]
        steps = self._dict["steps"]
        train_time_s = self._dict["train_time_s"]

        avg_sample_time = self.avg_sample_time * 1000
        first_step_time = self.first_step_time * 1000
        if self.avg_step_time is not None:
            avg_step_time = self.avg_step_time * 1000
        else:
            avg_step_time = float("nan")

        if self.invalid_time:
            invalid = "INVALID "
        else:
            invalid = ""

        return f"""{conf_str}
Loss: {loss:.4f}{expected_loss} Training data: {train_data:.1f} M / {total_data:.1f} M
Training for {train_time_s:.1f} s ({invalid}{planned_time} s). Steps {steps}, sample {avg_sample_time:.2f} ms, step {first_step_time:.2f}|{avg_step_time:.2f} ms"""

    def jsonl(self):
        json.dumps(self._dict)
