from datetime import datetime
import math
import numpy as np
from typing import Optional


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
        model_spec: str,
        weights: int,
        steps: int,
        train_time_s: float,
        learning_rate: float,
        regularization: float,
        train_sample_len: int,
        train_batch: int,
        total_data: int,
        loss: float,
        test_sample_len: int,
        test_batch: int,
        test_poisoned: bool,
        init_scale: float,
        planned_time_s: int,
        final_time_s: int,
        loss_model_v: int,
        loss_model_params: np.ndarray,
        checkpoint_id: Optional[int] = None,
    ):
        assert isinstance(timestamp, datetime)
        self.timestamp = timestamp

        self.model_spec = model_spec
        self.weights = weights

        # Time in seconds that the training was supposed to take. Usually
        # this is close to trin_time_s, but if a single optimization step
        # takes a long time, it could diverge.
        self.planned_time_s = planned_time_s

        # In case this report is for a checkpoint in the longer optimization,
        # this is an eventual planned optimization time. Otherwise
        # == planned_time_s.
        self.final_time_s = final_time_s

        # Actual training time
        self.train_time_s = train_time_s
        self.steps = steps

        # Metaparameters
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.train_sample_len = train_sample_len
        self.train_batch = train_batch
        self.init_scale = init_scale

        # Total available training data.
        self.total_data = total_data
        self.loss = loss
        self.loss_model_v = loss_model_v
        self.loss_model_params = loss_model_params

        self.test_sample_len = test_sample_len
        self.test_batch = test_batch
        self.test_poisoned = test_poisoned

        self.checkpoint_id: Optional[int] = checkpoint_id

    @property
    def train_data(self):
        return self.steps * self.train_sample_len * self.train_batch

    def __str__(self) -> str:
        if self.planned_time_s == self.final_time_s:
            planned_time = f"{self.planned_time_s} s"
        else:
            planned_time = f"{self.planned_time_s} s / {self.final_time_s} s"

        if self.loss_model_v == 1:
            c = 2**self.loss_model_params[0]
            expected_loss = f" (expected -> {c:.4f})"
        else:
            expected_loss = ""

        train_data = self.train_data / 1e6
        total_data = self.total_data / 1e6
        return f"""
Model: {self.model_spec}, {self.weights:,} weights
Loss: loss {self.loss:.4f}{expected_loss}
Training: {self.steps} steps, {self.train_time_s:.1f} s ({planned_time})
B {self.train_batch}  LEN {self.train_sample_len}  LR {self.learning_rate}
Training data: {train_data:.1f}M / {total_data:.1f}M
        """

    @staticmethod
    def from_csv_tuple(row):
        fields = {}
        fields["timestamp"] = datetime.fromisoformat(row[0])
        fields["model_spec"] = row[1]
        fields["weights"] = int(row[2])
        fields["steps"] = int(row[3])
        fields["train_time_s"] = float(row[4])
        fields["learning_rate"] = float(row[5])
        fields["regularization"] = float(row[6])
        fields["train_sample_len"] = int(row[7])
        fields["train_batch"] = int(row[8])
        fields["total_data"] = int(row[10])
        fields["loss"] = float(row[11])
        fields["test_sample_len"] = int(row[12])
        fields["test_batch"] = int(row[13])
        fields["test_poisoned"] = bool(int(row[14]))
        fields["init_scale"] = 1.0 if len(row < 16) else float(row[15])
        fields["planned_time_s"] = (
            _round_time(fields["train_time_s"])
            if len(row < 17)
            else int(row[16])
        )
        fields["final_time_s"] = (
            fields["planned_time_s"] if len(row) < 18 else int(row[17])
        )
        fields["loss_model_v"] = 0 if len(row) < 19 else int(row[18])
        if fields["loss_model_v"] > 0:
            assert len(row) >= 20
        fields["loss_model_params"] = (
            None
            if fields["loss_model_v"] == 0
            else np.fromiter(map(float, row[19].split(",")), dtype=np.float32)
        )
        fields["checkpoint_id"] = None if len(row) < 21 else row[20]

        return TrainingRecord(**fields)

    def csv_tuple(self) -> tuple:
        if self.loss_model_v > 0:
            loss_model_params = ",".join(map(str, self.loss_model_params))
        else:
            loss_model_params = None
        return (
            self.timestamp.isoformat(timespec="seconds"),
            self.model_spec,
            self.weights,
            self.steps,
            self.train_time_s,
            self.learning_rate,
            self.regularization,
            self.train_sample_len,
            self.train_batch,
            self.train_data,
            self.total_data,
            self.loss,
            self.test_sample_len,
            self.test_batch,
            1 if self.test_poisoned else 0,
            self.init_scale,
            self.planned_time_s,
            self.final_time_s,
            self.loss_model_v,
            loss_model_params,
            self.checkpoint_id
        )
