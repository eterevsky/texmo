from datetime import datetime
from typing import Dict, Union, Tuple


class TrainingRecord(object):
    def __init__(
        self,
        timestamp: Union[datetime, str],
        model_spec: str,
        weights: Union[int, str],
        steps: Union[int, str],
        train_time_s: Union[float, str],
        learning_rate: Union[float, str],
        regularization: Union[float, str],
        train_sample_len: Union[int, str],
        train_batch: Union[int, str],
        total_data: Union[int, str],
        loss: Union[float, str],
        test_sample_len: Union[int, str],
        test_batch: Union[int, str],
        test_poisoned: Union[bool, str],
    ):
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(timestamp)
        self.timestamp = timestamp
        self.model_spec = model_spec
        self.weights = int(weights)
        self.steps = int(steps)
        self.train_time_s = float(train_time_s)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.train_sample_len = int(train_sample_len)
        self.train_batch = int(train_batch)
        self.total_data = int(total_data)
        self.loss = float(loss)
        self.test_sample_len = int(test_sample_len)
        self.test_batch = int(test_batch)
        if isinstance(test_poisoned, str):
            test_poisoned = bool(int(test_poisoned))
        self.test_poisoned = test_poisoned

    @property
    def train_data(self):
        return self.steps * self.train_sample_len * self.train_batch

    def csv_tuple(self) -> Tuple:
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
            1 if self.test_poiseoned else 0,
        )

    def __str__(self) -> str:
        w = self.weights / 1000
        train_data = self.train_data / 1E6
        total_data = self.total_data / 1E6
        return f"""
Model: {self.model_spec}, {w:.0f}k weights
Loss: loss {self.loss:.4f}
Training: {self.steps} steps, {self.train_time_s} s
LR: {self.learning_rate}, R{self.regularization}
Training data: {train_data:.0f}M / {total_data:.0f}M
        """
