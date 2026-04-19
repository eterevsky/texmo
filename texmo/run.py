"""Intermediate training losses for a single training run."""

from typing import Optional

import numpy as np

from .common import ttoa3
from .predict.loss_trend import LossTrendBase
from .tokens import TokenSet


class Checkpoint:
    def __init__(self, step: int, time: float, weights: dict):
        self.step = step
        self.time = time
        self.weights = weights
        self.loss = None

    def make_run(self, full_run: Run) -> Run:
        return Run(full_run.system, step_loss=full_run.step_loss[:self.step], loss=self.loss, train_time=self.time)


class Run(object):
    def __init__(
        self,
        system: str,
        id: Optional[int] = None,
        step_loss: Optional[list[float]] = None,
        loss: Optional[float] = None,
        loss_trend: LossTrendBase = None,
        train_time: Optional[float] = None,
    ):
        assert isinstance(system, str)
        self.system: str = system

        self.id: Optional[int] = id

        # Training loss per step, calculated on the training batch.
        if step_loss is None:
            step_loss = []
        self.step_loss: list[float] = step_loss

        # A model for loss per step
        assert loss_trend is None or isinstance(loss_trend, LossTrendBase)
        self.loss_trend: Optional[LossTrendBase] = loss_trend

        # Final loss, evaluated on the test set.
        self.loss: float = loss
        self.train_time: Optional[float] = train_time

        # step -> Checkpoint
        self.checkpoints: dict[int, Checkpoint] = {}

    @staticmethod
    def from_dict(d: dict):
        loss_trend = (
            LossTrendBase.from_dict(d["loss_trend"]) if d.get("loss_trend") else None
        )
        return Run(
            system=d["system"],
            id=d.get("id"),
            step_loss=d["step_loss"],
            loss=d["loss"],
            loss_trend=loss_trend,
            train_time=d["train_time"],
        )

    def __str__(self):
        loss = "None" if self.loss is None else f"{self.loss:.4f}"
        return f"{self.system} {loss} {ttoa3(self.train_time)}"

    def add_step(self, token_loss: float):
        self.step_loss.append(token_loss)

    def add_checkpoint(self, step: int, t: float, weights: dict):
        self.checkpoints[step] = Checkpoint(step, t, weights)

    @property
    def steps(self):
        return len(self.step_loss)

    def finalize(self, eval_loss: float, train_time: float | None):
        """Sets the eval loss and fits the loss trend model.

        Called after the model has been trained.
        """
        self.loss = eval_loss
        self.train_time = train_time
        self.step_loss = np.array(self.step_loss, dtype=np.float32)
        assert self.loss_trend is not None
        self.loss_trend.fit(self.step_loss)

    def report_recent_loss(self, tokenset: TokenSet):
        step = len(self.step_loss)
        loss = sum(self.step_loss[-10:]) / 10 if step >= 10 else self.step_loss[-1]
        byte_loss = loss / tokenset.avg_bytes_per_token
        return f"{step}  {byte_loss:.4f} b/B"

    def to_dict(self):
        d = {
            "step_loss": self.step_loss if type(self.step_loss) is list else self.step_loss.tolist(),
            "loss": self.loss,
            "loss_trend": self.loss_trend.to_dict() if self.loss_trend else None,
            "system": self.system,
            "train_time": self.train_time,
        }
        if self.id is not None:
            d["id"] = self.id
        return d

    @staticmethod
    def from_dict(d: dict):
        loss_trend = d["loss_trend"]
        if loss_trend:
            loss_trend = LossTrendBase.from_dict(loss_trend)
        return Run(
            system=d["system"],
            id=d.get("id"),
            step_loss=d["step_loss"],
            loss=d["loss"],
            loss_trend=loss_trend,
            train_time=d["train_time"],
        )
