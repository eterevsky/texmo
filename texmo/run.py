"""Intermediate training losses for a single training run."""

from typing import Optional

from math import isnan
import numpy as np

from .common import INF, ttoa3
from .tokens import TokenSet
from .predict.loss_trend import LossTrendBase


class Run(object):
    def __init__(
        self,
        system: str,
        id: Optional[int] = None,
        step_loss: Optional[list[float]] = None,
        loss: Optional[float] = None,
        loss_trend: LossTrendBase = None,
        train_time: Optional[float] = None,
        checkpoint: Optional[str] = None,
    ):
        assert isinstance(system, str)
        self.system: str = system

        self.id: Optional[int] = id

        # Training loss per step, calculated on the training batch.
        if step_loss is None:
            step_loss = []
        self.step_loss: list[float] = step_loss

        # A model for loss per step
        assert isinstance(loss_trend, LossTrendBase)
        self.loss_trend: LossTrendBase = loss_trend

        # Final loss, evaluated on the test set.
        self.loss: float = loss
        self.train_time: Optional[float] = train_time

        self.checkpoint: Optional[str] = checkpoint

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
        return f"{self.system} {self.loss:.4f} {ttoa3(self.train_time)}"

    def add_step(self, token_loss: float):
        self.step_loss.append(token_loss)

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
        loss_trend = self.loss_trend.to_dict() if self.loss_trend else self.loss_trend
        step_loss = np.minimum(self.step_loss, 1E10).tolist()
        train_time = self.train_time
        if any(isnan(l) for l in step_loss):
            step_loss = None
            loss_trend = None
            train_time = None
        loss = 1E10 if self.loss > 1E10 else self.loss
        if isnan(loss):
            loss = 1E10
        d = {
            "step_loss": step_loss,
            "loss": loss,
            "loss_trend": loss_trend,
            "system": self.system,
            "train_time": train_time,
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
