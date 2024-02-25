"""Intermediate training losses for a single training run."""

from typing import Optional

import numpy as np

from .common import INF
from .tokens import TokenSet


class LossTrendBase(object):
    def fit(self, step_byte_loss: list[float]):
        raise NotImplementedError

    def predict(self, step: int) -> float:
        raise NotImplementedError

    def params(self) -> np.ndarray:
        raise NotImplementedError


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
        d = {
            "step_loss": self.step_loss,
            "loss": self.loss,
            "loss_trend": loss_trend
        }
        if self.id is not None:
            d["id"] = self.id
        return d
