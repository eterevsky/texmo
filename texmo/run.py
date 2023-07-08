"""Intermediate training losses for a single training run."""

from typing import Optional

import numpy as np

from .common import INF
from .configuration import Configuration
from .tokens import TokenSet


class LossTrendBase(object):
    def fit(self, step_byte_loss: list[float]):
        raise NotImplementedError

    def predict(self, step: int) -> float:
        raise NotImplementedError


class Run(object):
    def __init__(
        self,
        id: Optional[int] = None,
        step_loss: Optional[list[float]] = None,
        loss: Optional[float] = None,
        loss_trend: LossTrendBase = None,
        checkpoint: str = None,
    ):
        self.id: Optional[int] = id

        # Training loss per step, calculated on the training batch.
        if step_loss is None:
            step_loss = []
        self.step_loss: list = step_loss
        self.step_byte_loss: list = []

        # A model for loss per step
        # assert loss_trend is not None
        self.loss_trend: LossTrendBase = loss_trend

        # Final loss, evaluated on the test set.
        self.loss: float = loss

        self.checkpoint: str = checkpoint

    def add_step(self, token_loss: float, byte_loss: float):
        self.step_loss.append(token_loss)
        self.step_byte_loss.append(byte_loss)

    @property
    def steps(self):
        return len(self.step_loss)

    def finalize(self, eval_loss: float):
        """Sets the eval loss and fits the loss trend model.

        Called after the model has been trained.
        """
        self.loss = eval_loss
        self.step_loss = np.array(self.step_loss, dtype=np.float32)
        assert self.loss_trend is not None
        self.loss_trend.fit(self.step_byte_loss)

    def report_recent_loss(self, token_set: TokenSet):
        step = len(self.step_loss)
        loss = sum(self.step_loss[-10:]) / 10 if step >= 10 else self.step_loss[-1]
        byte_loss = token_set.byte_loss(loss)
        return f"{step}  {loss:.4f} b/token  {byte_loss:.4f} b/byte"

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

    # @staticmethod
    # def from_dict(d):
    #     step_loss = d.get("step_loss")
    #     if step_loss is not None:
    #         step_loss = np.array(step_loss, dtype=np.float32)
    #     loss_trend = build_loss_trend(d.get("loss_trend"))
    #     return Run(
    #         id=d.get("id"),
    #         step_loss=step_loss,
    #         loss=d["loss"],
    #         loss_trend=loss_trend,
    #     )
