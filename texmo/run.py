"""Intermediate training losses for a single training run."""

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from . import latency
from .configuration import Configuration
from .predict_common import prediction_score
from .tokens import TokenSet


def _pred_log(c1, c2, eps, step):
    return c1 + c2 * step**eps


class LossTrend(object):
    """A simple model estimating training losses."""

    version = 1

    def __init__(self, c1=None, c2=None, eps=None):
        self._c1 = c1
        self._c2 = c2
        self._eps = eps

    def to_dict(self):
        return {
            "version": 1,
            "c1": self._c1,
            "c2": self._c2,
            "eps": self._eps,
        }

    def fit(self, losses):
        with latency.timer("StepLossPredictor.fit"):
            losses = np.log2(losses)

            if len(losses) == 0:
                self._c1 = 8
                self._c2 = 0
                self._eps = 0
                return

            start = max(round(0.55 * len(losses)), 2)
            steps = np.array(range(start, len(losses)))

            if len(steps) < 3:
                self._c1 = losses[-1]
                self._c2 = 0
                self._eps = 0
                return

            target_losses = np.array(losses[start:])

            def _score(x):
                c1, c2, eps = x
                prediction = _pred_log(c1, c2, eps, steps)
                return prediction_score(target_losses, prediction)

            init = np.array([1.0, 1.0, -1.0])
            res = minimize(_score, init)

            self._c1, self._c2, self._eps = res.x

    def predict(self, step):
        step = np.array(step)
        if self._eps > 0 or self._c2 == 0:
            prediction = np.array([self._c1] * step)
        else:
            prediction = _pred_log(self._c1, self._c2, self._eps, step)
        return 2**prediction

    def params(self) -> np.ndarray:
        return np.array([self._c1, self._c2, self._eps])


def build_loss_trend(step_loss, model_version, params):
    if params is None or model_version != 1:
        predictor = LossTrend()
        predictor.fit(step_loss)
        return predictor
    c1, c2, eps = params
    return LossTrend(c1, c2, eps)


def loss_trend_from_dict(loss_trend_dict: dict, step_loss: np.ndarray):
    if loss_trend_dict is None or loss_trend_dict["version"] != 1:
        loss_trend = LossTrend()
        loss_trend.fit(step_loss)
        return loss_trend
    return LossTrend(
        c1=loss_trend_dict["c1"],
        c2=loss_trend_dict["c2"],
        eps=loss_trend_dict["eps"],
    )


class Run(object):
    def __init__(
        self,
        id: Optional[int] = None,
        step_loss: Optional[list[float]] = None,
        loss: Optional[float] = None,
        loss_trend: Optional[LossTrend] = None,
        checkpoint: str = None,
    ):
        self.id: Optional[int] = id

        # Training loss per step, calculated on the training batch.
        if step_loss is None:
            step_loss = []
        self.step_loss: list = step_loss
        self.step_byte_loss: list = []

        # A model for loss per step
        self.loss_trend = loss_trend

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
        self.loss_trend = LossTrend()
        self.loss_trend.fit(self.step_byte_loss)

    def report_recent_loss(self, token_set: TokenSet):
        step = len(self.step_loss)
        loss = sum(self.step_loss[-10:]) / 10 if step >= 10 else self.step_loss[-1]
        byte_loss = token_set.byte_loss(loss)
        return f"{step}  {loss:.4f} b/token  {byte_loss:.4f} b/byte"

    def to_dict(self):
        d = {
            "step_loss": self.step_loss,
            "loss": self.loss,
            "loss_trend": self.loss_trend.to_dict(),
        }
        if self.id is not None:
            d["id"] = self.id
        return d

    @staticmethod
    def from_dict(d):
        step_loss = d.get("step_loss")
        if step_loss is not None:
            step_loss = np.array(step_loss, dtype=np.float32)
        loss_trend = build_loss_trend(d.get("loss_trend"))
        return Run(
            id=d.get("id"),
            step_loss=step_loss,
            loss=d["loss"],
            loss_trend=loss_trend,
        )
