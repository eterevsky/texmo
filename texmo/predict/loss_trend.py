import logging
import numpy as np
from scipy.optimize import minimize

from .. import latency
from ..common import INF


def _pred_log(c1, c2, eps, step):
    return c1**2 + (c2**2) * step**eps



def _pred_score(true_losses: np.ndarray, predicted_losses: np.ndarray):
    n = len(true_losses)
    assert len(predicted_losses) == n

    true_losses_log = np.log2(true_losses)
    predicted_losses_log = np.log2(predicted_losses)

    return np.average(np.abs(true_losses_log - predicted_losses_log))


class LossTrendBase(object):
    @staticmethod
    def from_dict(d: dict):
        assert d["version"] == 1
        return LossTrend(d["c1"], d["c2"], d["eps"])

    def fit(self, step_byte_loss: list[float]):
        raise NotImplementedError

    def predict(self, step: int) -> float:
        raise NotImplementedError

    def params(self) -> np.ndarray:
        raise NotImplementedError


class LossTrend(LossTrendBase):
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
        logging.info(f'Fitting loss trend for {len(losses)} losses')
        with latency.timer("LossTrend.fit"):
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
                return _pred_score(target_losses, prediction)

            init = np.array([1.0, 1.0, -1.0])
            res = minimize(_score, init)

            self._c1, self._c2, self._eps = res.x

    def predict(self, step: int) -> float:
        step = np.array(step)
        if self._eps > 0 or self._c2 == 0:
            prediction = np.array([self._c1] * step)
        else:
            prediction = _pred_log(self._c1, self._c2, self._eps, step)
        return np.where(prediction < 10, 2**prediction, INF)

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
