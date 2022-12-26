"""This module builds a model to predict loss after a number of steps."""

import numpy as np
from scipy.optimize import minimize

from . import latency
from .predict_common import prediction_score


def pred_log(c1, c2, eps, step):
    return c1 + c2 * step**eps


class StepLossPredictor(object):
    version = 1

    def __init__(self, c1=None, c2=None, eps=None):
        self._c1 = c1
        self._c2 = c2
        self._eps = eps

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
                prediction = pred_log(c1, c2, eps, steps)
                return prediction_score(target_losses, prediction)

            init = np.array([1.0, 1.0, -1.0])
            res = minimize(_score, init)

            self._c1, self._c2, self._eps = res.x


    def predict(self, step):
        if self._eps > 0 or self._c2 == 0:
            prediction = self._c1
        else:
            prediction = pred_log(self._c1, self._c2, self._eps, step)
        return 2**prediction

    def params(self) -> np.ndarray:
        return np.array([self._c1, self._c2, self._eps])


def build_loss_model(losses, model_version, params):
    if params is None or model_version != 1:
        predictor = StepLossPredictor()
        predictor.fit(losses)
        return predictor
    c1, c2, eps = params
    return StepLossPredictor(c1, c2, eps)
