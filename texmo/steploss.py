"""This module builds a model to predict loss after a number of steps."""

import numpy as np
from scipy.optimize import minimize

from .predict import prediction_score

class StepLossPredictor(object):
    def __init__(self):
        self._c1 = None
        self._c2 = 0.0
        self._c3 = 0.0
        self._eps = -1.0
        self._last_loss = None

    def fit(self, losses):
        losses = np.log2(losses)

        if len(losses) == 0:
            self._c1 = 10
            self._last_loss = 8
            return

        self._last_loss = losses[-1]
        start = max(len(losses) // 3, 2)
        steps = np.array(range(start, len(losses)))

        if len(steps) < 3:
            self._c1 = self._last_loss
            self._c2 = 0
            return

        target_losses = np.array(losses[start:])
        def _score(x):
            c1, c2, eps = x
            prediction = c1 + c2 * steps ** eps
            return prediction_score(target_losses, prediction)

        init = np.array([1.0, 1.0, -1.0])
        res = minimize(_score, init)

        self._c1, self._c2, self._eps = res.x

    def predict(self, step):
        if self._eps > 0 or step < 4:
            prediction = self._last_loss
        else:
            prediction = self._c1 + self._c2 * step ** self._eps
        return 2**prediction

    def params(self) -> np.ndarray:
        return np.array([self._c1, self._c2, self._c3, self._eps])