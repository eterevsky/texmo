"""This module builds a model to predict loss after a number of steps."""

import numpy as np
from scipy.optimize import minimize

class StepLossPredictor(object):
    def __init__(self):
        self._c1 = None
        self._c2 = None
        self._eps = None

    def fit(self, losses):
        start = len(losses) // 8
        log_losses = np.log2(np.array(losses[start:]))
        steps = np.array(range(start, len(losses)))
        
        def _model_loss(x):
            c1, c2, eps = x
            prediction = np.log2(c1 + c2 * steps ** eps)
            return np.average(np.abs(prediction - log_losses))

        init = np.array([1.0, 1.0, -1.0])
        res = minimize(_model_loss, init)
        print(res)

        self._c1, self._c2, self._eps = res.x

    def predict(self, step):
        return self._c1 + self._c2 * step ** self._eps

    def params(self) -> np.ndarray:
        return np.array([self._c1, self._c2, self._eps])