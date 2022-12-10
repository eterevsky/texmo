import numpy as np
from statistics import median

from .configuration import Configuration


class Run(object):
    def __init__(self, loss: float, step_loss: list[float]=None):
        # Final loss, evaluated on the test set.
        self.loss: float = loss
        # Training loss per step, calculated on the training batch.
        self.step_loss: np.ndarray = None
        if step_loss is not None:
            self.step_loss = np.array(step_loss, dtype=np.float32)


class ConfResults(object):
    def __init__(self, id: int, conf: Configuration):
        self.id: int = id
        self.conf: Configuration = conf
        self.runs: list[Run] = []
        self.pred_score: float = None

    @property
    def median_score(self) -> float:
        if self.runs:
            return median(r.loss for r in self.runs)
        else:
            return None

    def add_run(self, run: Run):
        self.runs.append(run)


