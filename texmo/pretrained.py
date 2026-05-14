import logging
import os
import pickle
from copy import copy

from .configuration import Configuration, conf_to_dict
from .db import DbReader
from .model2 import Weights
from .run import Run


class Checkpoint(object):
    def __init__(self, conf: Configuration, run: Run):
        self.history: list[tuple[Configuration, Run]] = [(conf, run)]

    def __str__(self):
        runs = len(self.history)
        return f"loss {self.loss:.4f} R{runs} T{self.total_time}"

    @property
    def total_time(self):
        return sum(c.t for (c, _) in self.history)

    @property
    def loss(self):
        return self.history[-1][1].loss

    def add_run(self, conf, run):
        self.history.append((conf, run))

    def load_weights(self, checkpoints_path: str):
        path = os.path.join(checkpoints_path, self.history[-1][1].checkpoint)
        with open(path, "b") as f:
            save = pickle.load(f)
        return save["weights"]

    def save(self, weights: Weights, checkpoints_path: str):
        conf, run = self.history[-1]

        base_name = f"{run.loss:.4f}-{conf.model}"
        name = base_name + ".pickle"
        i = 0
        save = {
            "training": [
                {
                    "conf": conf_to_dict(conf),
                    "run": run.to_dict(),
                }
            ],
            "weights": weights
        }
        while os.path.exists(os.path.join(checkpoints_path, name)):
            i += 1
            name = f"{base_name}({i}).pickle"
        with open(os.path.join(checkpoints_path, name), "wb") as f:
            logging.info(f"Saving the model to {name}")
            pickle.dump(save, f)
        run.checkpoint = name

    def clone(self):
        return copy(self)


class Checkpoints(object):
    def __init__(self, db: DbReader):
        self._db = db
