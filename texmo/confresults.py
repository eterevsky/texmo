from statistics import median, StatisticsError
from typing import Optional

from .configuration2 import Configuration2
from .run import Run


class ConfResults(object):

    def __init__(self, id: int, conf: Configuration2):
        self.id: int = id
        assert isinstance(conf, Configuration2)
        self.conf: Configuration2 = conf
        self.runs: list[Run] = []
        # Median of the all neighbors' scores
        self.neighbors_score: Optional[float] = None
        self.neighbors_time: Optional[float] = None
        self.pred_score: Optional[float] = None

    @property
    def median_score(self) -> Optional[float]:
        if self.runs:
            return median(r.loss for r in self.runs)
        else:
            return None
        
    def median_time(self, system: str) -> Optional[float]:
        try:
            return median(r.train_time for r in self.runs if r.system == system)
        except StatisticsError:
            return None
    
    def estimated_time(self, system: str) -> Optional[float]:
        try:
            return median(r.train_time for r in self.runs if r.system == system)
        except StatisticsError:
            return self.neighbors_time

    def add_run(self, run: Run):
        self.runs.append(run)


