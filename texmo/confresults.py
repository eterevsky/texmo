from collections import defaultdict
from statistics import median, StatisticsError
from typing import Optional

from .configuration2 import Configuration2
from .run import Run


class ConfTiming(object):
    def __init__(self):
        self._similar_times: list[float] = []
        self._neighbors_times: list[float] = []

    def add_similar_time(self, time: float):
        self._similar_times.append(time)
    
    def add_neighbors_time(self, time: float):
        self._neighbors_times.append(time)

    def median(self) -> Optional[float]:
        if self._similar_times:
            return median(self._similar_times)
        else:
            return None

    def estimate(self):
        if self._similar_times:
            return median(self._similar_times)
        elif self._neighbors_times:
            return median(self._neighbors_times)
        else:
            return None


class ConfResults(object):

    def __init__(self, id: int, conf: Configuration2):
        self.id: int = id
        assert isinstance(conf, Configuration2)
        self.conf: Configuration2 = conf
        self.runs: list[Run] = []
        # Median of the all neighbors' scores
        self.neighbors_score: Optional[float] = None
        self.pred_score: Optional[float] = None

        self._times: defaultdict[str, ConfTiming] = defaultdict(ConfTiming)

    @property
    def median_score(self) -> Optional[float]:
        if self.runs:
            return median(r.loss for r in self.runs)
        else:
            return None

    def median_time(self, system: str) -> Optional[float]:
        return self._times[system].median()

    def estimated_time(self, system: str) -> Optional[float]:
        return self._times[system].estimate()

    def add_run(self, run: Run):
        self.runs.append(run)
        self._times[run.system].add_similar_time(run.train_time)

    def add_neighbor_run(self, neighbor: Configuration2, run: Run):
        assert neighbor != self.conf

        scaled_time = run.train_time * self.conf.steps / neighbor.steps

        if self.conf.model == neighbor.model and self.conf.length == neighbor.length and self.conf.batch == neighbor.batch:
            self._times[run.system].add_similar_time(scaled_time)
        else:
            self._times[run.system].add_neighbors_time(scaled_time)


