from collections import namedtuple
import csv
import math
from statistics import median

from record import TrainingRecord
from spec import ModelSpec, ObsoleteSpec


INF = float("inf")


Configuration = namedtuple(
    "Configuration",
    ["spec", "lr", "sample_len", "batch", "regularization", "init_scale", "t"],
)


def conf_from_record(record):
    spec = ModelSpec.parse(record.model_spec)
    return Configuration(
        spec,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.regularization,
        record.init_scale,
        record.time_round,
    )


class ConfResults(object):
    def __init__(self, conf):
        self.conf = conf
        self.scores = []

    def add_record(self, record):
        score = record.loss
        if math.isnan(score):
            score = INF
        self.scores.append(score)

    @property
    def score(self):
        return median(self.scores)


class ResultSet(object):
    def __init__(self, t=None):
        self._t = t
        self._conf_results = {}

    @staticmethod
    def from_csv(filename, t=None):
        result_set = ResultSet(t=t)
        with open(filename) as csvfile:
            for row in csv.reader(csvfile):
                record = TrainingRecord.from_csv_tuple(row)
                result_set.add_record(record)
        return result_set

    def add_record(self, record):
        if self._t is not None and record.time_round != self._t: return

        try:
            conf = conf_from_record(record)
        except ObsoleteSpec:
            return
        conf_results = self._conf_results.get(conf, None)
        if conf_results is None:
            conf_results = ConfResults(conf)
            self._conf_results[conf] = conf_results

        assert conf_results.conf == conf

        conf_results.add_record(record)

    def all_results(self):
        return self._conf_results.values()

