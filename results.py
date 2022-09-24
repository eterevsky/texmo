from collections import namedtuple
import csv
import itertools
import math
from statistics import median, StatisticsError

import latency
from record import TrainingRecord
from spec import ModelSpec, ObsoleteSpec


INF = float("inf")
LRS = [
    0.00001,
    0.00002,
    0.00005,
    0.0001,
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
]


def neighbor_numbers(x):
    """Produce neighbor numbers from a close to exponent but readable table"""
    i = LRS.index(x)
    if i > 0:
        yield LRS[i - 1]
    if i < len(LRS) - 1:
        yield LRS[i + 1]


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


def conf_neighbors(conf, vary=()):
    for spec in conf.spec.neighbors(vary):
        yield conf._replace(spec=spec)

    if "batch" in vary:
        yield conf._replace(batch=conf.batch * 2)
        if conf.batch > 1:
            yield conf._replace(batch=conf.batch // 2)

    if "lr" in vary:
        for x in neighbor_numbers(conf.lr):
            yield conf._replace(lr=x)

    if "len" in vary:
        if conf.sample_len >= 4:
            yield conf._replace(sample_len=conf.sample_len // 2)
        yield conf._replace(sample_len=conf.sample_len * 2)

    if "regularization" in vary:
        for x in neighbor_numbers(conf.regularization):
            yield conf._replace(regularization=x)

    if "init_scale" in vary:
        for x in neighbor_numbers(conf.init_scale):
            yield conf._replace(init_scale=x)


class ConfResults(object):
    def __init__(self, conf):
        self.conf = conf
        self.scores = []
        self.neighbors = None
        self.cluster_score = None

    def add_record(self, record):
        score = record.loss
        if math.isnan(score):
            score = INF
        self.scores.append(score)

    @property
    def score(self):
        return median(self.scores) if self.scores else INF

    @property
    def weights_limit(self):
        """The number of weights rounded up to a power of 2."""
        weights = self.conf.spec.weights()
        log = math.log2(weights)
        return 2 ** math.ceil(log)


class ResultSet(object):
    def __init__(self, t=None, vary=()):
        self._t = t
        # t -> conf -> conf results
        self._conf_results = {}
        self._vary = vary

    @staticmethod
    def from_csv(filename, t=None, vary=()):
        result_set = ResultSet(t=t, vary=vary)
        with open(filename) as csvfile:
            for row in csv.reader(csvfile):
                record = TrainingRecord.from_csv_tuple(row)
                result_set.add_record(record)
        return result_set

    def add_record(self, record):
        if self._t is not None and record.time_round != self._t:
            return

        try:
            conf = conf_from_record(record)
        except ObsoleteSpec:
            return
        if not conf.spec.is_valid():
            return
        if conf.t is None or conf.t < 1:
            return

        if conf.t not in self._conf_results:
            self._conf_results[conf.t] = {}

        results_for_t = self._conf_results[conf.t]

        conf_results = results_for_t.get(conf, None)
        if conf_results is None:
            conf_results = ConfResults(conf)
            conf_results.neighbors = list(conf_neighbors(conf, self._vary))
            results_for_t[conf] = conf_results

        for neighbor in conf_results.neighbors:
            neighbor_results = results_for_t.get(neighbor)
            if neighbor_results is None:
                neighbor_results = ConfResults(neighbor)
                neighbor_results.neighbors = list(
                    conf_neighbors(neighbor, self._vary)
                )
                results_for_t[neighbor] = neighbor_results
            else:
                neighbor_results.cluster_score = None

        assert conf_results.conf == conf

        conf_results.add_record(record)

    def all_results(self):
        for results_for_t in self._conf_results.values():
            for cr in results_for_t.values():
                yield cr

    def results_for_t(self, t):
        results = self._conf_results.get(t, None)
        return () if results is None else results.values()

    def runs_count(self, t, max_weights=INF):
        return sum(
            len(cr.scores)
            for cr in self.results_for_t(t)
            if cr.conf.spec.weights() <= max_weights
        )

    def confs_count(self, t, max_weights=INF):
        return sum(1 for cr in self.results_for_t(t) if cr.scores and cr.conf.spec.weights() <= max_weights)

    def top_conf(self, t):
        """A configuration with the highest (self) score with time = t."""
        results_for_t = self._conf_results.get(t, None)
        if results_for_t is None or not results_for_t:
            return None
        _, conf = min(
            (cr.score, cr.conf) for cr in results_for_t.values() if cr.scores
        )
        return results_for_t[conf]

    def top_confs(self, t, max_weights):
        with latency.timer("top_confs"):
            matching_crs = (
                cr
                for cr in self.results_for_t(t)
                if cr.conf.spec.weights() <= max_weights
            )
            return sorted(matching_crs, key=lambda cr: cr.score)

    def find(self, conf):
        return self._conf_results.get(conf.t, {}).get(conf, None)

    def cluster_score(self, t, cr):
        conf_results = self._conf_results.get(t, {})

        if cr.cluster_score is not None:
            return cr.cluster_score

        self_score = cr.score if cr.scores else INF

        def iter_neighbors():
            for neighbor_conf in cr.neighbors:
                neighbor_results = conf_results.get(neighbor_conf)
                if neighbor_results is not None and neighbor_results.scores:
                    yield neighbor_results.score

        try:
            neighbor_score = median(
                itertools.chain(cr.scores, iter_neighbors())
            )
        except StatisticsError:
            neighbor_score = INF

        cluster_score = min(self_score, neighbor_score)
        cr.cluster_score = cluster_score
        return cluster_score

    def top_cluster_confs(self, t, max_weights):
        with latency.timer("top_cluster_confs"):
            matching_crs = (
                cr
                for cr in self.results_for_t(t)
                if cr.conf.spec.weights() <= max_weights
            )
            return sorted(matching_crs, key=lambda cr: self.cluster_score(t, cr))

    @property
    def total_confs(self):
        return sum(len(r) for r in self._conf_results.values())

    @property
    def total_runs(self):
        return sum(len(cr.scores) for cr in self.all_results())
