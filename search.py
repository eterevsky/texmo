import argparse
from bisect import bisect
from collections import namedtuple
import csv
import itertools
import math
import random
from statistics import median, StatisticsError

from dataset import DataSet
from layered import LayeredModel2
from manager import Manager
from model import NCHAR
from record import TrainingRecord
from spec import ModelSpec
from train import train_and_eval


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


Configuration = namedtuple(
    "Configuration",
    ["spec", "lr", "sample_len", "batch", "regularization", "init_scale"],
)


def conf_from_record(record):
    spec = ModelSpec.parse(record.model_spec)
    return Configuration(
        spec,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.regularization,
        1.0,
    )


def neighbor_numbers(x):
    """Produce neighbor numbers from an close to exponent but readable table"""
    i = LRS.index(x)
    if i > 0:
        yield LRS[i - 1]
    if i < len(LRS) - 1:
        yield LRS[i + 1]


def conf_neighbors(conf, max_weights=None, vary=()):
    yield conf

    for spec in conf.spec.neighbors(vary):
        if max_weights is None or spec.weights() <= max_weights:
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


def is_power2(x):
    return type(x) is int and x >= 1 and x & (x - 1) == 0


def conf_is_valid(conf):
    return (
        conf.spec.is_valid()
        and conf.lr in LRS
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and conf.regularization in LRS
        and conf.init_scale in LRS
    )


INF = float("inf")


class ConfResults(object):
    def __init__(self, conf, max_weights, vary):
        self.conf = conf
        self.neighbor_confs = list(conf_neighbors(conf, max_weights, vary))
        self.scores = []
        self.cluster_score = None

    def add_report(self, report):
        score = report.loss
        if math.isnan(score):
            score = INF
        self.scores.append(score)

    @property
    def scores_count(self):
        return len(self.scores)

    @property
    def score(self):
        return median(self.scores)


class ResultsSet(object):
    def __init__(self, max_weights, vary):
        # Configuration -> ConfResults
        self._results = {}
        self._sorted_results = []
        self.total_runs = 0
        self._max_weights = max_weights
        self._vary = vary
        self.startup_conf = None

    def add_conf(self, conf):
        if conf in self._results:
            return
        conf_results = ConfResults(conf, self._max_weights, self._vary)
        self._results[conf] = conf_results
        self._sorted_results.append(conf_results)

    def cluster_score(self, conf_results):
        if conf_results.cluster_score is not None:
            return conf_results.cluster_score

        self_score = conf_results.score if conf_results.scores else INF

        def iter_neighbors():
            for neighbor_conf in conf_results.neighbor_confs:
                neighbor = self._results.get(neighbor_conf)
                if neighbor is not None and neighbor.scores:
                    yield neighbor.score

        try:
            neighbor_score = median(
                itertools.chain(conf_results.scores, iter_neighbors())
            )
        except StatisticsError:
            neighbor_score = INF

        cluster_score = min(self_score, neighbor_score)
        conf_results.cluster_score = cluster_score
        return cluster_score

    def select_conf(self):
        """
        i + 1 <= N // 2^k -> >= k
        (i + 1) * 2^k <= N
        """
        if self.startup_conf is not None:
            conf = self.startup_conf
            self.startup_conf = None

            if conf not in self._results:
                self.add_conf(conf)
            return self._results[conf]

        self._sorted_results.sort(key=self.cluster_score)

        k = 16
        k2 = 3**k
        stop_early = random.choice([False, True])
        best_conf_results = None
        best_count_gap = 0

        for i, conf_results in enumerate(self._sorted_results):
            while k > 1 and (i + 1) * k2 > self.total_runs:
                k -= 1
                k2 //= 3
            if k - conf_results.scores_count > best_count_gap:
                best_count_gap = k - conf_results.scores_count
                best_conf_results = conf_results
                if stop_early:
                    break
            if best_count_gap >= k:
                break

        return best_conf_results

    def add_run(self, conf, report):
        self.total_runs += 1

        self.add_conf(conf)
        conf_results = self._results[conf]

        for neighbor_conf in conf_results.neighbor_confs:
            self.add_conf(neighbor_conf)
            self._results[neighbor_conf].cluster_score = None

        conf_results.add_report(report)
        conf_results.cluster_score = None

    def top_score(self, n):
        self._sorted_results.sort(key=self.cluster_score)
        return self._sorted_results[:n]


def print_top(results):
    print()
    for conf_results in results.top_score(20):
        conf = conf_results.conf
        score = f"{conf_results.score:.4f}" if conf_results.scores else "      "
        print(
            f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
            + f"B{conf.batch:4}  R{conf.regularization:4} "
            + f"I{conf.init_scale:4}  {score} ({conf_results.scores_count})"
        )
    print()


def load_previous_runs(
    results, path, max_weights, time_limit, sample_len, init_scale, vary
):
    print(f"Loading previous runs from {path}")
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            record = TrainingRecord.from_csv_tuple(row)
            try:
                conf = conf_from_record(record)
            except Exception:
                continue

            if (
                conf_is_valid(conf)
                and (max_weights is None or record.weights <= max_weights)
                and (
                    "len" in vary
                    or sample_len is None
                    or record.train_sample_len == sample_len
                )
                and ("init_scale" in vary or record.init_scale == init_scale)
            ) and time_limit / 1.1 < record.train_time_s < time_limit * 1.1:
                results.add_run(conf, record)
    print("Runs loaded:", results.total_runs)


def main(
    data,
    model_spec,
    learning_rate,
    sample_len,
    batch_size,
    regularization,
    time_limit,
    log,
    load,
    max_weights,
    init_scale,
    vary="",
    max_attempts=1000000,
):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()

    vary = vary.split(",")

    results = ResultsSet(max_weights, vary)

    if load is not None:
        load_previous_runs(
            results, load, max_weights, time_limit, sample_len, init_scale, vary
        )

    if model_spec is not None:
        model_spec = ModelSpec.parse(model_spec)
        assert max_weights is None or model_spec.weights() <= max_weights
        results.add_conf(
            Configuration(
                model_spec,
                learning_rate,
                sample_len,
                batch_size,
                regularization,
                init_scale=init_scale,
            )
        )

    first = True

    for _ in range(max_attempts):
        conf_results = results.select_conf()
        conf = conf_results.conf
        weights = conf.spec.weights()
        print(
            f"{conf.spec} ({weights})  LR {conf.lr}  LEN {conf.sample_len}  "
            + f"B {conf.batch}  R {conf.regularization}  I {conf.init_scale}"
        )
        model = LayeredModel2.parse(str(conf.spec))
        manager = Manager(
            model,
            conf.lr,
            regularization=conf.regularization,
            init_scale=conf.init_scale,
        )
        manager.init(quiet=True)
        assert model.total_weights(manager.weights) == weights
        report = train_and_eval(
            manager,
            steps=None,
            time_limit=time_limit,
            train_set=dataset,
            sample_len=conf.sample_len,
            batch_size=conf.batch,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=log,
            quiet=True,
        )

        if first:
            first = False
        else:
            results.add_run(conf, report)
            print_top(results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="directory with training data",
    )
    parser.add_argument(
        "-c",
        "--model-spec",
        metavar="SPEC",
        default=None,
        help="initial model spec",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="the maximum number of weights in the model",
    )
    parser.add_argument(
        "-v",
        "--vary",
        type=str,
        default="struct,size,suffix,lr,batch,activation",
        help="model parameters that can be varied with search. "
        + "A comma-separated list of struct, size, act, lr, batch, len.",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=8,
        metavar="SECONDS",
        help="time limit for training",
    )
    parser.add_argumen
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        metavar="RATE",
        help="learning rate",
        default=0.05,
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        help="L2 regularization coefficient",
        default=0.1,
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="LEN",
        help="length of text fragments used for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="BATCH",
        default=32,
        help="training data batch size",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="LOG",
        help="path to a CSV file for logging",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="LOG",
        help="a CSV log file with previous runs",
    )
    parser.add_argument(
        "--init_scale",
        default=1,
        type=float,
        help="scale weight initialization",
    )

    return parser.parse_args()


# if __name__ == "__main__":
#     print("TexMo parameter search")
#     args = parse_args()
#     main(**vars(args))


# if __name__ == "__main__":
#     for i in range(11, 64):
#         max_weights = 2**i
#         main(
#             "data",
#             None,
#             0.1,
#             128,
#             256,
#             0.1,
#             128,
#             "search4-gpu.csv",
#             "search4-gpu.csv",
#             max_weights,
#             1,
#             vary="lr,batch,size,activation,struct,suffix",
#             max_attempts=30,
#         )
