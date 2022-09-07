import argparse
from collections import namedtuple
import itertools
from statistics import median, StatisticsError

from dataset import DataSet
from layered import LayeredModel2
from manager import Manager
from model import NCHAR
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
]


Configuration = namedtuple(
    "Configuration", ["spec", "lr", "sample_len", "batch"]
)


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
        ilr = LRS.index(conf.lr)
        if ilr > 0:
            yield conf._replace(lr=LRS[ilr - 1])
        if ilr < len(LRS) - 1:
            yield conf._replace(lr=LRS[ilr + 1])


class ConfResults(object):
    def __init__(self, conf, max_weights, vary):
        self.conf = conf
        self.reports = []
        self.neighbors = list(conf_neighbors(conf, max_weights, vary))
        self.cluster_score = None

    def add_report(self, report):
        self.reports.append(report)

    @property
    def report_count(self):
        return len(self.reports)

    @property
    def score(self):
        return median(
            report.loss if report.loss < 1000 else 1000
            for report in self.reports
        )

    @property
    def scores(self):
        return list(
            sorted(
                report.loss if report.loss < 1000 else 1000
                for report in self.reports
            )
        )


class ResultsSet(object):
    def __init__(self, max_weights, vary):
        # Configuraion -> ConfResults
        self._conf_to_results = {}
        self.total_runs = 0
        self._max_weights = max_weights
        self._vary = vary

    def cluster_score(self, conf):
        """Score based on train runs +

        A median of all trials of a given configuration + all known (median)
        scores of its neighbors.
        """
        conf_results = self._conf_to_results[conf]

        def iter_neighbors():
            for neighbor in conf_results.neighbors:
                neighbor_results = self._conf_to_results.get(neighbor)
                if (
                    neighbor_results is not None
                    and neighbor_results.report_count > 0
                ):
                    yield neighbor_results.score

        try:
            score = median(itertools.chain(conf_results.scores, iter_neighbors()))
        except StatisticsError:
            score = 1000

        return score

    def add_conf(self, conf):
        conf_results = ConfResults(conf, self._max_weights, self._vary)
        self._conf_to_results[conf] = conf_results

    def add_run(self, conf, report):
        self.total_runs += 1

        conf_results = self._conf_to_results[conf]
        conf_results.add_report(report)

        conf_results.cluster_score = self.cluster_score(conf)

        for neighbor in conf_results.neighbors:
            if neighbor in self._conf_to_results:
                neighbor_results = self._conf_to_results[neighbor]
            else:
                neighbor_results = ConfResults(
                    neighbor, self._max_weights, self._vary
                )
                self._conf_to_results[neighbor] = neighbor_results

            neighbor_results.cluster_score = self.cluster_score(neighbor)

    def select_conf(self):
        """
        i + 1 <= N // 2^k -> >= k
        (i + 1) * 2^k <= N
        """
        k = 16
        k2 = 3**k
        for i, conf_results in enumerate(
            sorted(
                self._conf_to_results.values(),
                key=lambda r: r.cluster_score
            )
        ):
            while k > 1 and (i + 1) * k2 > self.total_runs:
                k -= 1
                k2 //= 3
            if conf_results.report_count < k:
                return conf_results

    def top_self_score(self, n):
        return sorted(self._conf_to_results.values(),
                      key=lambda r: r.score if r.report_count > 0 else r.cluster_score)[:n]


def print_top(results):
    print()
    for conf_results in results.top_self_score(20):
        conf = conf_results.conf
        score = f"{conf_results.score:.4f}" if conf_results.report_count > 0 else "      "
        print(
            f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
            + f"B{conf.batch:4}  {score} ({conf_results.report_count}) {conf_results.cluster_score:.4f}"
        )
    print()


def main(
    data,
    model_spec,
    learning_rate,
    sample_len,
    batch_size,
    regularization,
    time_limit,
    log,
    max_weights,
    vary="",
):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()

    model_spec = ModelSpec.parse(model_spec)
    vary = vary.split(",")

    assert max_weights is None or model_spec.weights() <= max_weights

    results = ResultsSet(max_weights, vary)
    results.add_conf(Configuration(model_spec, learning_rate, sample_len, batch_size))

    while True:
        conf_results = results.select_conf()
        conf = conf_results.conf
        weights = conf.spec.weights()
        print(
            f"{conf.spec} ({weights})  LR {conf.lr}  LEN {conf.sample_len}  "
            + f"B {conf.batch}"
        )
        model = LayeredModel2.parse(str(conf.spec))
        manager = Manager(model, conf.lr, regularization)
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
        default="dense.128.tanh",
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

    return parser.parse_args()


if __name__ == "__main__":
    print("TexMo parameter search")
    args = parse_args()
    main(**vars(args))
