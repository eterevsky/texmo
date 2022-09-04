import argparse
from collections import namedtuple
import random
from statistics import median

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
            report.loss if report.loss < 1000 else 1000
            for report in self.reports
        )


def select_conf(results) -> Configuration:
    best = None
    best_score = 10000
    for conf, conf_results in results.items():
        assert conf_results.report_count > 0
        score = conf_results.score
        if score >= best_score:
            continue
        for neighbor_conf in conf_results.neighbors:
            if neighbor_conf not in results:
                best = neighbor_conf
                best_score = score
                break
            neighbor_results = results[neighbor_conf]
            neighbor_score = (
                score + min(0.1, score * 0.02) * neighbor_results.report_count
            )

            if neighbor_score < best_score:
                best = neighbor_conf
                best_score = neighbor_score

    return best


def print_top(results):
    top = []
    for conf_results in results.values():
        top.append((conf_results.score, conf_results.conf, conf_results.scores))

    top.sort()

    print()
    for score, conf, scores in top[:20]:
        if len(scores) > 1:
            scores = " [" + ", ".join(f"{s:.4f}" for s in sorted(scores)) + "]"
        else:
            scores = ""
        print(
            f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
            + f"B{conf.batch:4}  {score:.4f}{scores}"
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

    start_conf = Configuration(
        model_spec, learning_rate, sample_len, batch_size
    )
    results = {}

    while True:
        if results:
            conf = select_conf(results)
        else:
            conf = start_conf

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

        if conf not in results:
            results[conf] = ConfResults(conf, max_weights, vary)
        results[conf].add_report(report)

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
