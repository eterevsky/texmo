from bisect import bisect_left
from collections import namedtuple
import random
from statistics import median

from dataset import DataSet
from layered import LayeredModel2
from manager import Manager
from train import train_and_validate


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


def layer_neighbors(layer):
    components = layer.split(".")
    name = components[0]
    params = components[1:]
    assert len(params) >= 1
    length = int(params[0])
    if name == "suffix":
        assert len(params) == 1
        if length > 2:
            yield f"suffix.{length-1}"
        yield f"suffix.{length+1}"
    else:
        try:
            l = int(params[0])
        except Exception:
            print(layer)
            print(components)
            print(params)
            raise
        if len(params) > 1:
            activation = params[1]
        else:
            activation = "tanh"

        if name != "gru":
            yield f"gru.{length}.tanh"
            if name in ("dense", "rec") and length >= 2:
                yield f"gru.{length // 2}.tanh"
        if name != "lstm":
            yield f"lstm.{length}"
            if name in ("dense", "rec") and length >= 2:
                yield f"lstm.{length // 2}"
        if name != "rec":
            yield f"rec.{length}.{activation}"
            if name in ("gru", "lstm"):
                yield f"rec.{length*2}.{activation}"
        if name != "dense":
            yield f"dense.{length}.{activation}"
            if name in ("gru", "lstm"):
                yield f"dense.{length*2}.{activation}"

        if length > 1:
            new_len = length // 2
            if name == "lstm":
                yield f"lstm.{new_len}"
            else:
                yield f"{name}.{new_len}.{activation}"

        new_len = length * 2
        if name == "lstm":
            yield f"lstm.{new_len}"
        else:
            yield f"{name}.{new_len}.{activation}"

        if name != "lstm":
            if activation != "tanh":
                yield f"{name}.{length}.tanh"
            if activation != "sigmoid":
                yield f"{name}.{length}.sigmoid"
            if activation != "relu":
                yield f"{name}.{length}.relu"


def spec_neighbors(spec):
    yield spec + "-dense.128.tanh"
    if not spec.startswith("suffix"):
        yield "suffix.2-" + spec

    layers = spec.split("-")
    if len(layers) > 1:
        yield "-".join(layers[:-1])

    if layers[0] == "suffix.2":
        yield "-".join(layers[1:])

    for i, layer in enumerate(layers):
        for modified in layer_neighbors(layer):
            yield "-".join(layers[:i] + [modified] + layers[i + 1 :])


def conf_neighbors(conf):
    ilr = LRS.index(conf.lr)
    if ilr > 0:
        yield conf._replace(lr=LRS[ilr - 1])
    if ilr < len(LRS) - 1:
        yield conf._replace(lr=LRS[ilr + 1])

    for spec in spec_neighbors(conf.spec):
        yield conf._replace(spec=spec)

    if conf.batch > 1:
        yield conf._replace(batch=conf.batch // 2)
        yield conf._replace(batch=conf.batch // 2, sample_len=conf.sample_len * 2)
    yield conf._replace(batch=conf.batch * 2)

    if conf.sample_len > 1:
        yield conf._replace(sample_len=conf.sample_len // 2)
        yield conf._replace(sample_len=conf.sample_len // 2, batch=conf.batch * 2)
    yield conf._replace(sample_len=conf.sample_len * 2)


class ConfResults(object):
    def __init__(self, conf):
        self.conf = conf
        self.reports = []
        self.neighbors = list(conf_neighbors(conf))

    def add_report(self, report):
        self.reports.append(report)

    @property
    def report_count(self):
        return len(self.reports)

    @property
    def score(self):
        return median(
            report.loss if report.loss < 100 else 100 for report in self.reports
        )

    @property
    def scores(self):
        return list(
            report.loss if report.loss < 100 else 100 for report in self.reports
        )


def select_conf(results) -> Configuration:
    prev = None
    prev_score = 0
    best = None
    best_score = 10000
    for conf, conf_results in results.items():
        assert conf_results.report_count > 0
        score = conf_results.score
        if score >= best_score:
            continue
        random.shuffle(conf_results.neighbors)
        for neighbor_conf in conf_results.neighbors:
            if neighbor_conf not in results:
                prev = conf
                prev_score = score
                best = neighbor_conf
                best_score = score
                best_count = 0
                break
            neighbor_results = results[neighbor_conf]
            neighbor_score = score + 0.3 * neighbor_results.report_count

            if neighbor_score < best_score:
                prev = conf
                prev_score = score
                best = neighbor_conf
                best_score = neighbor_score
                best_count = neighbor_results.report_count

    return best


def print_top(results):
    top = []
    for conf_results in results.values():
        top.append((conf_results.score, conf_results.conf, conf_results.scores))

    top.sort()

    print()
    for score, conf, scores in top[:20]:
        if len(scores) > 1:
            scores = f" {scores}"
        else:
            scores = ""
        print(f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  B{conf.batch:4}  {score:.4f}{scores}")
    print()


def search(
    data,
    learning_rate,
    sample_len,
    batch,
    regularization,
    time_limit,
    log,
):
    print(f"Training data: {data}")
    train_set = DataSet(data)

    start_confs = [
        Configuration("suffix.2-rec.64.relu-dense.512.relu-dense.128.tanh", learning_rate, sample_len, batch)
    ]
    results = {}

    while True:
        if start_confs:
            conf = start_confs.pop()
        else:
            conf = select_conf(results)

        print(f"{conf.spec:32} LR{conf.lr} LEN{conf.sample_len} B{conf.batch}")
        model = LayeredModel2.parse(conf.spec)
        manager = Manager(model, conf.lr, regularization, 100000)
        manager.init()
        report = train_and_validate(
            manager,
            steps=None,
            time_limit=time_limit,
            train_set=train_set,
            sample_len=conf.sample_len,
            batch_size=conf.batch,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=log,
        )

        if conf not in results:
            results[conf] = ConfResults(conf)
        results[conf].add_report(report)

        print_top(results)


if __name__ == '__main__':
    conf = ConfResults(Configuration("suffix.3-dense.128.relu-dense.256.relu-dense.256.tanh", 0.05, 8, 256))
    for n in conf.neighbors:
        print(n)