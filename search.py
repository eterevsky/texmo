import argparse
from collections import namedtuple
import random
from statistics import median

from dataset import DataSet
from layered import LayeredModel2
from manager import Manager
from model import NCHAR
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


def layer_weights(name, size, in_size, out_size):
    if name == "suffix":
        return in_size * size * out_size
    elif name == "dense":
        return size + size * in_size + size * out_size
    elif name == "rec":
        return size + size * in_size + size * size + size * out_size
    elif name == "gru":
        return 3 * size + 3 * size * in_size + 3 * size * size + size * out_size
    elif name == "lstm":
        return 4 * size + 4 * size * in_size + 4 * size * size + size * out_size
    elif name == "conv":
        return (
            size * in_size * out_size
        )  # This is not right, but this will have to do


def log_distance(x, y):
    if x < y:
        return y / x
    else:
        return x / y


def change_layer_type(name, size, prev_layer, next_layer):
    if name == "norm" or name == "suffix":
        return
    if prev_layer is None:
        in_size = NCHAR
    else:
        components = prev_layer.split(".")
        if components[0] == "suffix":
            in_size = int(components[1]) * NCHAR
        else:
            in_size = int(components[1])

    if next_layer is None:
        out_size = NCHAR
    else:
        components = next_layer.split(".")
        next_name = components[0]
        size = int(components[1])
        if next_name in ("dense", "rec"):
            out_size = size
        elif next_name == "gru":
            out_size = 3 * size
        elif next_name == "lstm":
            out_size = 4 * size
        elif next_name == "suffix":
            out_size = size * NCHAR  # This is not right!

    current_weights = layer_weights(name, size, in_size, out_size)

    for new_type in ("dense", "rec", "gru", "lstm"):
        if new_type == name:
            continue
        new_size = size
        new_weights = layer_weights(new_type, new_size, in_size, out_size)
        move_down = new_weights > current_weights
        dist = log_distance(current_weights, new_weights)
        prev_dist = 100
        while dist < prev_dist:
            best_size = new_size
            if move_down:
                if new_size == 1:
                    break
                new_size //= 2
            else:
                new_size *= 2
            new_weights = layer_weights(new_type, new_size, in_size, out_size)
            prev_dist = dist
            dist = log_distance(current_weights, new_weights)
        if new_type == "lstm":
            yield f"lstm.{best_size}"
        else:
            yield f"{new_type}.{best_size}.tanh"


def layer_neighbors(layer, prev_layer, next_layer, vary):
    components = layer.split(".")
    name = components[0]
    if name == "norm":
        return
    params = components[1:]
    assert len(params) >= 1
    size = int(params[0])

    if name == "suffix":
        assert len(params) == 1
        if "size" in vary:
            if size > 2:
                yield f"suffix.{size-1}"
            yield f"suffix.{size+1}"
        return

    activation = "." + params[1] if len(params) > 1 else ""

    if size in vary:
        if size > 1:
            yield f"{name}.{size // 2}{activation}"
        yield f"{name}.{size * 2}{activation}"

    if activation and "act" in vary:
        assert name != "lstm"
        if activation != ".tanh":
            yield f"{name}.{size}.tanh"
        if activation != ".relu":
            yield f"{name}.{size}.relu"

    # if name != "dense":
    #     yield f"dense.{size}.tanh"
    # if name != "rec":
    #     yield f"rec.{size}.tanh"
    # if name != "gru":
    #     yield f"gru.{size}.tanh"
    # if name != "lstm":
    #     yield f"lstm.{size}"

    if "struct" in vary:
        for neighbor in change_layer_type(name, size, prev_layer, next_layer):
            yield neighbor


def spec_neighbors(spec, vary):
    layers = spec.split("-")

    for i, layer in enumerate(layers):
        prev_layer = layers[i - 1] if i > 0 else None
        if prev_layer == "norm":
            prev_layer = layers[i - 2] if i > 1 else None
        next_layer = layers[i + 1] if i < len(layers) - 1 else None
        if next_layer == "norm":
            next_layer = layers[i + 2] if i < len(layers) - 2 else None
        assert prev_layer != "norm"
        assert next_layer != "norm"
        for modified in layer_neighbors(layer, prev_layer, next_layer, vary):
            yield "-".join(layers[:i] + [modified] + layers[i + 1 :])

    if "size" in vary:
        if not spec.startswith("suffix"):
            yield "suffix.2-" + spec
        if layers[0] == "suffix.2" and len(layers) > 1 and layers[1] != "norm":
            yield "-".join(layers[1:])

    if "norm" in vary:
        for i in range(1, len(layers)):
            if layers[i].startswith("norm"):
                yield "-".join(layers[:i] + layers[i + 1 :])
            elif (
                not layers[i - 1].startswith("norm")
                and not layers[i - 1].startswith("suffix")
                and not layers[i].startswith("norm")
                and not layers[i].startswith("suffix")
            ):
                yield "-".join(layers[:i] + ["norm"] + layers[i:])

        if not layers[-1].startswith("norm") and not layers[-1].startswith(
            "suffix"
        ):
            yield "-".join(layers + ["norm"])

    if "struct" in vary:
        if len(layers) > 1:
            yield "-".join(layers[:-1])

    last_layer = layers[-1] if layers[-1] != "norm" else layers[-2]
    components = last_layer.split(".")
    size = int(components[1])
    if components[0] == "suffix":
        size *= NCHAR

    if "struct" in vary:
        out_size = min(128, size)
        yield spec + f"-dense.{out_size}.tanh"


TOTAL_WEIGHTS_MEMO = {}


def total_weights(spec):
    w = TOTAL_WEIGHTS_MEMO.get(spec, 0)
    if w > 0:
        return w

    layer_specs = spec.split("-")
    weights = 0
    cur_size = NCHAR
    for layer_spec in layer_specs:
        components = layer_spec.split(".")
        name = components[0]
        if name == "norm":
            continue
        size = int(components[1])
        weights += layer_weights(name, size, cur_size, 0)
        cur_size = cur_size * size if name == "suffix" else size
    weights += cur_size * NCHAR + NCHAR
    TOTAL_WEIGHTS_MEMO[spec] = weights
    return weights


def conf_neighbors(conf, max_weights=None, vary=()):
    yield conf

    for spec in spec_neighbors(conf.spec, vary):
        if max_weights is None or total_weights(spec) <= max_weights:
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
            report.loss if report.loss < 100 else 100 for report in self.reports
        )

    @property
    def scores(self):
        return list(
            report.loss if report.loss < 100 else 100 for report in self.reports
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
            scores = " [" + ", ".join(f"{s:.4f}" for s in scores) + "]"
        else:
            scores = ""
        print(
            f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  B{conf.batch:4}  {score:.4f}{scores}"
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

    vary = vary.split(",")

    assert max_weights is None or total_weights(model_spec) <= max_weights

    start_confs = [
        Configuration(model_spec, learning_rate, sample_len, batch_size)
    ]
    results = {}

    while True:
        if start_confs:
            conf = start_confs.pop()
        else:
            conf = select_conf(results)

        weights = total_weights(conf.spec)
        print(
            f"{conf.spec} ({weights})  LR {conf.lr}  LEN {conf.sample_len}  B {conf.batch}"
        )
        model = LayeredModel2.parse(conf.spec)
        manager = Manager(model, conf.lr, regularization, 100000)
        manager.init(quiet=True)
        assert manager.model.total_weights(manager.weights) == total_weights(
            conf.spec
        )
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
        default="struct,lr,batch",
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
