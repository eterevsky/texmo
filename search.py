from bisect import bisect_left
from collections import namedtuple
import random
from statistics import median

from dataset import DataSet
from layered import LayeredModel2
from manager import Manager
from model import NCHAR
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


def log_distance(x, y):
    if x < y:
        return y / x
    else:
        return x / y


def change_layer_type(name, size, prev_layer, next_layer):
    if name == 'norm': return
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
        name = components[0]
        size = int(components[1])
        if name in ("dense", "rec"):
            out_size = size
        elif name == "gru":
            out_size = 3 * size
        elif name == "lstm":
            out_size = 4 * size

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


def layer_neighbors(layer, prev_layer, next_layer):
    components = layer.split(".")
    name = components[0]
    if name == 'norm': return
    params = components[1:]
    assert len(params) >= 1
    size = int(params[0])

    if name == "suffix":
        assert len(params) == 1
        if size > 2:
            yield f"suffix.{size-1}"
        yield f"suffix.{size+1}"
        return

    activation = "." + params[1] if len(params) > 1 else ""

    if size > 1:
        yield f"{name}.{size // 2}{activation}"
    yield f"{name}.{size * 2}{activation}"

    if activation:
        assert name != "lstm"
        if activation != "tanh":
            yield f"{name}.{size}.tanh"
        if activation != "relu":
            yield f"{name}.{size}.relu"

    for neighbor in change_layer_type(name, size, prev_layer, next_layer):
        yield neighbor


def spec_neighbors(spec):
    layers = spec.split("-")

    for i, layer in enumerate(layers):
        prev_layer = layers[i - 1] if i > 0 else None
        if prev_layer == 'norm':
            prev_layer = layers[i - 2] if i > 1 else None
        next_layer = layers[i + 1] if i < len(layers) - 1 else None
        if next_layer == 'norm':
            next_layer = layers[i + 2] if i < len(layers) - 2 else None
        for modified in layer_neighbors(layer, prev_layer, next_layer):
            yield "-".join(layers[:i] + [modified] + layers[i + 1 :])

    if not spec.startswith("suffix"):
        yield "suffix.2-" + spec
    if layers[0] == "suffix.2":
        yield "-".join(layers[1:])

    for i in range(1, len(layers)):
        if layers[i].startswith("norm"):
            yield "-".join(layers[:i] + layers[i + 1:])
        elif (
            not layers[i - 1].startswith("norm")
            and not layers[i - 1].startswith("suffix")
            and not layers[i].startswith("norm")
            and not layers[i].startswith("suffix")
        ):
            yield "-".join(layers[:i] + ["norm"] + layers[i + 1 :])
    if not layers[-1].startswith('norm') and not layers[-1].startswith('suffix'):
        yield '-'.join(layers + ['norm'])

    if len(layers) > 1:
        yield "-".join(layers[:-1])

    last_layer = layers[-1] if layers[-1] != "norm" else layers[-2]
    last_layer_size = int(last_layer.split(".")[1])
    out_size = min(256, last_layer_size)
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
        if name == 'norm': continue
        try:
            size = int(components[1])
        except IndexError:
            print('!!!!!!!!!!!!', layer_spec)
            print('!!!!!!!!!!!!', spec)
            continue
        weights += layer_weights(name, size, cur_size, 0)
        cur_size = cur_size * size if name == "suffix" else size
    weights += cur_size * NCHAR + NCHAR
    TOTAL_WEIGHTS_MEMO[spec] = weights
    return weights


def conf_neighbors(conf, max_weights=None):
    yield conf

    for spec in spec_neighbors(conf.spec):
        if max_weights is None or total_weights(spec) <= max_weights:
            yield conf._replace(spec=spec)

    # if conf.batch > 1:
    #     yield conf._replace(
    #         sample_len=conf.sample_len * 2, batch=conf.batch // 2
    #     )
    # if conf.sample_len > 2:
    #     yield conf._replace(
    #         sample_len=conf.sample_len // 2, batch=conf.batch * 2
    #     )

    yield conf._replace(batch=conf.batch * 2)
    if conf.batch > 1:
        yield conf._replace(batch=conf.batch // 2)

    ilr = LRS.index(conf.lr)
    if ilr > 0:
        yield conf._replace(lr=LRS[ilr - 1])
    if ilr < len(LRS) - 1:
        yield conf._replace(lr=LRS[ilr + 1])


class ConfResults(object):
    def __init__(self, conf, max_weights):
        self.conf = conf
        self.reports = []
        self.neighbors = list(conf_neighbors(conf, max_weights))

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
        for neighbor_conf in conf_results.neighbors:
            if neighbor_conf not in results:
                prev = conf
                prev_score = score
                best = neighbor_conf
                best_score = score
                best_count = 0
                break
            neighbor_results = results[neighbor_conf]
            neighbor_score = (
                score + min(0.1, score * 0.02) * neighbor_results.report_count
            )

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
            scores = " [" + ", ".join(f"{s:.4f}" for s in scores) + "]"
        else:
            scores = ""
        print(
            f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  B{conf.batch:4}  {score:.4f}{scores}"
        )
    print()


def search(
    data,
    learning_rate,
    sample_len,
    batch,
    regularization,
    time_limit,
    log,
    max_weights,
    start_spec,
):
    print(f"Training data: {data}")
    train_set = DataSet(data)

    assert max_weights is None or total_weights(start_spec) <= max_weights

    start_confs = [Configuration(start_spec, learning_rate, sample_len, batch)]
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
            quiet=True,
        )

        if conf not in results:
            results[conf] = ConfResults(conf, max_weights)
        results[conf].add_report(report)

        print_top(results)


if __name__ == "__main__":
    conf = ConfResults(
        Configuration(
            "suffix.3-dense.128.relu-dense.256.relu-dense.256.tanh",
            0.05,
            8,
            256,
        )
    )
    for n in conf.neighbors:
        print(n)
