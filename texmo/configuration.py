import argparse
import math
import re
from collections import namedtuple
from typing import Any

from .common import INF, is_power2, power2_neighbors
from .model2 import Model2, build_model
from .record import TrainingRecord

Configuration = namedtuple(
    "Configuration",
    [
        "model",
        "lr",
        "sample_len",
        "batch",
        "t",
    ],
)


def conf_from_record(record: TrainingRecord) -> Configuration:
    model = build_model(record.ntokens, record.model_spec)
    return Configuration(
        model,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.planned_time_s,
    )


def conf_to_dict(conf: Configuration) -> dict[str, Any]:
    return {
        "ntokens": conf.model.ntokens,
        "spec": str(conf.model),
        "lr": conf.lr,
        "sample_len": conf.sample_len,
        "batch": conf.batch,
        "t": conf.t,
    }


def conf_from_dict(conf_dict: dict) -> Configuration:
    model = build_model(conf_dict["ntokens"], conf_dict["spec"])
    return Configuration(
        model,
        conf_dict["lr"],
        conf_dict["sample_len"],
        conf_dict["batch"],
        conf_dict["t"],
    )


def conf_is_valid(conf: Configuration) -> bool:
    return (
        conf.model is not None
        and conf.model.is_valid()
        and is_power2(conf.lr)
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and is_power2(conf.t)
    )


def conf_to_string(conf: Configuration) -> str:
    return (
        f"{conf.model} ({conf.model.weights})  LEN{conf.sample_len}  B{conf.batch}  LR{conf.lr:.4f}  T{conf.t}"
    )


def match_bounds(bounds, value):
    return bounds is None or bounds[0] <= value <= bounds[1]


def make_bounds(v):
    if v is None:
        return None
    try:
        return tuple(v)
    except TypeError:
        return (v, v)


def parse_interval(arg: str, num_type) -> tuple:
    if arg is None:
        return None
    comps = arg.split("-")
    assert len(comps) in (1, 2)
    if len(comps) == 1:
        v = num_type(comps[0])
        return (v, v)
    else:
        return num_type(comps[0]), num_type(comps[1])


class Template(object):
    def __init__(
        self,
        spec_regex=None,
        lr=None,
        sample_len=None,
        batch=None,
        t=None,
        max_weights=None,
    ):
        self.regex = re.compile(spec_regex) if spec_regex is not None else None
        self.lr = make_bounds(lr)
        self.sample_len = make_bounds(sample_len)
        self.batch = make_bounds(batch)
        self.t = make_bounds(t)
        self.max_weights = max_weights

    @staticmethod
    def from_args(args: argparse.Namespace):
        return Template(
            spec_regex=args.spec_regex,
            batch=parse_interval(args.batch, int),
            lr=parse_interval(args.lr, float),
            sample_len=parse_interval(args.sample_len, int),
            t=parse_interval(args.time, int),
            max_weights=args.max_weights,
        )

    def match_conf(self, conf):
        return (
            match_bounds(self.lr, conf.lr)
            and match_bounds(self.sample_len, conf.sample_len)
            and match_bounds(self.batch, conf.batch)
            and match_bounds(self.t, conf.t)
            and self.match_model(conf.model)
        )

    def match_model(self, model):
        if self.max_weights is not None:
            if model.weights > self.max_weights:
                return False
        return self.regex is None or self.regex.fullmatch(str(model))

    def clone(self):
        return Template(
            spec_regex=self.regex,
            lr=self.lr,
            sample_len=self.sample_len,
            batch=self.batch,
            t=self.t,
            max_weights=self.max_weights,
        )


def _pick_default_value(interval: tuple[float, float]) -> float:
    m = (math.log2(interval[0]) + math.log2(interval[1])) / 2
    value = 2 ** round(m)
    assert interval[0] <= value <= interval[1]
    return value


def default_from_template(template: Template) -> Configuration:
    lr = _pick_default_value(template.lr)
    sample_len = _pick_default_value(template.sample_len)
    batch = _pick_default_value(template.batch)
    t = _pick_default_value(template.t)
    return Configuration(
        model=None, lr=lr, sample_len=sample_len, batch=batch, t=t
    )


_model_neighbors: dict[Model2, list[Model2]] = {}


def conf_neighbors(conf: Configuration, template: Template):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    cache = _model_neighbors.get(conf.model)
    if cache is None:
        cache = []
        _model_neighbors[conf.model] = cache
        for neighbor_model in conf.model.neighbors():
            if template.match_model(neighbor_model):
                cache.append(neighbor_model)

    for neighbor_model in cache:
        if not template.match_model(neighbor_model): continue

        yield conf._replace(model=neighbor_model)

        # For neighbor specs that add or remove layers we'll also add
        # configurations with increased/decreased time limit.

        if len(neighbor_model.layers) > len(conf.model.layers) and match_bounds(
            template.t, conf.t * 2
        ):
            yield conf._replace(model=neighbor_model, t=conf.t * 2)
            if match_bounds(template.lr, conf.lr / 2):
                yield conf._replace(
                    model=neighbor_model, t=conf.t * 2, lr=conf.lr / 2
                )
        elif len(neighbor_model.layers) < len(
            conf.model.layers
        ) and match_bounds(template.t, conf.t // 2):
            yield conf._replace(model=neighbor_model, t=conf.t // 2)
            if match_bounds(template.lr, conf.lr * 2):
                yield conf._replace(
                    model=neighbor_model, t=conf.t // 2, lr=conf.lr * 2
                )

    for x in (conf.lr / 2, conf.lr * 2):
        if match_bounds(template.lr, x):
            yield conf._replace(lr=x)

    for x in power2_neighbors(conf.sample_len):
        if match_bounds(template.sample_len, x):
            yield conf._replace(sample_len=x)

    for x in power2_neighbors(conf.batch):
        if match_bounds(template.batch, x):
            yield conf._replace(batch=x)

    for x in power2_neighbors(conf.t):
        if match_bounds(template.t, x):
            yield conf._replace(t=x)
