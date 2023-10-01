import argparse
import math
import re
from collections import namedtuple
from typing import Any

from .common import INF, is_power2, power2_neighbors
from .model2 import Model2, build_model
from .tokens import parse_token_set_name, get_tokenizer

TOKEN_TYPES = ("all", "bits1", "bits2", "bits4", "dist2", "dist4", "dist8")

Configuration = namedtuple(
    "Configuration",
    [
        "model",
        "ntokens",
        "token_type",
        "token_processing",
        "lr",
        "sample_len",
        "batch",
        "t",
    ],
)


def conf_to_dict(conf: Configuration) -> dict[str, Any]:
    return {
        "ntokens": conf.model.ntokens,
        "spec": str(conf.model),
        "lr": conf.lr,
        "sample_len": conf.sample_len,
        "batch": conf.batch,
        "t": conf.t,
        "ntokens": conf.ntokens,
        "token_type": conf.token_type,
        "token_processing": conf.token_processing,
    }


def conf_from_dict(conf_dict: dict) -> Configuration:
    model = build_model(conf_dict["ntokens"], conf_dict["spec"])
    return Configuration(
        model,
        conf_dict["ntokens"],
        conf_dict["token_type"],
        conf_dict["token_processing"],
        conf_dict["lr"],
        conf_dict["sample_len"],
        conf_dict["batch"],
        conf_dict["t"],
    )


def conf_is_valid(conf: Configuration) -> bool:
    tokenizer = get_tokenizer(conf_tokens_name(conf))
    return (
        tokenizer is not None
        and conf.model is not None
        and conf.model.is_valid()
        # and is_power2(conf.lr)
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and is_power2(conf.t)
        and is_power2(conf.ntokens)
        and conf.ntokens >= 2
        and conf.token_type in TOKEN_TYPES
        and conf.token_processing in ("raw", "capswords")
    )


def conf_to_string(conf: Configuration) -> str:
    tokens = conf_tokens_name(conf)
    return (
        f"{tokens} | {conf.model} ({conf.model.weights})  LEN{conf.sample_len}  "
        + f"B{conf.batch}  LR{conf.lr:.4f}  T{conf.t}"
    )


def conf_tokens_name(conf: Configuration) -> str:
    return f"tokens{conf.ntokens}_{conf.token_processing}_{conf.token_type}"


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
        ntokens=None,
        token_type=("all", "bits1", "bits2", "bits4"),
        token_processing=("raw", "capswords"),
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
        self.ntokens = make_bounds(ntokens)
        self.token_type = token_type
        self.token_processing = token_processing

    @staticmethod
    def from_args(args: argparse.Namespace):
        return Template(
            spec_regex=args.spec_regex,
            batch=parse_interval(args.batch, int),
            lr=parse_interval(args.lr, float),
            sample_len=parse_interval(args.sample_len, int),
            t=parse_interval(args.time, int),
            max_weights=args.max_weights,
            ntokens=parse_interval(args.ntokens, int),
            token_type=args.token_type.split(","),
            token_processing=args.token_processing.split(","),
        )

    def match_conf(self, conf):
        return (
            match_bounds(self.lr, conf.lr)
            and match_bounds(self.sample_len, conf.sample_len)
            and match_bounds(self.batch, conf.batch)
            and match_bounds(self.t, conf.t)
            and match_bounds(self.ntokens, conf.ntokens)
            and conf.token_type in self.token_type
            and conf.token_processing in self.token_processing
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
            ntokens=self.ntokens,
            token_type=self.token_type,
            token_processing=self.token_processing,
        )


def _pick_default_value(interval: tuple[float, float]) -> float:
    m = (math.log2(interval[0]) + math.log2(interval[1])) / 2
    value = 2 ** round(m)
    assert interval[0] <= value <= interval[1]
    return value


def default_from_template(template: Template, spec: str) -> Configuration:
    lr = _pick_default_value(template.lr)
    sample_len = _pick_default_value(template.sample_len)
    batch = _pick_default_value(template.batch)
    t = _pick_default_value(template.t)
    ntokens = template.ntokens[0]

    if ntokens <= 16:
        if "raw" in template.token_processing:
            token_processing = "raw"
        else:
            token_processing = template.token_processing[0]
        if "bits1" in template.token_type:
            token_type = "bits1"
        elif "bits2" in template.token_type:
            token_type = "bits2"
        else:
            token_type = template.token_type[0]
    else:
        if "capswords" in template.token_processing:
            token_processing = "capswords"
        else:
            token_processing = template.token_processing[0]
        if "bits4" in template.token_type:
            token_type = "bits4"
        elif "bits2" in template.token_type:
            token_type = "bits2"
        else:
            token_type = template.token_type[0]

    if spec is not None:
        model = build_model(ntokens, spec)
        if not template.match_model:
            raise RuntimeError("Default model doesn't fit the template")
    else:
        for spec in (
            "suffix.2",
            "dense.1.gelu",
            "dense.1.gelu",
            "rec.1.relu",
            "rec.1.gelu",
            "gru.1",
            "mgru.1",
            "lstm.1",
        ):
            model = build_model(ntokens, spec)
            if template.match_model(model):
                break
        if not template.match_model:
            raise RuntimeError(
                "Can't pick up a default model that would fit the template"
            )
    return Configuration(
        model=model,
        lr=lr,
        sample_len=sample_len,
        batch=batch,
        t=t,
        ntokens=ntokens,
        token_type=token_type,
        token_processing=token_processing,
    )


_model_neighbors: dict[Model2, list[Model2]] = {}


def reset_neighbors_cache():
    global _model_neighbors
    _model_neighbors = {}


_NEIGHBOR_TOKEN_TYPES = {
    "all": ["dist8", "bits1"],
    "dist2": ["dist4", "bits4"],
    "dist4": ["dist2", "dist8", "bits2"],
    "dist8": ["dist4", "bits1", "all"],
    "bits1": ["bits2", "all", "dist8"],
    "bits2": ["bits1", "bits4", "dist4"],
    "bits4": ["bits2", "dist2"],
}

_NEIGHBOR_TOKEN_PROCESSING = {
    "raw": ["capswords"],
    # "caps": ["raw", "capswords"],
    "capswords": ["raw"],
}


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
        if not template.match_model(neighbor_model):
            continue

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

    for x in power2_neighbors(conf.ntokens):
        if match_bounds(template.ntokens, x):
            new_model = build_model(x, str(conf.model))
            if not template.match_model(new_model):
                continue
            conf_mod = conf._replace(ntokens=x, model=new_model)
            if get_tokenizer(conf_tokens_name(conf_mod)):
                yield conf_mod

    for x in _NEIGHBOR_TOKEN_TYPES[conf.token_type]:
        if x in template.token_type:
            conf_mod = conf._replace(token_type=x)
            name = conf_tokens_name(conf_mod)
            if get_tokenizer(conf_tokens_name(conf_mod)):
                yield conf_mod

    for x in ("raw", "capswords"):
        if x in template.token_processing and x != conf.token_processing:
            conf_mod = conf._replace(token_processing=x)
            if get_tokenizer(conf_tokens_name(conf_mod)):
                yield conf_mod
