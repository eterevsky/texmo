import argparse
import re
from typing import Optional, Self, Iterable

from .common import INF, itoa3
from .model3 import Model3, build_model
from . import latency
from .tokens.tokenizer import Tokenizer


def is_valid_int(x: int) -> bool:
    return isinstance(x, int) and x >= 1


class Configuration2(object):
    __slots__ = ("model", "lr", "length", "batch", "steps")

    def __init__(self, model: Model3, lr: float, length: int, batch: int, steps: int):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "batch", batch)
        object.__setattr__(self, "steps", steps)

    def __setattr__(self, _key, _value):
        raise AttributeError("Configuration is immutable")

    @staticmethod
    def from_dict(d: dict) -> Self:
        model = build_model(d["spec"])
        return Configuration2(
            model=model,
            lr=d["lr"],
            length=d["length"],
            batch=d["batch"],
            steps=d["steps"],
        )

    def __eq__(self, other: Self) -> bool:
        return (
            self.lr == other.lr
            and self.length == other.length
            and self.batch == other.batch
            and self.steps == other.steps
            and self.model == other.model
        )

    def __hash__(self) -> int:
        return hash((str(self.model), self.lr, self.length, self.batch, self.steps))

    def replace(
        self,
        model: Optional[Model3] = None,
        lr: Optional[float] = None,
        length: Optional[int] = None,
        batch: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> Self:
        model = model or self.model
        lr = lr or self.lr
        length = length or self.length
        batch = batch or self.batch
        steps = steps or self.steps
        return Configuration2(
            model=model, lr=lr, length=length, batch=batch, steps=steps
        )

    def to_dict(self) -> dict:
        return {
            "spec": str(self.model),
            "lr": self.lr,
            "length": self.length,
            "batch": self.batch,
            "steps": self.steps,
        }

    def __repr__(self) -> str:
        return f"Configuration2('{self.model}', {self.lr}, {self.length}, {self.batch}, {self.steps})"

    def __str__(self) -> str:
        model = str(self.model)
        steps = f"  S{itoa3(self.steps)}" if self.steps else ""
        return (
            f"{model} ({itoa3(self.model.weights)})  LEN{itoa3(self.length)}"
            + f"  B{itoa3(self.batch)}  LR{self.lr:.4f}{steps}"
        )

    def is_valid(self) -> bool:
        return (
            self.model.is_valid()
            and self.lr > 0
            and is_valid_int(self.length)
            and is_valid_int(self.batch)
            and is_valid_int(self.steps)
        )

    @property
    def tokenizer(self) -> Tokenizer:
        return self.model.input.tokenizer
    
    @property
    def tokens_name(self) -> str:
        return self.tokenizer.token_set.name


class Bounds(object):
    def __init__(self, limits: Optional[float | tuple[float, float]], min_value: float):
        if limits is None:
            self.min = min_value
            self.max = INF
        else:
            try:
                self.min, self.max = limits
            except TypeError:
                self.min = self.max = limits

    def match(self, value: float) -> bool:
        return self.min <= value <= self.max

    def neighbors(self, value: float | int) -> Iterable[float | int]:
        assert self.match(value)

        if value * 2 <= self.max:
            yield value * 2

        value2 = value // 2 if isinstance(value, int) else value / 2

        if value2 >= self.min:
            yield value2


Limits = Optional[float | tuple[float, float]]


def _parse_interval(arg: str, num_type: type) -> tuple:
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
        spec_regex: Optional[str],
        lr: Optional[float | tuple[float, float]],
        length: Limits,
        batch: Limits,
        steps: Limits,
        max_weights: Optional[int] = None,
    ):
        self.regex = re.compile(spec_regex) if spec_regex else None
        self.lr = Bounds(lr, 0)
        self.length = Bounds(length, 1)
        self.batch = Bounds(batch, 1)
        self.steps = Bounds(steps, 1)
        self.max_weights = max_weights or INF

    @staticmethod
    def from_args(args: argparse.Namespace):
        return Template(
            spec_regex=args.spec_regex,
            lr=_parse_interval(args.lr, float),
            length=_parse_interval(args.length, int),
            batch=_parse_interval(args.batch, int),
            steps=_parse_interval(args.steps, int),
            max_weights=args.max_weights,
        )

    def match_model(self, model: Model3) -> bool:
        if model.weights > self.max_weights:
            return False
        return self.regex is None or self.regex.fullmatch(str(model))

    def match(self, conf: Configuration2) -> bool:
        return (
            self.match_model(conf.model)
            and self.lr.match(conf.lr)
            and self.length.match(conf.length)
            and self.batch.match(conf.batch)
            and self.steps.match(conf.steps)
        )


def default_from_template(template: Template, spec: Optional[str]) -> Configuration2:
    lr = template.lr.pick_default(1 / 32)
    length = template.length.pick_default(64)
    batch = template.batch.pick_default(1)
    steps = template.steps.pick_default(256)

    for tokens in ("tokens.2", "bytes", "tokens.256"):
        for pos in ("", "-pos.16", "-pos.256"):
            for emb in ("", "-emb.64", "-emb.256"):
                input_spec = tokens + pos + emb
                for layers in (
                    "",
                    "dense.1.gelu",
                    "rec.1.gelu",
                    "gru.1",
                    "lstm.1",
                ):
                    spec = input_spec + "|" + layers
                    model = build_model(spec)
                    if template.match_model(model):
                        return Configuration2(
                            model=model,
                            lr=lr,
                            length=length,
                            batch=batch,
                            steps=steps,
                        )
    raise RuntimeError("Can't pick up a default model that would fit the template")


def reset_neighbors_cache():
    pass


def _conf_neighbors(
    conf: Configuration2, template: Template
) -> Iterable[Configuration2]:
    for model in conf.model.neighbors():
        if template.match_model(model):
            yield conf.replace(model=model)
    for lr in template.lr.neighbors(conf.lr):
        yield conf.replace(lr=lr)
    for length in template.length.neighbors(conf.length):
        yield conf.replace(length=length)
    for batch in template.batch.neighbors(conf.batch):
        yield conf.replace(batch=batch)
    for steps in template.steps.neighbors(conf.steps):
        yield conf.replace(steps=steps)


def conf_neighbors(
    conf: Configuration2, template: Template
) -> Iterable[Configuration2]:
    with latency.timer("conf_neighbors"):
        return list(_conf_neighbors(conf, template))
