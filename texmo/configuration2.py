import argparse
import enum
import re
from typing import Optional, Iterable, Self
import math

import jax
import jax.numpy as jnp

from .common import INF, itoa3, itoa3_aligned
from .model3 import Model3, build_model
from . import latency
from .tokens.tokenizer import Tokenizer


Self = ()


def is_valid_int(x: int) -> bool:
    return isinstance(x, int) and x >= 1


class Precision(enum.StrEnum):
    FP32 = enum.auto()
    FP16 = enum.auto()
    BF16 = enum.auto()

    @property
    def dtype(self):
        match self:
            case Precision.FP32:
                return jnp.float32
            case Precision.FP16:
                return jnp.float16
            case Precision.BF16:
                return jax.dtypes.bfloat16


class Configuration2(object):
    __slots__ = ('model', 'precision', 'lr', 'length', 'batch', 'steps')

    def __init__(
        self,
        model: Model3,
        precision: str,
        lr: float,
        length: int,
        batch: int,
        steps: int,
    ):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'precision', Precision(precision))
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'length', length)
        object.__setattr__(self, 'batch', batch)
        object.__setattr__(self, 'steps', steps)

    def __setattr__(self, _key, _value):
        raise AttributeError('Configuration is immutable')

    @staticmethod
    def from_dict(d: dict) -> Self:
        model = build_model(d['spec'])
        return Configuration2(
            model=model,
            precision=d['precision'],
            lr=d['lr'],
            length=d['length'],
            batch=d['batch'],
            steps=d['steps'],
        )

    def __eq__(self, other: Self) -> bool:
        return (
            self.lr == other.lr
            and self.precision == other.precision
            and self.length == other.length
            and self.batch == other.batch
            and self.steps == other.steps
            and self.model == other.model
        )

    def __hash__(self) -> int:
        return hash(
            (str(self.model), self.lr, self.length, self.batch, self.steps)
        )

    def replace(
        self,
        model: Optional[Model3] = None,
        precision: Optional[str] = None,
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
        precision = precision or self.precision
        return Configuration2(
            model=model, precision=precision, lr=lr, length=length, batch=batch, steps=steps
        )

    def to_dict(self) -> dict:
        return {
            'spec': str(self.model),
            'precision': str(self.precision),
            'lr': self.lr,
            'length': self.length,
            'batch': self.batch,
            'steps': self.steps,
        }

    def __repr__(self) -> str:
        return f"Configuration2('{self.model}', '{self.precision}', {self.lr}, {self.length}, {self.batch}, {self.steps})"

    def __str__(self) -> str:
        model = str(self.model)
        steps = f'  S{itoa3(self.steps)}' if self.steps else ''
        return (
            f'{model} ({itoa3(self.model.weights)})  {self.precision}   '
            + f'LEN{itoa3(self.length)}  '
            + f'B{itoa3(self.batch)}  '
            + f'LR{self.lr:.4f}{steps}'
        )

    def aligned_str(self) -> str:
        model = str(self.model)
        return (
            f'L{itoa3_aligned(self.length)} B{itoa3_aligned(self.batch)} '
            + f'LR{self.lr:.4f}   S{itoa3_aligned(self.steps)}  '
            + f'{self.precision}  {model} ({self.model.weights})'
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
        return self.tokenizer.tokenset.name

    def neighbors(self) -> Iterable['Configuration2']:
        if self.steps > 2:
            yield self.replace(steps=self.steps // 2)
        for precision in Precision:
            if precision != self.precision:
                yield self.replace(precision=precision)
        yield self.replace(steps=self.steps * 2)
        yield self.replace(lr=self.lr / 2)
        yield self.replace(lr=self.lr * 2)
        if self.batch > 1:
            yield self.replace(batch=self.batch // 2)
        yield self.replace(batch=self.batch * 2)
        if self.length > 1:
            yield self.replace(length=self.length // 2)
        yield self.replace(length=self.length * 2)
        for model in self.model.neighbors():
            yield self.replace(model=model)


class Bounds(object):
    def __init__(
        self, limits: Optional[float | tuple[float, float]], min_value: float
    ):
        if limits is None:
            self.min = min_value
            self.max = INF
        else:
            try:
                self.min, self.max = limits
            except TypeError:
                self.min = self.max = limits

    def __str__(self) -> str:
        return f'({self.min}, {self.max})'

    def match(self, value: float) -> bool:
        return self.min <= value <= self.max

    def pick_default(self, default: float) -> float:
        if self.min <= default <= self.max:
            return default
        elif self.min == 0:
            return self.max
        else:
            mid = math.sqrt(self.min * self.max)

        round_mid = 2 ** round(math.log2(mid))
        assert self.min <= round_mid <= self.max
        return round_mid

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
    comps = arg.split('-')
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
        max_weights: Limits,
        precision: list[Precision],
    ):
        self.regex = re.compile(spec_regex) if spec_regex else None
        self.lr = Bounds(lr, 0)
        self.length = Bounds(length, 1)
        self.batch = Bounds(batch, 1)
        self.steps = Bounds(steps, 2)
        self.max_weights = Bounds(max_weights, 16)
        self.precision = precision
        self._conf_neighbors_cache = {}

    @staticmethod
    def from_args(args: argparse.Namespace):
        if not args.precision:
            precision = list(Precision)
        else:
            precision = list(map(Precision, args.precision.split(',')))
        return Template(
            spec_regex=args.spec_regex,
            lr=_parse_interval(args.lr, float),
            length=_parse_interval(args.length, int),
            batch=_parse_interval(args.batch, int),
            steps=_parse_interval(args.steps, int),
            max_weights=_parse_interval(args.weights, int),
            precision=precision,
        )

    def __str__(self):
        return (
            f'Template({self.regex}, '
            + f'precision={self.precision}, '
            + f'lr={self.lr}, length={self.length}, '
            + f'batch={self.batch}, '
            + f'steps={self.steps}, '
            + f'max_weights={self.max_weights})'
        )

    def update_regex(self, regex: Optional[str]):
        self.regex = re.compile(regex) if regex else None

    def match_model(self, model: Model3) -> bool:
        if model.weights > self.max_weights.max:
            return False
        return self.regex is None or bool(self.regex.fullmatch(str(model)))

    def match(self, conf: Configuration2) -> bool:
        return (
            self.match_model(conf.model)
            and conf.precision in self.precision
            and self.lr.match(conf.lr)
            and self.length.match(conf.length)
            and self.batch.match(conf.batch)
            and self.steps.match(conf.steps)
        )

    def _conf_neighbors(
        self, conf: Configuration2
    ) -> Iterable[Configuration2]:
        for steps in self.steps.neighbors(conf.steps):
            yield conf.replace(steps=steps)
        for lr in self.lr.neighbors(conf.lr):
            yield conf.replace(lr=lr)
        for batch in self.batch.neighbors(conf.batch):
            yield conf.replace(batch=batch)
        for length in self.length.neighbors(conf.length):
            yield conf.replace(length=length)
        for model in conf.model.neighbors():
            if self.match_model(model):
                yield conf.replace(model=model)
        for precision in self.precision:
            if precision != conf.precision:
                yield conf.replace(precision=precision)
            


def conf_neighbors(
    conf: Configuration2, template: Template
) -> Iterable[Configuration2]:
    with latency.timer('conf_neighbors'):
        neighbors = template._conf_neighbors_cache.get(conf)

        if neighbors is not None:
            return neighbors

        neighbors = list(template._conf_neighbors(conf))
        template._conf_neighbors_cache[conf] = neighbors

        return neighbors


def default_from_template(
    template: Template, spec: Optional[str]
) -> Configuration2:
    lr = template.lr.pick_default(1 / 32)
    length = template.length.pick_default(8)
    batch = template.batch.pick_default(1)
    steps = template.steps.pick_default(2)

    if spec is not None:
        return Configuration2(
            model=build_model(spec),
            precision=template.precision[0],
            lr=lr,
            length=length,
            batch=batch,
            steps=steps,
        )

    for tokens in ('tokens.2.raw.b1', 'tokens.4.raw.bh', 'tokens.256.raw.b8'):
        for pos in ('', '-pos.16', '-pos.256'):
            for emb in ('', '-emb.64', '-emb.256'):
                input_spec = tokens + pos + emb
                for layers in (
                    '',
                    'dense.1.gelu',
                    'rec.1.gelu',
                    'gru.1',
                    'lstm.1',
                ):
                    spec = input_spec + '|' + layers
                    model = build_model(spec)
                    if template.match_model(model):
                        return Configuration2(
                            model=model,
                            precision=template.precision[0],
                            lr=lr,
                            length=length,
                            batch=batch,
                            steps=steps,
                        )
    raise RuntimeError(
        "Can't pick up a default model that would fit the template"
    )
