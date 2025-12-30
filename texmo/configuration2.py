import argparse
import enum
import re
from typing import Optional, Iterable
import math

import jax
import jax.numpy as jnp

from .common import INF, console, itoa3, itoa3_aligned
from .model3 import Model3, build_model
from . import latency
from .tokens.tokenizer import Tokenizer


def is_valid_int(x: int) -> bool:
    return isinstance(x, int) and x >= 1


class Precision(enum.StrEnum):
    FP64 = enum.auto()
    FP32 = enum.auto()
    FP16 = enum.auto()
    BF16 = enum.auto()

    @property
    def dtype(self):
        match self:
            case Precision.FP64:
                return jnp.float64
            case Precision.FP32:
                return jnp.float32
            case Precision.FP16:
                return jnp.float16
            case Precision.BF16:
                return jax.dtypes.bfloat16

    @property
    def neighbors(self):
        match self:
            case Precision.FP64:
                return (Precision.FP32,)
            case Precision.FP32:
                return (Precision.FP64, Precision.FP16, Precision.BF16)
            case Precision.FP16:
                return (Precision.FP32, Precision.BF16)
            case Precision.BF16:
                return (Precision.FP32, Precision.BF16)


class Optimizer(enum.StrEnum):
    ADAM = enum.auto()
    FROMAGE = enum.auto()

    @property
    def letter(self):
        match self:
            case Optimizer.ADAM: return 'A'
            case Optimizer.FROMAGE: return 'F'


class Configuration2(object):
    __slots__ = ('model', 'precision', 'lr', 'length',
                 'batch', 'steps', 'optimizer', 'decay')

    def __init__(
        self,
        model: Model3,
        precision: Precision | str,
        lr: float,
        length: int,
        batch: int,
        steps: int,
        optimizer: Optimizer | str,
        decay: float,
    ):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'precision', Precision(precision))
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'length', length)
        object.__setattr__(self, 'batch', batch)
        object.__setattr__(self, 'steps', steps)
        assert optimizer is not None
        object.__setattr__(self, 'optimizer', Optimizer(optimizer))
        assert decay is not None
        object.__setattr__(self, 'decay', decay)

    def __setattr__(self, _key, _value):
        raise AttributeError('Configuration is immutable')

    @staticmethod
    def from_dict(d: dict) -> Configuration2:
        model = build_model(d['spec'])
        return Configuration2(
            model=model,
            precision=d['precision'],
            lr=d['lr'],
            length=d['length'],
            batch=d['batch'],
            steps=d['steps'],
            optimizer=d['optimizer'],
            decay=d['decay']
        )

    def __eq__(self, other: Configuration2) -> bool:
        return (
            self.lr == other.lr
            and self.precision == other.precision
            and self.length == other.length
            and self.batch == other.batch
            and self.steps == other.steps
            and self.optimizer == other.optimizer
            and self.decay == other.decay
            and self.model == other.model
        )

    def __hash__(self) -> int:
        return hash(
            (str(self.model), self.lr, self.length, self.batch,
             self.steps, self.precision, self.optimizer, self.decay)
        )

    def replace(
        self,
        model: Optional[Model3] = None,
        precision: Optional[Precision | str] = None,
        lr: Optional[float] = None,
        length: Optional[int] = None,
        batch: Optional[int] = None,
        steps: Optional[int] = None,
        optimizer: Optional[Optimizer | str] = None,
        decay: Optional[float] = None,
    ) -> Configuration2:
        model = model or self.model
        lr = lr or self.lr
        length = length or self.length
        batch = batch or self.batch
        steps = steps or self.steps
        precision = precision or self.precision
        optimizer = optimizer or self.optimizer
        decay = decay or self.decay
        return Configuration2(
            model=model, precision=precision, lr=lr, length=length, batch=batch, steps=steps,
            optimizer=optimizer, decay=decay
        )

    def to_dict(self) -> dict:
        return {
            'spec': str(self.model),
            'precision': str(self.precision),
            'lr': self.lr,
            'length': self.length,
            'batch': self.batch,
            'steps': self.steps,
            'optimizer': str(self.optimizer),
            'decay': self.decay,
        }

    def __repr__(self) -> str:
        return f"Configuration2('{self.model}', '{self.precision}', {self.lr}, {self.length}, {self.batch}, {self.steps}, '{self.optimizer}', {self.decay})"

    def __str__(self) -> str:
        model = str(self.model)
        steps = f'  S{itoa3(self.steps)}' if self.steps else ''
        decay = '' if self.decay == 1 else f'*{self.decay:.4f}'
        optimizer = f'{self.optimizer.letter}{self.lr:.4f}{decay}'

        return (
            f'{model} ({itoa3(self.model.weights)})  {self.precision}   '
            + f'{self.batch}×{self.length}  S{self.steps}  '
            + self.learning_str
        )

    def aligned_str(self) -> str:
        model = str(self.model)
        return (
            f'{self.batch:>5d}×{self.length:<5d}  {self.learning_str:<10} S{self.steps:>5d}  '
            + f'{self.precision}  {model} ({self.model.weights})'
        )

    def is_valid(self) -> bool:
        return (
            self.model.is_valid()
            and self.lr > 0
            and is_valid_int(self.length)
            and is_valid_int(self.batch)
            and is_valid_int(self.steps)
            and type(self.precision) == Precision
            and type(self.optimizer) == Optimizer
            and 0 < self.decay <= 1
        )

    @property
    def learning_str(self) -> str:
        render_lr = lambda lr: str(int(lr)) if lr >= 1 else f'1/{str(int(1/lr))}'
        lr = render_lr(self.lr)
        if self.decay == 1:
            final = ''
        else:
            final = self.lr * self.decay
            final = '→' + render_lr(final)

        return f'{self.optimizer.letter}{lr}{final}'


    @property
    def tokenizer(self) -> Tokenizer:
        return self.model.input.tokenizer

    @property
    def tokens_name(self) -> str:
        return self.model.input.tokens_name

    def neighbors(self) -> Iterable[Configuration2]:
        if self.steps > 2:
            yield self.replace(steps=self.steps // 2)
        for precision in self.precision.neighbors:
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
        if self.decay < 1:
            yield self.replace(decay=self.decay * 2)
        yield self.replace(decay=self.decay / 2)


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

        value2 = value // 2 if isinstance(value, int) and value >= 2 else value / 2

        if value2 >= self.min:
            yield value2


Limits = Optional[float | tuple[float, float]]


def _parse_number(arg: str, num_type: type) -> int|float:
    if arg == 'inf':
        return INF
    return num_type(arg)


def _parse_interval(arg: str, num_type: type) -> tuple:
    if arg is None or arg == '':
        return None
    comps = arg.split('-')
    assert len(comps) in (1, 2)
    comps = tuple(_parse_number(c, num_type) for c in comps)
    if len(comps) == 1:
        return (comps[0], comps[0])
    else:
        return comps


class Template(object):
    def __init__(
        self,
        spec: Optional[str],  # Exact match or a regex
        lr: Optional[float | tuple[float, float]],
        length: Limits,
        batch: Limits,
        steps: Limits,
        max_weights: Limits,
        precision: list[Precision],
        optimizer: list[Optimizer],
        decay: Optional[float | tuple[float, float]]
    ):
        self.update_spec(spec)
        self.max_weights = Bounds(max_weights, 16)

        self.length = Bounds(length, 1)
        self.batch = Bounds(batch, 1)

        self.precision = precision

        self.optimizer = optimizer
        self.lr = Bounds(lr, 0)
        self.decay = Bounds(decay, 0)

        self.steps = Bounds(steps, 2)

        self._conf_neighbors_cache = {}

    @staticmethod
    def from_args(args: argparse.Namespace):
        if not args.precision:
            precision = list(Precision)
        else:
            precision = list(map(Precision, args.precision.split(',')))

        if not args.optimizer:
            optimizer = [Optimizer.ADAM]
        else:
            optimizer = list(map(Optimizer, args.optimizer.split(',')))

        return Template(
            spec=args.spec,
            lr=_parse_interval(args.lr, float),
            length=_parse_interval(args.length, int),
            batch=_parse_interval(args.batch, int),
            steps=_parse_interval(args.steps, int),
            max_weights=_parse_interval(args.weights, int),
            precision=precision,
            optimizer=optimizer,
            decay=_parse_interval(args.decay, float)
        )

    def __str__(self):
        precision = ','.join(map(str, self.precision))
        optimizer = ','.join(map(str, self.optimizer))
        spec = self.spec if self.spec is not None else self.regex
        return (
            f'Template(spec={repr(spec)}, '
            + f'precision=\'{precision}\', '
            + f'lr={self.lr}, length={self.length}, '
            + f'batch={self.batch}, '
            + f'steps={self.steps}, '
            + f'max_weights={self.max_weights}, '
            + f'optimizer=\'{optimizer}\', '
            + f'decay={self.decay})'
        )

    def update_spec(self, spec: Optional[str]):
        if spec is None:
            self.regex = None
            self.spec = None
        else:
            try:
                model = build_model(spec)
            except Exception:
                model = None
            if model is None:
                self.regex = re.compile(spec)
                self.spec = None
            else:
                self.regex = None
                self.spec = spec

    def update_weights(self, weights: Optional[str]):
        self.max_weights =  Bounds(_parse_interval(weights, int), 2)

    def update_length(self, value: Optional[str]):
        self.length =  Bounds(_parse_interval(value, int), 1)

    def update_batch(self, value: Optional[str]):
        self.batch =  Bounds(_parse_interval(value, int), 1)

    def update_precision(self, value: list[Precision]):
        assert type(value) is list
        assert len(value) > 0
        assert type(value[0]) is Precision
        self.precision = value

    def update_optimizer(self, value: list[Optimizer]):
        assert type(value) is list
        assert len(value) > 0
        assert type(value[0]) is Optimizer
        self.optimizer = value

    def update_lr(self, value: Optional[str]):
        self.lr =  Bounds(_parse_interval(value, float), 0)

    def update_decay(self, value: Optional[str]):
        self.decay =  Bounds(_parse_interval(value, float), 0)

    def update_steps(self, value: Optional[str]):
        self.steps =  Bounds(_parse_interval(value, int), 1)

    def match_model(self, model: Model3) -> bool:
        if model.weights > self.max_weights.max:
            return False
        if self.spec is not None:
            return self.spec == str(model)
        if self.regex is not None:
            return bool(self.regex.fullmatch(str(model)))
        return True

    def match(self, conf: Configuration2) -> bool:
        return (
            self.match_model(conf.model)
            and conf.precision in self.precision
            and self.lr.match(conf.lr)
            and self.length.match(conf.length)
            and self.batch.match(conf.batch)
            and self.steps.match(conf.steps)
            and conf.optimizer in self.optimizer
            and self.decay.match(conf.decay)
        )

    def _conf_neighbors(
        self, conf: Configuration2
    ) -> Iterable[Configuration2]:
        for precision in conf.precision.neighbors:
            if precision in self.precision:
                yield conf.replace(precision=precision)
        for steps in self.steps.neighbors(conf.steps):
            assert steps >= 2
            yield conf.replace(steps=steps)
        for lr in self.lr.neighbors(conf.lr):
            yield conf.replace(lr=lr)
        for batch in self.batch.neighbors(conf.batch):
            yield conf.replace(batch=batch)
        for length in self.length.neighbors(conf.length):
            yield conf.replace(length=length)
        for decay in self.decay.neighbors(conf.decay):
            yield conf.replace(decay=decay)
        for model in conf.model.neighbors():
            match = self.match_model(model)
            if self.match_model(model):
                yield conf.replace(model=model)


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
    lr = template.lr.pick_default(1 / 128)
    length = template.length.pick_default(8)
    batch = template.batch.pick_default(1)
    steps = template.steps.pick_default(2)
    decay = template.decay.pick_default(1.0)
    optimizer = template.optimizer[0]

    if spec is None:
        if template.spec is not None:
            spec = template.spec
        else:
            found = False
            for tokens in ('bits.1', 'bits.2', 'bits.4', 'bits.8'):
                for oh in ('', '.oh'):
                    for pos in ('', '+bp', '+pos'):
                        if found: break
                        input_spec = tokens + oh + pos
                        for layers in (
                            '',
                            'dense.1.gelu',
                            'rec.1.tanh',
                            'mingru.1',
                            'mgru.1',
                            'gru.1',
                            'lstm.1',
                        ):
                            full_spec = input_spec + '|' + layers
                            model = build_model(full_spec)
                            if not model.is_valid():
                                continue
                            if template.match_model(model):
                                spec = str(model)
                                found = True

    if spec is not None:
        return Configuration2(
                        model=build_model(spec),
                        precision=template.precision[0],
                        lr=lr,
                        length=length,
                        batch=batch,
                        steps=steps,
                        optimizer=optimizer,
                        decay=decay,
                    )


    raise RuntimeError(
        "Can't pick up a default model that would fit the template"
    )
