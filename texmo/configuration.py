import argparse
import re
from typing import Optional, Iterable
import math

from .common import INF, itoa3
from .model import ModelDef, build_model_def
from .precision import Precision
from . import latency
from .tokens.tokenizer import Tokenizer


def is_valid_int(x: int) -> bool:
    return isinstance(x, int) and x >= 1


class Configuration(object):
    __slots__ = ('model', 'lr', 'length',
                 'batch', 'steps', 'decay')

    def __init__(
        self,
        model: ModelDef,
        lr: float,
        length: int,
        batch: int,
        steps: int,
        decay: float,
    ):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'length', length)
        object.__setattr__(self, 'batch', batch)
        object.__setattr__(self, 'steps', steps)
        assert decay is not None
        object.__setattr__(self, 'decay', decay)

    @property
    def precision(self) -> Precision:
        return self.model.precision

    def __setattr__(self, _key, _value):
        raise AttributeError('Configuration is immutable')

    @staticmethod
    def from_dict(d: dict) -> Configuration:
        model = build_model_def(d['spec'], precision=Precision(d['precision']))
        return Configuration(
            model=model,
            lr=d['lr'],
            length=d['length'],
            batch=d['batch'],
            steps=d['steps'],
            decay=d['decay'],
        )

    def __eq__(self, other: Configuration) -> bool:
        return (
            self.lr == other.lr
            and self.precision == other.precision
            and self.length == other.length
            and self.batch == other.batch
            and self.steps == other.steps
            and self.decay == other.decay
            and self.model == other.model
        )

    def __hash__(self) -> int:
        return hash(
            (str(self.model), self.lr, self.length, self.batch,
             self.steps, self.precision, self.decay)
        )

    def replace(
        self,
        model: Optional[ModelDef] = None,
        lr: Optional[float] = None,
        length: Optional[int] = None,
        batch: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
    ) -> Configuration:
        model = model or self.model
        lr = lr or self.lr
        length = length or self.length
        batch = batch or self.batch
        steps = steps or self.steps
        decay = decay or self.decay
        return Configuration(
            model=model, lr=lr, length=length, batch=batch, steps=steps,
            decay=decay,
        )

    def to_dict(self) -> dict:
        return {
            'spec': str(self.model),
            'precision': str(self.precision),
            'lr': self.lr,
            'length': self.length,
            'batch': self.batch,
            'steps': self.steps,
            'decay': self.decay,
        }

    def __repr__(self) -> str:
        return f"Configuration('{self.model}', '{self.precision}', {self.lr}, {self.length}, {self.batch}, {self.steps}, {self.decay})"

    @property
    def num_weights(self) -> int:
        return self.model.num_weights

    def __str__(self) -> str:
        model = str(self.model)
        steps = str(self.steps)

        return (
            f'{model} ({itoa3(self.num_weights)})  {self.precision}   '
            + f'{self.batch}×{self.length}  S{steps}  '
            + self.learning_str
        )

    def aligned_str(self) -> str:
        model = str(self.model)
        steps = f'{self.steps:>5d}'
        return (
            f'{self.batch:>5d}×{self.length:<5d}  {self.learning_str:<10} S{steps}  '
            + f'{self.precision}  {model} ({self.num_weights})'
        )

    def is_valid(self) -> bool:
        return (
            self.model.is_valid()
            and self.lr > 0
            and is_valid_int(self.length)
            and is_valid_int(self.batch)
            and is_valid_int(self.steps)
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

        return f'{lr}{final}'


    @property
    def tokenizer(self) -> Tokenizer:
        return self.model.input.tokenizer

    @property
    def tokens_name(self) -> str:
        return self.model.input.tokens_name



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


IntLimits = Optional[int | tuple[int, int]]
FloatLimits = Optional[float | tuple[float, float]]


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
        lr: FloatLimits,
        length: IntLimits,
        batch: IntLimits,
        steps: IntLimits,
        max_weights: IntLimits,
        precision: list[Precision],
        decay: FloatLimits
    ):
        if not spec:
            self.regex = None
            self.spec = None
        else:
            try:
                build_model_def(spec, precision=Precision.FP32)
                self.regex = None
                self.spec = spec
            except Exception:
                self.regex = re.compile(spec)
                self.spec = None

        self.max_weights = Bounds(max_weights, 16)
        self.length = Bounds(length, 1)
        self.batch = Bounds(batch, 1)
        self.precision = precision
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

        return Template(
            spec=args.spec,
            lr=_parse_interval(args.lr, float),
            length=_parse_interval(args.length, int),
            batch=_parse_interval(args.batch, int),
            steps=_parse_interval(args.steps, int),
            max_weights=_parse_interval(args.weights, int),
            precision=precision,
            decay=_parse_interval(args.decay, float)
        )

    @staticmethod
    def from_form(params: dict):
        """Create a Template from web form parameters."""
        precision = []
        for p in Precision:
            if params.get(str(p)):
                precision.append(p)

        return Template(
            spec=params["spec"],
            lr=_parse_interval(params['lr'], float),
            length=_parse_interval(params['length'], int),
            batch=_parse_interval(params['batch'], int),
            steps=_parse_interval(params['steps'], int),
            max_weights=_parse_interval(params['weights'], int),
            precision=precision,
            decay=_parse_interval(params['decay'], float),
        )

    def __str__(self):
        precision = ','.join(map(str, self.precision))
        spec = self.spec if self.spec is not None else self.regex
        return (
            f'Template(spec={repr(spec)}, '
            + f'precision=\'{precision}\', '
            + f'lr={self.lr}, length={self.length}, '
            + f'batch={self.batch}, '
            + f'steps={self.steps}, '
            + f'max_weights={self.max_weights}, '
            + f'decay={self.decay})'
        )

    def match_model(self, model: ModelDef) -> bool:
        if model.num_weights > self.max_weights.max:
            return False
        if self.spec is not None:
            return self.spec == str(model)
        if self.regex is not None:
            return bool(self.regex.fullmatch(str(model)))
        return True

    def match(self, conf: Configuration) -> bool:
        return (
            self.match_model(conf.model)
            and conf.precision in self.precision
            and self.lr.match(conf.lr)
            and self.length.match(conf.length)
            and self.batch.match(conf.batch)
            and self.steps.match(conf.steps)
            and self.decay.match(conf.decay)
        )

    def _conf_neighbors(
        self, conf: Configuration
    ) -> Iterable[Configuration]:
        # Model neighbors (includes precision + architecture changes)
        for model in conf.model.neighbors():
            if model.precision in self.precision and self.match_model(model):
                yield conf.replace(model=model)
        # Training hyperparameter neighbors
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


def conf_neighbors(
    conf: Configuration, template: Template
) -> list[Configuration]:
    with latency.timer('conf_neighbors'):
        neighbors = template._conf_neighbors_cache.get(conf)

        if neighbors is not None:
            return neighbors

        neighbors = list(template._conf_neighbors(conf))
        template._conf_neighbors_cache[conf] = neighbors

        return neighbors


def default_from_template(
    template: Template, spec: Optional[str]
) -> Configuration:
    precision = template.precision[0]
    lr = template.lr.pick_default(1 / 128)
    length = template.length.pick_default(8)
    batch = template.batch.pick_default(1)
    steps = template.steps.pick_default(2)
    decay = template.decay.pick_default(1.0)

    if spec is None:
        if template.spec is not None:
            spec = template.spec
        else:
            found = False
            for tokens in ('bits.1', 'bits.2', 'bits.4', 'bits.8'):
                for oh in ('', '.oh'):
                    for pos in ('', '+bp'):
                        if found: break
                        input_spec = tokens + oh + pos
                        for layer in (
                            '',
                            'dense.1.tanh',
                            'rnn.1.tanh',
                            'gru.1',
                            'lstm.1',
                            'mingru.1',
                            'mgru.1',
                        ):
                            if found: break
                            full_spec = input_spec + '|' + layer
                            model = build_model_def(full_spec, precision=precision)
                            if not model.is_valid():
                                continue
                            if template.match_model(model):
                                spec = str(model)
                                found = True

    if spec is not None:
        return Configuration(
                        model=build_model_def(spec, precision=precision),
                        lr=lr,
                        length=length,
                        batch=batch,
                        steps=steps,
                        decay=decay,
                    )


    raise RuntimeError(
        "Can't pick up a default model that would fit the template"
    )
