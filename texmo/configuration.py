import argparse
import enum
import math
import re
from typing import Iterable, Optional

from . import latency
from .common import INF, itoa3
from .model import ModelDef, build_model_def
from .precision import Precision
from .tokens.tokenizer import Tokenizer


class DecayType(enum.StrEnum):
    """The kind of LR schedule a Configuration uses, derived from
    (decay, cosine) — see `Configuration.decay_type`."""
    NONE = 'none'
    EXP = 'exp'
    COSINE = 'cosine'


def is_valid_int(x: int) -> bool:
    return isinstance(x, int) and x >= 1


class Configuration(object):
    __slots__ = ('model', 'lr', 'length',
                 'batch', 'steps', 'decay', 'cosine')

    def __init__(
        self,
        model: ModelDef,
        lr: float,
        length: int,
        batch: int,
        steps: int,
        decay: float,
        cosine: bool = False,
    ):
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'length', length)
        object.__setattr__(self, 'batch', batch)
        object.__setattr__(self, 'steps', steps)
        assert decay is not None
        object.__setattr__(self, 'decay', decay)
        object.__setattr__(self, 'cosine', bool(cosine))

    @property
    def precision(self) -> Precision:
        return self.model.precision

    def __setattr__(self, _key, _value):
        raise AttributeError('Configuration is immutable')

    @staticmethod
    def from_dict(d) -> Configuration:
        model = build_model_def(d['spec'], precision=Precision(d['precision']))
        # `d` may be a plain dict (from JSON) or a sqlite3.Row. Row
        # raises IndexError on a missing column; dict raises KeyError.
        try:
            cosine = bool(d['cosine'])
        except (KeyError, IndexError):
            cosine = False
        return Configuration(
            model=model,
            lr=d['lr'],
            length=d['length'],
            batch=d['batch'],
            steps=d['steps'],
            decay=d['decay'],
            cosine=cosine,
        )

    def __eq__(self, other: Configuration) -> bool:
        return (
            self.lr == other.lr
            and self.precision == other.precision
            and self.length == other.length
            and self.batch == other.batch
            and self.steps == other.steps
            and self.decay == other.decay
            and self.cosine == other.cosine
            and self.model == other.model
        )

    def __hash__(self) -> int:
        return hash(
            (str(self.model), self.lr, self.length, self.batch,
             self.steps, self.precision, self.decay, self.cosine)
        )

    def replace(
        self,
        model: Optional[ModelDef] = None,
        lr: Optional[float] = None,
        length: Optional[int] = None,
        batch: Optional[int] = None,
        steps: Optional[int] = None,
        decay: Optional[float] = None,
        cosine: Optional[bool] = None,
    ) -> Configuration:
        model = model or self.model
        lr = lr or self.lr
        length = length or self.length
        batch = batch or self.batch
        steps = steps or self.steps
        decay = decay or self.decay
        cosine = self.cosine if cosine is None else cosine
        return Configuration(
            model=model, lr=lr, length=length, batch=batch, steps=steps,
            decay=decay, cosine=cosine,
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
            'cosine': self.cosine,
        }

    def __repr__(self) -> str:
        return (
            f"Configuration('{self.model}', '{self.precision}', "
            f"{self.lr}, {self.length}, {self.batch}, {self.steps}, "
            f"{self.decay}, cosine={self.cosine})"
        )

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
        # 6 digits fit step counts up to 999999 — search has produced
        # configs at 131072 steps, so 5 was too narrow.
        steps = f'{self.steps:>6d}'
        return (
            f'{self.batch:>5d}×{self.length:<5d}  {self.learning_str:<10} S{steps}  '
            + f'{self.precision}  {model} ({self.num_weights})'
        )

    def is_valid(self) -> bool:
        if self.cosine and self.decay != 1:
            # Cosine schedule already decays LR to 0 over `steps`; combining
            # it with exponential decay would multiply two decay curves.
            return False
        return (
            self.model.is_valid()
            and self.lr > 0
            and is_valid_int(self.length)
            and is_valid_int(self.batch)
            and is_valid_int(self.steps)
            and 0 < self.decay <= 1
        )

    @property
    def decay_type(self) -> 'DecayType':
        if self.cosine:
            return DecayType.COSINE
        if self.decay == 1.0:
            return DecayType.NONE
        return DecayType.EXP

    @property
    def learning_str(self) -> str:
        render_lr = lambda lr: str(int(lr)) if lr >= 1 else f'1/{str(int(1/lr))}'
        lr = render_lr(self.lr)
        if self.cosine:
            return f'{lr}↘0'
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
        self,
        limits: Optional[float | tuple[float, float]],
        min_value: float,
        max_value: float = INF,
    ):
        if limits is None:
            self.min = min_value
            self.max = max_value
        else:
            try:
                self.min, self.max = limits
            except TypeError:
                self.min = self.max = limits
            # Enforce system-level upper bound (e.g. decay ≤ 1).
            self.max = min(self.max, max_value)

    def __str__(self) -> str:
        return f'({self.min}, {self.max})'

    def match(self, value: float) -> bool:
        return self.min <= value <= self.max

    def pick_default(self, default: float) -> float:
        if self.min <= default <= self.max:
            return default
        if self.min == 0:
            return self.max
        if self.max == INF:
            # No finite upper bound -- anchor to min (symmetric to
            # the min == 0 case above). Geometric mean would be inf.
            return self.min
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


# Used internally by Template._conf_neighbors for the exp-decay walk.
# Independent of the template — a conf at decay=0.5 has decay neighbors
# (1.0, 0.25), and the matching pass against `decay_types` decides
# whether each survives. Lower bound is 0 so the walk can halve
# indefinitely; in practice the search doesn't chase decay to zero.
_DECAY_NEIGHBOR_BOUNDS = Bounds(None, 0, max_value=1)


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
        decay_types: Optional[Iterable[DecayType | str]] = None,
        num_layers: Optional[IntLimits] = None,
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
        # num_layers bounds the count of intermediate layers in the
        # pipeline (everything between input and output dense; counts
        # suffix/norm/skip too, matching what's visible in the spec).
        # min default 0 -- specs like 'bits.1+bp|' (no hidden layers)
        # are valid.
        self.num_layers = Bounds(num_layers, 0)
        self.length = Bounds(length, 1)
        self.batch = Bounds(batch, 1)
        self.precision = precision
        self.lr = Bounds(lr, 0)
        self.steps = Bounds(steps, 2)
        if decay_types is None:
            decay_types = list(DecayType)
        # Coerce strings (CLI / form) into enum members; explicit list
        # of DecayType passes through.
        self.decay_types = [DecayType(t) for t in decay_types]

    @staticmethod
    def from_args(args: argparse.Namespace):
        if not args.precision:
            # FP64 is supported but not in the default set — it's mostly
            # useful as a baseline and isn't supported on MPS. Opt in
            # explicitly via --precision.
            precision = [Precision.FP32, Precision.FP16, Precision.BF16]
        else:
            precision = list(map(Precision, args.precision.split(',')))

        decay_types = [
            t.strip() for t in args.decay_types.split(',') if t.strip()
        ]

        return Template(
            spec=args.spec,
            lr=_parse_interval(args.lr, float),
            length=_parse_interval(args.length, int),
            batch=_parse_interval(args.batch, int),
            steps=_parse_interval(args.steps, int),
            max_weights=_parse_interval(args.weights, int),
            precision=precision,
            decay_types=decay_types,
            num_layers=_parse_interval(args.num_layers, int),
        )

    @staticmethod
    def from_form(params: dict):
        """Create a Template from web form parameters."""
        precision = []
        for p in Precision:
            if params.get(str(p)):
                precision.append(p)

        decay_types = [
            t for t in DecayType if params.get(f'decay_{t}')
        ]

        return Template(
            spec=params["spec"],
            lr=_parse_interval(params['lr'], float),
            length=_parse_interval(params['length'], int),
            batch=_parse_interval(params['batch'], int),
            steps=_parse_interval(params['steps'], int),
            max_weights=_parse_interval(params['weights'], int),
            precision=precision,
            decay_types=decay_types,
            num_layers=_parse_interval(params.get('num_layers'), int),
        )

    def __str__(self):
        precision = ','.join(map(str, self.precision))
        decay_types = ','.join(self.decay_types)
        spec = self.spec if self.spec is not None else self.regex
        return (
            f'Template(spec={repr(spec)}, '
            + f'precision=\'{precision}\', '
            + f'lr={self.lr}, length={self.length}, '
            + f'batch={self.batch}, '
            + f'steps={self.steps}, '
            + f'max_weights={self.max_weights}, '
            + f'num_layers={self.num_layers}, '
            + f'decay_types=\'{decay_types}\')'
        )

    def match_model(self, model: ModelDef) -> bool:
        if model.num_weights > self.max_weights.max:
            return False
        if not self.num_layers.match(len(model.layers)):
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
            and conf.decay_type in self.decay_types
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
        if not conf.cosine:
            # Exp-decay neighbor walk. The destination is filtered by
            # decay_types: decay=1 lands as 'none', decay<1 as 'exp'.
            for decay in _DECAY_NEIGHBOR_BOUNDS.neighbors(conf.decay):
                cand = conf.replace(decay=decay)
                if cand.decay_type in self.decay_types:
                    yield cand
        # Cosine schedule toggle. Asymmetric: from any exp-decay conf
        # you reach (cosine=True, decay=1) directly; going back from
        # cosine only gets you (cosine=False, decay=1) — exp variants
        # are then reachable via the regular decay walk.
        if conf.cosine:
            cand = conf.replace(cosine=False)  # type=none
            if cand.decay_type in self.decay_types:
                yield cand
        else:
            cand = conf.replace(cosine=True, decay=1.0)  # type=cosine
            if cand.decay_type in self.decay_types:
                yield cand


def conf_neighbors(
    conf: Configuration, template: Template
) -> list[Configuration]:
    with latency.timer('conf_neighbors'):
        return list(template._conf_neighbors(conf))


def default_from_template(
    template: Template, spec: Optional[str]
) -> Configuration:
    precision = template.precision[0]
    lr = template.lr.pick_default(1 / 128)
    length = template.length.pick_default(8)
    batch = template.batch.pick_default(1)
    steps = template.steps.pick_default(2)
    # Pick a decay/cosine pair compatible with the allowed decay_types.
    if 'none' in template.decay_types:
        decay, cosine = 1.0, False
    elif 'exp' in template.decay_types:
        decay, cosine = 0.5, False
    elif 'cosine' in template.decay_types:
        decay, cosine = 1.0, True
    else:
        raise RuntimeError("Template has empty decay_types")

    if spec is None:
        if template.spec is not None:
            spec = template.spec
        else:
            found = False
            # Same restricted input set as the search neighbors.
            for input_spec in (
                'bits.1+bp', 'bits.2.oh+bp', 'bits.4.oh+bp', 'bytes',
            ):
                if found: break
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
        model = build_model_def(spec, precision=precision)
        if not model.is_valid():
            raise ValueError(
                f"Default spec {spec!r} is not a valid model "
                f"(at precision={precision.value})."
            )
        return Configuration(
            model=model,
            lr=lr,
            length=length,
            batch=batch,
            steps=steps,
            decay=decay,
            cosine=cosine,
        )

    raise RuntimeError(
        "Can't pick up a default model that would fit the template"
    )
