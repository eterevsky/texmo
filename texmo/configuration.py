import argparse
import copy
import enum
import json
import math
import random
import re
from typing import Iterable, Optional

from . import latency
from .common import INF, itoa3
from .model2 import Model2Def
from .pjson import pjson
from .spec_parser import parse_model2
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
        model: Model2Def,
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
        # DB rows and API payloads carry canonical split-form specs
        # (the result DB was migrated off legacy skip.* in 2026-07).
        model = parse_model2(d['spec'], Precision(d['precision']))
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
            # Compare by canonical spec string, matching __hash__.
            and str(self.model) == str(other.model)
        )

    def __hash__(self) -> int:
        return hash(
            (str(self.model), self.lr, self.length, self.batch,
             self.steps, self.precision, self.decay, self.cosine)
        )

    def replace(
        self,
        model: Optional[Model2Def] = None,
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
        lr = format_lr(self.lr)
        if self.cosine:
            return f'{lr}↘0'
        if self.decay == 1:
            final = ''
        else:
            final = '→' + format_lr(self.lr * self.decay)

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


def format_lr(lr: float) -> str:
    """Render a learning rate as either an integer or a `1/X` fraction.

    Matches the format the train CLI accepts via `parse_lr`, so a
    value formatted here can be pasted back as `--lr=1/128`.
    """
    return str(int(lr)) if lr >= 1 else f'1/{int(1 / lr)}'


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


def _parse_spec_filter(
    spec: Optional[str],
) -> tuple[Optional[str], Optional[re.Pattern]]:
    """Split a spec filter into its `(spec, regex)` pair.

    A filter that parses as a model is kept as the canonical spec
    string, so the literal match in `match_model` lines up with
    `str(model)` for confs built via `parse_model2`. Anything else is
    compiled as a regex.
    """
    if not spec:
        return None, None
    try:
        model = parse_model2(spec, Precision.FP32)
        return str(model), None
    except Exception:
        return None, re.compile(spec)


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
        self.spec, self.regex = _parse_spec_filter(spec)

        self.max_weights = Bounds(max_weights, 16)
        # num_layers bounds the count of intermediate layers in the
        # pipeline (everything between input and output dense; counts
        # suffix/norm/split too, matching what's visible in the spec).
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
    def from_form(params: dict, spec: Optional[str] = None):
        """Create a Template from web form parameters.

        `spec` is passed in rather than read from the form: the index
        page has no spec field: model specs live in the sub-search rows,
        one per entry, and the base template they share is unrestricted.
        The compare page does have one spec box per column and passes it
        here explicitly.
        """
        precision = []
        for p in Precision:
            if params.get(str(p)):
                precision.append(p)

        decay_types = [
            t for t in DecayType if params.get(f'decay_{t}')
        ]

        return Template(
            spec=spec,
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

    @property
    def spec_pattern(self) -> Optional[str]:
        """The spec filter as a single string: the canonical literal
        spec, the regex source, or None when unrestricted."""
        if self.spec is not None:
            return self.spec
        if self.regex is not None:
            return self.regex.pattern
        return None

    def with_spec(self, spec: Optional[str]) -> Template:
        """This template with its spec filter replaced by `spec`.

        Every other bound is shared with the original: a shallow copy,
        whose sub-objects are never mutated (same contract as
        `_warmup_template` / `_layer_capped_template` in `search.py`).
        Returns `self` when the filter is unchanged, so a single-entry
        `TemplateSet` keeps using the base template object itself.
        """
        if spec == self.spec_pattern:
            return self
        out = copy.copy(self)
        out.spec, out.regex = _parse_spec_filter(spec)
        return out

    def matches_spec(self, model_spec: str) -> bool:
        """True when `model_spec` passes the spec filter alone. The
        bounds (weights, layers, ...) are deliberately not consulted."""
        if self.spec is not None:
            return self.spec == model_spec
        if self.regex is not None:
            return bool(self.regex.fullmatch(model_spec))
        return True

    def match_model(self, model: Model2Def) -> bool:
        if model.num_weights > self.max_weights.max:
            return False
        if not self.num_layers.match(model.num_layers):
            return False
        return self.matches_spec(str(model))

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
            candidates = []
            # Same restricted input set as the search neighbors. The
            # one-hot candidates come first so templates that match
            # both codecs keep their historical default.
            for input_spec in (
                'bits.1+bp', 'bits.2.oh+bp', 'bits.4.oh+bp', 'bytes',
                'tokens.32.hexbpe.emb.1', 'tokens.32.fold.emb.1',
            ):
                for layer in (
                    '',
                    'dense.1.tanh',
                    'rnn.1.tanh',
                    'gru.1',
                    'lstm.1',
                    'mingru.1',
                    'mgru.1',
                ):
                    candidates.append(f'{input_spec}|{layer}')
            # Tied-codec candidates: the table width X follows the
            # chain's final width, so the chains are sized to X.
            for domain in ('bits.1', 'bits.2', 'bits.4', 'bytes'):
                for x in (1, 2, 4, 8, 16):
                    for layer in (
                        '',
                        f'dense.{x}.tanh',
                        f'rnn.{x}.tanh',
                        f'gru.{x}',
                        f'lstm.{x}',
                        f'mingru.{x}',
                        f'mgru.{x}',
                    ):
                        candidates.append(f'{domain}.emb.{x}|{layer}')
            for full_spec in candidates:
                model = parse_model2(full_spec, precision)
                if not model.is_valid():
                    continue
                if template.match_model(model):
                    spec = str(model)
                    break

    if spec is not None:
        model = parse_model2(spec, precision)
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


# --- weighted template set ---------------------------------------------------
#
# Several sub-searches running concurrently over one server, each with
# a slice of the select budget. They share EVERY template bound (lr,
# length, batch, steps, weights, precision, decay types, num_layers)
# and differ only in their spec filter -- so an entry is "the same
# search, restricted to this corner of model space". The point is to
# let a freshly implemented layer be measured on its own budget
# instead of competing from cold against a mature general population.

# Name of the lone entry a server runs when no sub-searches are
# configured.
MAIN_ENTRY = 'main'

_ENTRY_KEYS = frozenset({'name', 'share', 'regex', 'default_spec'})

# The same four fields as the web form's per-row inputs, in column
# order (see `TemplateSet.from_rows` / `.rows`).
_ROW_KEYS = ('name', 'regex', 'default_spec', 'share')


class TemplateEntry(object):
    """One sub-search: a name, a budget share, and a spec filter.

    `template` is the base template with this entry's spec substituted;
    `default` is the conf that bootstraps the sub-space -- the smallest
    conf agreeing with `template`, either given explicitly as
    `default_spec` or derived by `default_from_template`. Without it a
    brand-new sub-space has nothing to hand out and its budget would
    quietly drain back into the general search.
    """

    __slots__ = ('name', 'share', 'spec', 'default_spec', 'template',
                 'default', 'min_layers')

    def __init__(
        self,
        base: Template,
        name: str,
        share: float,
        spec: Optional[str] = None,
        default_spec: Optional[str] = None,
        default: Optional[Configuration] = None,
    ):
        name = (name or '').strip()
        if not name:
            raise ValueError('template entry needs a non-empty name')
        if share < 0:
            raise ValueError(
                f'entry {name!r}: share must be >= 0, got {share}')
        self.name = name
        self.share = float(share)
        self.spec = spec or None
        self.default_spec = default_spec or None

        try:
            self.template = base.with_spec(self.spec)
        except re.error as exc:
            raise ValueError(
                f'entry {name!r}: bad spec regex {self.spec!r}: {exc}'
            ) from exc

        if default is None:
            try:
                default = default_from_template(
                    self.template, spec=self.default_spec)
            except Exception as exc:
                raise ValueError(f'entry {name!r}: {exc}') from exc
            # `default_from_template` applies the spec filter only when
            # it has to auto-pick; an explicit `default_spec` goes
            # through unchecked, and one that its own entry rejects
            # could never be selected. Catch it at configuration time.
            # (A `default` handed in by the caller is their business --
            # that is the pre-template-set path, where a mismatched
            # --default-spec is simply skipped at select time.)
            if not self.template.matches_spec(str(default.model)):
                raise ValueError(
                    f'entry {name!r}: default spec {str(default.model)!r} '
                    f'does not match the entry spec filter {self.spec!r}')
        self.default = default

        # The smallest layer count this sub-space can produce, taken
        # from its default (the smallest conf that agrees with it).
        # Cached: `search._layer_cap_probs` needs it on every select.
        self.min_layers = self.default.model.num_layers

    def to_dict(self) -> dict:
        out: dict = {'name': self.name}
        share = self.share
        out['share'] = int(share) if share == int(share) else share
        if self.spec:
            out['regex'] = self.spec
        if self.default_spec:
            out['default_spec'] = self.default_spec
        return out

    def __str__(self) -> str:
        spec = self.spec if self.spec else '(unrestricted)'
        return (
            f'TemplateEntry({self.name!r}, share={self.share:g}, '
            f'spec={spec}, default={self.default.model})'
        )


class TemplateSet(object):
    """An ordered list of `TemplateEntry`, drawn per select in
    proportion to their shares (which are normalized, so they need not
    sum to 100)."""

    def __init__(self, entries: Iterable[TemplateEntry]):
        entries = list(entries)
        if not entries:
            raise ValueError('template set is empty')
        names = [e.name for e in entries]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f'duplicate template names: {", ".join(duplicates)}')
        total = sum(e.share for e in entries)
        if total <= 0:
            raise ValueError('template shares must add up to more than 0')
        self.entries = entries
        self.total_share = total

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    @property
    def is_single(self) -> bool:
        return len(self.entries) == 1

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def by_name(self, name: str) -> Optional[TemplateEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def nominal_share(self, entry: TemplateEntry) -> float:
        """`entry`'s share as a percentage of the whole set."""
        return 100.0 * entry.share / self.total_share

    def draw(self) -> TemplateEntry:
        """Draw one entry weighted by share.

        Short-circuits for a single entry so the no-template-set case
        doesn't consume randomness -- the selection path (and its RNG
        stream) is then exactly what it was before template sets
        existed.
        """
        if len(self.entries) == 1:
            return self.entries[0]
        return random.choices(
            self.entries, weights=[e.share for e in self.entries], k=1
        )[0]

    @staticmethod
    def single(
        base: Template,
        spec: Optional[str] = None,
        default_spec: Optional[str] = None,
        default: Optional[Configuration] = None,
    ) -> TemplateSet:
        """A one-entry set: `main`, covering `spec` (the whole template
        when that is None). This is what a server with no sub-searches
        configured runs -- one unrestricted search, same as before
        template sets existed, but represented as an ordinary entry."""
        return TemplateSet([TemplateEntry(
            base, name=MAIN_ENTRY, share=100.0, spec=spec,
            default_spec=default_spec, default=default,
        )])

    @staticmethod
    def from_json(text: str, base: Template) -> TemplateSet:
        """Parse a template set from its JSON list form:

            [{"name": "main", "share": 60},
             {"name": "attn", "share": 40, "regex": ".*attn.*",
              "default_spec": "bytes|split.add(attn.4.1)"}]

        Raises ValueError on anything malformed -- callers surface that
        to the user and keep running on the previous set.
        """
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f'invalid JSON: {exc}') from exc
        if not isinstance(data, list):
            raise ValueError(
                f'template set must be a JSON list of entries, got '
                f'{type(data).__name__}')
        entries = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f'entry {i}: expected an object, got '
                    f'{type(item).__name__}')
            unknown = sorted(set(item) - _ENTRY_KEYS)
            if unknown:
                raise ValueError(
                    f'entry {i}: unknown key(s) {", ".join(unknown)}; '
                    f'known keys are {", ".join(sorted(_ENTRY_KEYS))}')
            name = item.get('name')
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f'entry {i}: "name" must be a non-empty string')
            share = item.get('share', 0)
            if isinstance(share, bool) or not isinstance(share, (int, float)):
                raise ValueError(
                    f'entry {name!r}: "share" must be a number, got '
                    f'{share!r}')
            for key in ('regex', 'default_spec'):
                value = item.get(key)
                if value is not None and not isinstance(value, str):
                    raise ValueError(
                        f'entry {name!r}: "{key}" must be a string, got '
                        f'{value!r}')
            entries.append(TemplateEntry(
                base,
                name=name,
                share=share,
                spec=item.get('regex'),
                default_spec=item.get('default_spec'),
            ))
        return TemplateSet(entries)

    def to_json(self) -> str:
        """The JSON form `from_json` accepts, for `--templates`."""
        return pjson([e.to_dict() for e in self.entries])

    @staticmethod
    def from_rows(
        rows: Iterable[dict],
        base: Template,
        default_spec: Optional[str] = None,
        default: Optional[Configuration] = None,
    ) -> TemplateSet:
        """Build a set from the web form's parallel row fields.

        Each row is a dict of raw strings keyed by `_ROW_KEYS`. Rows
        whose every field is blank are dropped, so a stray row left by
        the form's Add button is harmless rather than an error; with
        nothing left this is one unrestricted `main` entry (built from
        `default_spec` / `default`, same as `single()`).

        Raises ValueError on anything malformed -- callers surface that
        to the user and keep running on the previous set.
        """
        filled = [
            (i, row) for i, row in enumerate(rows)
            if any((row.get(k) or '').strip() for k in _ROW_KEYS)
        ]
        if not filled:
            return TemplateSet.single(
                base, default_spec=default_spec, default=default)

        entries = []
        for i, row in filled:
            name = (row.get('name') or '').strip()
            if not name:
                raise ValueError(f'sub-search row {i + 1}: name is required')
            share_text = (row.get('share') or '').strip()
            if not share_text:
                raise ValueError(f'entry {name!r}: share is required')
            try:
                share = float(share_text)
            except ValueError:
                raise ValueError(
                    f'entry {name!r}: share {share_text!r} is not a number'
                ) from None
            entries.append(TemplateEntry(
                base,
                name=name,
                share=share,
                spec=(row.get('regex') or '').strip(),
                default_spec=(row.get('default_spec') or '').strip(),
            ))
        return TemplateSet(entries)

    def rows(self) -> list[dict]:
        """Row dicts for the web form -- the inverse of `from_rows`.

        Always at least one row, since a set always has at least one
        entry: a server with no sub-searches configured shows a single
        `main` row with a blank regex, which is the unrestricted search
        it was already running, now visible and editable.
        """
        return [{
            'name': e.name,
            'regex': e.spec or '',
            'default_spec': e.default_spec or '',
            'share': f'{e.share:g}',
        } for e in self.entries]

    def __str__(self) -> str:
        return 'TemplateSet(' + ', '.join(
            f'{e.name}:{self.nominal_share(e):.0f}%' for e in self.entries
        ) + ')'
