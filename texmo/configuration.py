import argparse
from collections import namedtuple
import re

from .common import INF, is_power2, power2_neighbors
from .model2 import build_model


Configuration = namedtuple(
    "Configuration",
    [
        "model",
        "lr",
        "sample_len",
        "batch",
        "regularization",
        "init_scale",
        "t",
    ],
)


# Database fields required to create a Configuration instance using
# conf_from_row
CONF_FIELDS = "spec, lr, sample_len, batch, regularization, init_scale, t"


def conf_from_record(record):
    model = build_model(record.model_spec)
    return Configuration(
        model,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.regularization,
        record.init_scale,
        record.planned_time_s,
    )


def conf_from_row(row) -> Configuration:
    """Create Configuration from a database row."""
    if row is None:
        return None
    model = build_model(row[0])
    return Configuration(model, *row[1:])


def conf_to_dict(conf: Configuration) -> dict:
    return {
        "model": str(conf.model),
        "lr": conf.lr,
        "sample_len": conf.sample_len,
        "batch": conf.batch,
        "regularization": conf.regularization,
        "init_scale": conf.init_scale,
        "t": conf.t,
    }


def conf_from_dict(spec: dict) -> Configuration:
    model = build_model(spec["model"])
    return Configuration(
        model,
        spec["lr"],
        spec["sample_len"],
        spec["batch"],
        spec["regularization"],
        spec["init_scale"],
        spec["t"],
    )


def conf_is_valid(conf: Configuration) -> bool:
    return (
        conf.model is not None
        and conf.model.is_valid()
        and is_power2(conf.lr)
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and is_power2(conf.regularization)
        and is_power2(conf.init_scale)
        and is_power2(conf.t)
    )


def conf_to_string(conf: Configuration) -> str:
    s = f"{conf.model}  LR{conf.lr:.3f}  B{conf.batch}  T{conf.t}"
    s += f"  LEN{conf.sample_len}"
    if conf.regularization != 0.125:
        s += f"  R{conf.regularization:.3f}"
    if conf.init_scale != 1.0:
        s += f"  I{conf.regularization:.3f}"
    return s


_COMPONENT_RE = re.compile("^([A-Z]+)([^A-Z].*)$")


def parse_conf(s: str, defaults: Configuration) -> Configuration:
    components = s.split()
    model = build_model(components[0])
    lr = defaults.lr
    sample_len = defaults.sample_len
    batch = defaults.batch
    regularization = defaults.regularization
    init_scale = defaults.init_scale
    t = defaults.t

    for c in components[1:]:
        name, value = _COMPONENT_RE.match(c).groups()
        if name == "LR":
            lr = float(value)
        elif name == "B":
            batch = int(value)
        elif name == "R":
            regularization = float(value)
        elif name == "I":
            init_scale = float(value)
        elif name == "T":
            t = int(value)

    return Configuration(
        model, lr, sample_len, batch, regularization, init_scale, t
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
        regularization=None,
        init_scale=None,
        t=None,
        max_weights=None,
    ):
        self.regex = re.compile(spec_regex) if spec_regex is not None else None
        self.lr = make_bounds(lr)
        self.sample_len = make_bounds(sample_len)
        self.batch = make_bounds(batch)
        self.regularization = make_bounds(regularization)
        self.init_scale = make_bounds(init_scale)
        self.t = make_bounds(t)
        self.max_weights = max_weights

    @staticmethod
    def from_args(args: argparse.Namespace):
        return Template(
            spec_regex=args.spec_regex,
            batch=parse_interval(args.batch, int),
            lr=parse_interval(args.lr, float),
            sample_len=parse_interval(args.sample_len, int),
            regularization=parse_interval(args.regularization, float),
            init_scale=parse_interval(args.init_scale, float),
            t=parse_interval(args.time, int),
            max_weights=args.max_weights,
        )

    def match_conf(self, conf):
        return (
            match_bounds(self.lr, conf.lr)
            and match_bounds(self.sample_len, conf.sample_len)
            and match_bounds(self.batch, conf.batch)
            and match_bounds(self.regularization, conf.regularization)
            and match_bounds(self.init_scale, conf.init_scale)
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
            regularization=self.regularization,
            init_scale=self.init_scale,
            t=self.t,
            max_weights=self.max_weights,
        )


def _pick_default_value(range, default):
    if range is None or range[0] <= default <= range[1]:
        return default
    return range[0]


def default_from_template(template: Template) -> Configuration:
    lr = _pick_default_value(template.lr, 0.125)
    sample_len = _pick_default_value(template.sample_len, 128)
    batch = _pick_default_value(template.batch, 64)
    regularization = _pick_default_value(template.regularization, 0.125)
    init_scale = _pick_default_value(template.init_scale, 1.0)
    t = _pick_default_value(template.t, 1)
    return Configuration(
        None, lr, sample_len, batch, regularization, init_scale, t
    )


def add_template_args(parser: argparse.ArgumentParser):
    """Add command-line arguments describing a template."""
    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex convering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default="0.001-10",
        help="range of acceptable learning rates (default: 0.001-10)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="128",
        help="range of acceptable sample lens (default: 128)",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=str,
        default="0.125",
        help="range of values for regularization coefficient (default: 0.125)",
    )
    parser.add_argument(
        "-i",
        "--init-scale",
        type=str,
        default="1.0",
        help="range of values of the coefficient for the initial weights (default: 1.0)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined (default: unrestricted)",
    )


def add_default_conf_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--spec-default",
        type=str,
        default="dense.1.relu",
        help="initial spec (default: dense.1.relu)",
    )
    parser.add_argument(
        "--batch-default",
        type=int,
        default=64,
        help="default batch size. Should agree with limits from -b and be a power of 2. (default: 64)",
    )
    parser.add_argument(
        "--lr-default",
        type=float,
        default=0.125,
        help="default learning rate. (default: 0.125)",
    )
    parser.add_argument(
        "--sample-len-default",
        type=int,
        default=None,
        help="default sample length (default: taken from --sample-len)",
    )
    parser.add_argument(
        "--regularization-default",
        type=float,
        default=None,
        help="default value for regularization coefficient (default: taken from -r)",
    )
    parser.add_argument(
        "--init-scale-default",
        type=float,
        default=None,
        help="default value for init scaling coefficient (default: taken from --init-scale)",
    )


_model_neighbors = {}


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

    for x in (conf.regularization / 2, conf.regularization * 2):
        if match_bounds(template.regularization, x):
            yield conf._replace(regularization=x)

    for x in (conf.init_scale / 2, conf.init_scale * 2):
        if match_bounds(template.init_scale, x):
            yield conf._replace(init_scale=x)

    for x in power2_neighbors(conf.t):
        if match_bounds(template.t, x):
            yield conf._replace(t=x)
