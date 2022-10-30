import re

from collections import namedtuple

from .spec import ModelSpec


INF = float("inf")
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
    100,
    200,
    500,
]


def next_number(x):
    i = LRS.index(x)
    return LRS[i + 1]


def prev_number(x):
    i = LRS.index(x)
    return LRS[i - 1]


def neighbor_numbers(x):
    """Produce neighbor numbers from a close to exponent but readable table."""
    i = LRS.index(x)
    if i > 0:
        yield LRS[i - 1]
    if i < len(LRS) - 1:
        yield LRS[i + 1]


def neighbor_power2(x):
    """Produce neighbor powers of 2."""
    if x > 1:
        return (x // 2, x * 2)
    else:
        return (x * 2,)


Configuration = namedtuple(
    "Configuration",
    [
        "id",
        "spec",
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
CONF_FIELDS = "id, spec, lr, sample_len, batch, regularization, init_scale, t"


def conf_from_record(record):
    spec = ModelSpec.parse(record.model_spec)
    return Configuration(
        None,
        spec,
        record.learning_rate,
        record.train_sample_len,
        record.train_batch,
        record.regularization,
        record.init_scale,
        record.time_round,
    )


def conf_from_row(row):
    """Create Configuration from a database row."""
    if row is None:
        return None
    spec = ModelSpec.parse(row[1])
    return Configuration(row[0], spec, *row[2:])


def is_power2(x):
    return type(x) is int and x >= 1 and x & (x - 1) == 0


def conf_is_valid(conf):
    return (
        conf.spec.is_valid()
        and conf.lr in LRS
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and conf.regularization in LRS
        and conf.init_scale in LRS
        and is_power2(conf.t)
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
    ):
        self.regex = re.compile(spec_regex) if spec_regex is not None else None
        self.lr = make_bounds(lr)
        self.sample_len = make_bounds(sample_len)
        self.batch = make_bounds(batch)
        self.regularization = make_bounds(regularization)
        self.init_scale = make_bounds(init_scale)
        self.t = make_bounds(t)

    def match_conf(self, conf):
        return (
            match_bounds(self.lr, conf.lr)
            and match_bounds(self.sample_len, conf.sample_len)
            and match_bounds(self.batch, conf.batch)
            and match_bounds(self.regularization, conf.regularization)
            and match_bounds(self.init_scale, conf.init_scale)
            and match_bounds(self.t, conf.t)
            and self.match_spec(conf.spec)
        )

    def match_spec(self, spec):
        return self.regex is None or self.regex.fullmatch(str(spec))


_spec_neighbors = {}


def conf_neighbors(conf: Configuration, template: Template):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    conf = conf._replace(id=None)
    if conf.spec in _spec_neighbors:
        for neighbor_spec in _spec_neighbors[conf.spec]:
            yield conf._replace(spec=neighbor_spec)
    else:
        cache = []
        _spec_neighbors[conf.spec] = cache
        for neighbor_spec, _ in conf.spec.all_neighbors():
            if template.match_spec(neighbor_spec):
                cache.append(neighbor_spec)
                yield conf._replace(spec=neighbor_spec)

    for x in neighbor_numbers(conf.lr):
        if match_bounds(template.lr, x):
            yield conf._replace(lr=x)

    for x in neighbor_power2(conf.sample_len):
        if match_bounds(template.sample_len, x):
            yield conf._replace(sample_len=x)

    for x in neighbor_power2(conf.batch):
        if match_bounds(template.batch, x):
            yield conf._replace(batch=x)

    for x in neighbor_numbers(conf.regularization):
        if match_bounds(template.regularization, x):
            yield conf._replace(regularization=x)

    for x in neighbor_numbers(conf.init_scale):
        if match_bounds(template.init_scale, x):
            yield conf._replace(init_scale=x)

    for x in neighbor_power2(conf.t):
        if match_bounds(template.t, x):
            yield conf._replace(t=x)

