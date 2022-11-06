import re

from collections import namedtuple

from .common import INF, is_power2
from .spec import ModelSpec


def neighbor_power2(x):
    """Produce neighbor integer powers of 2."""
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


def conf_is_valid(conf):
    return (
        conf.spec is not None
        and conf.spec.is_valid()
        and is_power2(conf.lr)
        and is_power2(conf.sample_len)
        and is_power2(conf.batch)
        and is_power2(conf.regularization)
        and is_power2(conf.init_scale)
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
        return (
            self.max_weights is None or spec.weights() <= self.max_weights
        ) and (self.regex is None or self.regex.fullmatch(str(spec)))


_spec_neighbors = {}


def conf_neighbors(conf: Configuration, template: Template):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    conf = conf._replace(id=None)
    cache = _spec_neighbors.get(conf.spec)
    if cache is None:
        cache = []
        _spec_neighbors[conf.spec] = cache
        for neighbor_spec, _ in conf.spec.all_neighbors():
            if template.match_spec(neighbor_spec):
                cache.append(neighbor_spec)

    for neighbor_spec in cache:
        yield conf._replace(spec=neighbor_spec)

        # For neighbor specs that add or remove layers we'll also add
        # configurations with increased/decreased time limit.

        if len(neighbor_spec._layers) > len(conf.spec._layers) and match_bounds(
            template.t, conf.t * 2
        ):
            yield conf._replace(spec=neighbor_spec, t=conf.t * 2)
            if match_bounds(template.lr, conf.lr / 2):
                yield conf._replace(
                    spec=neighbor_spec, t=conf.t * 2, lr=conf.lr / 2
                )
        elif len(neighbor_spec._layers) < len(
            conf.spec._layers
        ) and match_bounds(template.t, conf.t // 2):
            yield conf._replace(spec=neighbor_spec, t=conf.t // 2)
            if match_bounds(template.lr, conf.lr * 2):
                yield conf._replace(
                    spec=neighbor_spec, t=conf.t // 2, lr=conf.lr * 2
                )

    for x in (conf.lr / 2, conf.lr * 2):
        if match_bounds(template.lr, x):
            yield conf._replace(lr=x)

    for x in neighbor_power2(conf.sample_len):
        if match_bounds(template.sample_len, x):
            yield conf._replace(sample_len=x)

    for x in neighbor_power2(conf.batch):
        if match_bounds(template.batch, x):
            yield conf._replace(batch=x)

    for x in (conf.regularization / 2, conf.regularization * 2):
        if match_bounds(template.regularization, x):
            yield conf._replace(regularization=x)

    for x in (conf.init_scale / 2, conf.init_scale * 2):
        if match_bounds(template.init_scale, x):
            yield conf._replace(init_scale=x)

    for x in neighbor_power2(conf.t):
        if match_bounds(template.t, x):
            yield conf._replace(t=x)
