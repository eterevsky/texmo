from collections import namedtuple
from spec import ModelSpec


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


Results = namedtuple(
    "Results",
    [
        "score",
        "cluster_score",
        "num_runs",
    ],
)


VARY = {
    # Add an extra layer (implies "type")
    "layer": 1,
    # Change the type of a layer (dense, rec, gru, lstm), (implies "size")
    "type": 2,
    # Change the size of a layer (dense, rec, gru, lstm)
    "size": 3,
    # Introduce a suffix.2 layer or change the size of a suffix
    # or attention layer
    "suffix": 4,
    # Change layer type between suffix and attn
    "attn": 5,
    # Change batch size
    "batch": 6,
    # Learning rate
    "lr": 7,
    # Regualarization
    "reg": 8,
    # init_scale
    "init": 9,
    # Sample length
    "len": 10,
    # Training time
    "time": 11,
}


def neighbor_numbers(x):
    """Produce neighbor numbers from a close to exponent but readable table."""
    i = LRS.index(x)
    if i > 0:
        yield LRS[i - 1]
    if i < len(LRS) - 1:
        yield LRS[i + 1]


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


_spec_neighbors = {}


def all_conf_neighbors(conf):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    if conf.spec in _spec_neighbors:
        for neighbor_spec, vary in _spec_neighbors[conf.spec]:
            yield conf._replace(spec=neighbor_spec), vary
    else:
        cache = []
        _spec_neighbors[conf.spec] = cache
        for neighbor_spec, vary in conf.spec.all_neighbors():
            cache.append((neighbor_spec, VARY[vary]))
            yield conf._replace(spec=neighbor_spec), VARY[vary]

    yield conf._replace(batch=conf.batch * 2), VARY["batch"]
    if conf.batch > 1:
        yield conf._replace(batch=conf.batch // 2), VARY["batch"]

    for x in neighbor_numbers(conf.lr):
        yield conf._replace(lr=x), VARY["lr"]

    if conf.sample_len >= 4:
        yield conf._replace(sample_len=conf.sample_len // 2), VARY["len"]
    yield conf._replace(sample_len=conf.sample_len * 2), VARY["len"]

    for x in neighbor_numbers(conf.regularization):
        yield conf._replace(regularization=x), VARY["reg"]

    for x in neighbor_numbers(conf.init_scale):
        yield conf._replace(init_scale=x), VARY["init"]

    if conf.t > 1:
        yield conf._replace(t=conf.t // 2), VARY["time"]
    yield conf._replace(t=conf.t * 2), VARY["time"]


def conf_neighbors(conf, vary):
    """Generate all possible conf neighbors.

    Returns: iterator over pairs (neighbor conf, type of neighbor)
    """
    if conf.spec in _spec_neighbors:
        for neighbor_spec in _spec_neighbors[conf.spec]:
            yield conf._replace(spec=neighbor_spec)
    else:
        cache = []
        _spec_neighbors[conf.spec] = cache
        for neighbor_spec, v in conf.spec.all_neighbors():
            if v in vary:
                cache.append(neighbor_spec)
                yield conf._replace(spec=neighbor_spec)

    if "batch" in vary:
        yield conf._replace(batch=conf.batch * 2)
        if conf.batch > 1:
            yield conf._replace(batch=conf.batch // 2)

    if "lr" in vary:
        for x in neighbor_numbers(conf.lr):
            yield conf._replace(lr=x)

    if "len" in vary:
        if conf.sample_len >= 4:
            yield conf._replace(sample_len=conf.sample_len // 2)
        yield conf._replace(sample_len=conf.sample_len * 2)

    if "reg" in vary:
        for x in neighbor_numbers(conf.regularization):
            yield conf._replace(regularization=x)

    if "init" in vary:
        for x in neighbor_numbers(conf.init_scale):
            yield conf._replace(init_scale=x)

    if "time" in vary:
        if conf.t > 1:
            yield conf._replace(t=conf.t // 2)
        yield conf._replace(t=conf.t * 2)


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
