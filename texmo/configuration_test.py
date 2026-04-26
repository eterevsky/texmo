from texmo.common import INF
from texmo.configuration import (
    Configuration,
    Template,
    conf_neighbors,
)
from texmo.model import build_model_def
from texmo.precision import Precision


def _md(spec):
    return build_model_def(spec, Precision.FP32)


def _conf(spec, lr=0.25, length=128, batch=256, steps=256, decay=1.0):
    return Configuration(
        model=_md(spec), lr=lr, length=length, batch=batch,
        steps=steps, decay=decay,
    )


def test_conf_neighbors():
    template = Template(
        spec=r"bytes\|(dense|rnn)\.\d+\.relu(-suffix\.\d+)?",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=None,
    )

    conf = _conf("bytes|dense.1.relu", decay=1 / 32)
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|dense.2.relu")),
        conf.replace(model=_md("bytes|rnn.1.relu")),
        conf.replace(model=_md("bytes|dense.1.relu-suffix.2")),
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(decay=1 / 16),
        conf.replace(decay=1 / 64),
        conf.replace(batch=128),
        # Toggle to cosine snaps decay to 1.
        conf.replace(cosine=True, decay=1.0),
    }


def test_dense_layer_neighbor():
    template = Template(
        spec=r"bytes\|(dense|rnn)\.\d+\.\w+",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=1,
    )

    conf = _conf("bytes|dense.1.relu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|dense.2.relu")),
        conf.replace(model=_md("bytes|rnn.1.relu")),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_suffix():
    template = Template(
        spec=r"bytes\|suffix\.\d+",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=(0, 1),
    )

    conf = _conf("bytes|suffix.2")
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(model=_md("bytes|suffix.4")),
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(batch=128),
        conf.replace(decay=1 / 2),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_suffix_exact():
    template = Template(
        spec='bytes|suffix.2',
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=(0, 1),
    )

    conf = _conf("bytes|suffix.2")
    neighbors = set(conf_neighbors(conf, template))
    assert neighbors == {
        conf.replace(lr=0.125),
        conf.replace(lr=0.5),
        conf.replace(batch=128),
        conf.replace(decay=1 / 2),
        conf.replace(cosine=True),
    }


def test_conf_neighbors_precision():
    template = Template(
        spec=r"bytes\|dense\.\d+\.relu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32, Precision.FP16],
        decay=1,
    )

    conf = _conf("bytes|dense.1.relu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    # Should include FP16 precision neighbor
    fp16_conf = conf.replace(
        model=build_model_def("bytes|dense.1.relu", Precision.FP16))
    assert fp16_conf in neighbors
    # FP64 not in template precision list, should be excluded
    fp64_conf = conf.replace(
        model=build_model_def("bytes|dense.1.relu", Precision.FP64))
    assert fp64_conf not in neighbors


def test_conf_neighbors_input():
    template = Template(
        spec=r"(bytes|bits\.\d+[\w.+]*)\|dense\.\d+\.relu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=None,
        precision=[Precision.FP32],
        decay=1,
    )

    conf = _conf("bytes|dense.1.relu", batch=128, decay=1)
    neighbors = set(conf_neighbors(conf, template))
    specs = {str(n.model) for n in neighbors}
    assert "bits.4.oh+bp|dense.1.relu" in specs


def test_template_exact_spec_match():
    """When spec is a valid model, Template uses exact match, not regex.

    bits.1+bp contains '+' which is a regex special char — if treated as
    regex it would match wrong specs. Exact match ensures only the
    precise spec is accepted.
    """
    template = Template(
        spec="bits.1+bp|dense.1.relu",
        lr=None,
        length=None,
        batch=None,
        steps=None,
        max_weights=None,
        precision=[Precision.FP32],
        decay=None,
    )

    assert template.spec == "bits.1+bp|dense.1.relu"
    assert template.regex is None

    conf_match = _conf("bits.1+bp|dense.1.relu")
    assert template.match(conf_match)

    # Should NOT match a different spec, even if '+' were treated as regex
    conf_no_match = _conf("bits.1|dense.1.relu")
    assert not template.match(conf_no_match)

    conf_bp2 = _conf("bits.2+bp|dense.1.relu")
    assert not template.match(conf_bp2)


def test_conf_is_valid():
    conf = _conf("bytes|rnn.64.tanh-dense.32.gelu", decay=1 / 32)
    assert conf.is_valid()


def test_conf_not_valid_no_activation():
    conf = _conf("bytes|dense.32", decay=1 / 32)
    assert not conf.is_valid()


def test_cosine_requires_decay_one():
    conf_ok = _conf("bytes|dense.32.gelu", decay=1.0).replace(cosine=True)
    assert conf_ok.is_valid()
    conf_bad = _conf("bytes|dense.32.gelu", decay=0.5).replace(cosine=True)
    assert not conf_bad.is_valid()


def test_cosine_neighbor_toggle_from_exp_decay():
    """Toggling cosine on snaps decay to 1, regardless of current decay."""
    template = Template(
        spec="bytes|dense.32.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=(0, 1),
    )
    conf = _conf("bytes|dense.32.gelu", batch=128, decay=0.5)
    neighbors = set(conf_neighbors(conf, template))
    assert conf.replace(cosine=True, decay=1.0) in neighbors
    # Direct (cosine=True, decay=0.5) is invalid and must NOT be a neighbor.
    assert conf.replace(cosine=True) not in neighbors


def test_cosine_neighbor_toggle_back_to_exp_decay():
    """From cosine=True the toggle goes to (cosine=False, decay=1)."""
    template = Template(
        spec="bytes|dense.32.gelu",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=[Precision.FP32],
        decay=(0, 1),
    )
    conf = _conf("bytes|dense.32.gelu", batch=128, decay=1.0).replace(
        cosine=True)
    neighbors = set(conf_neighbors(conf, template))
    assert conf.replace(cosine=False) in neighbors
    # exp-decay walk from cosine=True is suppressed; the off-toggle
    # gets us to decay=1, and the regular walk takes over from there.
    assert conf.replace(cosine=False, decay=0.5) not in neighbors
