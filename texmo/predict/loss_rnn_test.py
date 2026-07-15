"""Tests for the loss-prediction RNN's feature extraction, focused on
the split handling.

split.add / split.cat markers use the feature conventions inherited
from the retired skip representation (jump-length slots, merged-in
output width), so data labeled before the representation switch maps
to the same features. split.mul (no skip analog) gets its own
jump-length slot.
"""

import numpy as np

from texmo.configuration import Configuration
from texmo.precision import Precision
from texmo.predict.loss_rnn import (
    N_INIT_GLOBAL,
    LossModel,
    _init_global_features,
    _is_simple_type,
    _layer_features,
    model_layers,
    discover_simple_types,
    fit,
)
from texmo.predict.predict_common import layer_type_id
from texmo.spec_parser import parse_model2


def _conf(model):
    return Configuration(
        model, lr=0.01, length=64, batch=16, steps=128, decay=1.0)


def _m2(spec):
    return _conf(parse_model2(spec, Precision.FP32))


def _feature_matrix(conf, simple_types):
    idx = {t: i for i, t in enumerate(simple_types)}
    n = len(simple_types)
    return np.array(
        [_layer_features(l, idx, n) for l in model_layers(conf)])


def _marker_features(conf):
    st = discover_simple_types([(conf, 0.0)])
    layers = model_layers(conf)
    marker = next(l for l in layers if layer_type_id(l).startswith('split.'))
    return (
        _layer_features(marker, {t: i for i, t in enumerate(st)}, len(st)),
        len(st),
    )


def test_split_add_dist_slot():
    conf = _m2(
        "bytes|dense.32.gelu-split.add(dense.32.gelu, pass)-dense.32.gelu")
    f, n = _marker_features(conf)
    assert f[3 + n + 3] == 1.0  # add-span slot = main-branch length
    assert f[3 + n + 4] == 0.0  # cat slot untouched


def test_split_cat_dist_slot_and_width_convention():
    # cat changes the merged width -- the marker reports the merged-in
    # (source = input) channels, not the summed width, matching how
    # data was labeled under the retired skip representation.
    conf = _m2(
        "bytes|dense.32.gelu-split.cat(suffix.2-dense.32.gelu, pass)"
        "-dense.16.gelu")
    f, n = _marker_features(conf)
    assert f[3 + n + 4] == 2.0            # cat-span slot = 2 layers
    assert f[2] == np.log2(32)            # out width = input channels


def test_flatten_inlines_main_branch_recursively():
    conf = _m2(
        "bits.1+bp|dense.4.gelu-split.add(dense.4.tanh-norm, pass)"
        "-dense.4.gelu")
    types = [layer_type_id(l) for l in model_layers(conf)]
    assert types == [
        "dense.gelu", "split.add", "dense.tanh", "norm", "dense.gelu"]


def test_split_mul_gate_folded_and_dist_slot():
    conf = _m2("bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)")
    st = discover_simple_types([(conf, 0.0)])
    n = len(st)
    layers = model_layers(conf)
    # marker + inlined main branch; the gate is NOT a separate step.
    assert [layer_type_id(l) for l in layers] == [
        "dense.gelu", "split.mul", "dense.gelu"]
    f = _layer_features(layers[1], {t: i for i, t in enumerate(st)}, n)
    # Gate (dense.4) weights folded into the marker; main branch is its
    # own step, so its weights are not double-counted here.
    assert f[0] > 0
    # split.mul uses its own slot; the reused skip/residual slots stay 0.
    assert f[3 + n + 8] == 1.0   # split_mul_dist == main-branch span
    assert f[3 + n + 3] == 0.0   # skip/split add
    assert f[3 + n + 4] == 0.0   # skip/split cat


def test_self_gate_marker_has_no_gate_weights():
    # split.mul(X, pass): the gate is the identity, so no folded weights.
    conf = _m2("bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, pass)")
    st = discover_simple_types([(conf, 0.0)])
    n = len(st)
    marker = model_layers(conf)[1]
    f = _layer_features(marker, {t: i for i, t in enumerate(st)}, n)
    assert f[0] == 0.0           # pass gate -> no weights
    assert f[3 + n + 8] == 1.0   # still a mul with span 1


def test_split_is_not_a_simple_type():
    assert not _is_simple_type("split.add")
    assert not _is_simple_type("split.mul")
    assert _is_simple_type("dense.gelu")


def test_fit_predict_on_mixed_specs():
    confs = [
        _m2("bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)"),
        _m2("bits.1+bp|dense.4.gelu-split.add(dense.4.gelu, pass)"),
        _m2("bytes|dense.8.gelu"),
    ]
    data = [(c, 1.5) for c in confs] * 10
    st = discover_simple_types(data)
    params, max_layers, _ = fit(data, st, steps=20, seed=0)
    preds = LossModel(
        params=params, simple_types=st, max_layers=max_layers,
    ).predict(confs)
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))


def test_init_globals_codec_features():
    tied = _m2("bytes.emb.4|dense.4.tanh")
    g = _init_global_features(tied)
    assert g.shape == (N_INIT_GLOBAL,)
    assert g[7] == 1.0  # tied head: no parameters of its own
    assert abs(g[8] - np.log2(1 + 256 * 4 + 1)) < 1e-6  # table + scale
    # Total weight budget: the whole model, codec included.
    assert abs(g[9] - np.log2(tied.model.num_weights)) < 1e-6

    plain = _m2("bytes|dense.4.gelu")
    g = _init_global_features(plain)
    assert g[7] == 0.0
    # One-hot codec weights = the implicit dense head (4*256+256).
    assert abs(g[8] - np.log2(1 + 4 * 256 + 256)) < 1e-6
