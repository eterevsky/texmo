"""Tests for the loss-prediction RNN's feature extraction, focused on
the split handling.

split.add / split.cat markers use the feature conventions inherited
from the retired skip representation (jump-length slots, merged-in
output width), so data labeled before the representation switch maps
to the same features. split.mul (no skip analog) gets its own
jump-length slot.
"""

import os

import numpy as np

from texmo.configuration import Configuration
from texmo.layers.pair_codec import PairCodecDef
from texmo.precision import Precision
from texmo.tokens import set_tokens_dir
from texmo.predict.loss_rnn import (
    _LOG_MAX,
    N_INIT_GLOBAL,
    LossModel,
    _init_global_features,
    _is_simple_type,
    _layer_features,
    discover_simple_types,
    fit,
    model_layers,
    predict,
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


def test_gated_head_fits_and_bounds():
    """The gated head trains and interpolates toward the divergence
    clip: predictions stay below LOG_MAX and a diverged-labeled conf
    pulls the gate open."""
    confs = [
        _m2('bits.1+bp|dense.4.gelu'),
        _m2('bytes|gru.8'),
    ]
    # First conf converges (loss 1.5), second reliably diverges.
    data = ([(confs[0], 1.5)] + [(confs[1], 12.0)]) * 30
    st = discover_simple_types(data)
    params, ml, _ = fit(
        data, st, steps=200, seed=0, gated_head=True,
        feat_proj=8, out_hidden=8)
    assert 'W_gate' in params
    preds = predict(
        params, confs, st, ml, feat_proj=8, out_hidden=8,
        gated_head=True)
    assert preds[1] > preds[0]
    # The gate interpolates toward the clip; it only hard-bounds when
    # the base regressor sits below it, so allow a small overshoot
    # (the exact landing point is seed- and feature-schema-sensitive
    # at this toy step count).
    assert preds[1] <= _LOG_MAX + 0.05


def test_tokenset_globals():
    set_tokens_dir(
        os.path.join(os.path.dirname(__file__), "..", "..", "tokens"))
    # Absolute slots, not negative ones: the pair features (2026-08-22)
    # were appended after these, so counting from the end would silently
    # read the wrong feature the next time a global is added.
    residual, log_bpt = 10, 11
    g = _init_global_features(_m2("tokens.32.fold.oh|rnn.4.tanh"))
    assert g.shape == (N_INIT_GLOBAL,)
    assert 0.3 < g[residual] < 0.4  # the fold set's residual charge
    assert g[log_bpt] == 0.0  # one token per byte
    g = _init_global_features(_m2("bits.4.oh+bp|rnn.4.tanh"))
    assert g[residual] == 0.0  # lossless input
    assert g[log_bpt] == -1.0  # two tokens per byte
    g = _init_global_features(_m2("bits.1+bp|rnn.4.tanh"))
    assert g[log_bpt] == -3.0  # eight tokens per byte


# -- hex-pair globals ---------------------------------------------------

# The three appended slots: (is .add, is .K, log2(1+k)).
_PAIR_SLOTS = slice(12, 15)


def test_pair_family_gets_its_own_globals():
    """The pair arms are a different output factorization, not a byte
    model with a cheap head -- the predictor is told so explicitly."""
    set_tokens_dir(
        os.path.join(os.path.dirname(__file__), "..", "..", "tokens"))

    g = _init_global_features(_m2("bits.4.pair.add|rnn.8.gelu"))
    assert g.shape == (N_INIT_GLOBAL,)
    np.testing.assert_array_equal(g[_PAIR_SLOTS], [1.0, 0.0, 0.0])

    g = _init_global_features(_m2("bits.4.pair.16|rnn.8.gelu"))
    np.testing.assert_allclose(
        g[_PAIR_SLOTS], [0.0, 1.0, np.log2(17)], rtol=1e-6)

    # Every non-pair codec leaves all three at zero -- including
    # `bytes`, which shares the pair arms' 256 logits and byte
    # granularity and is exactly what they'd otherwise be confused for.
    for spec in ("bytes|rnn.8.gelu", "bytes.emb.8|dense.8.tanh",
                 "bits.4.oh+bp|rnn.8.gelu", "bits.1+bp|dense.4.gelu",
                 "tokens.32.hexbpe.oh|rnn.4.tanh"):
        g = _init_global_features(_m2(spec))
        np.testing.assert_array_equal(
            g[_PAIR_SLOTS], [0.0, 0.0, 0.0], err_msg=spec)


def test_pair_log_k_is_smooth_and_distinguishes_k():
    """log2(1+k) is a real signal, not recoverable from the weight
    count: k and X trade off inside (16+k)X + 33k + 32."""
    ks = (1, 4, 16, 64)
    vals = [
        float(_init_global_features(
            _m2(f"bits.4.pair.{k}|rnn.8.gelu"))[14])
        for k in ks
    ]
    assert vals == sorted(vals)
    for k, v in zip(ks, vals):
        assert abs(v - np.log2(1 + k)) < 1e-6
    # Two confs with the SAME total weight count but different k still
    # differ in the globals.
    a = _init_global_features(_m2("bits.4.pair.4|rnn.8.gelu"))
    b = _init_global_features(_m2("bits.4.pair.64|rnn.8.gelu"))
    assert a[14] != b[14]


def test_pair_globals_detected_by_type_not_by_spec_string():
    """Detection is isinstance(codec, PairCodecDef) + codec.k, so a
    tokenset whose NAME contains 'pair' would not be miscounted."""
    conf = _m2("bits.4.pair.8|rnn.8.gelu")
    assert isinstance(conf.model.codec, PairCodecDef)
    assert conf.model.codec.k == 8
    g = _init_global_features(conf)
    np.testing.assert_allclose(
        g[_PAIR_SLOTS], [0.0, 1.0, np.log2(9)], rtol=1e-6)


def test_n_init_global_matches_the_vector():
    """The constant sizes W_glob in both predictors; a drift here is a
    silent shape bug at fit time."""
    for spec in ("bytes|dense.4.gelu", "bits.4.pair.add|rnn.8.gelu",
                 "bits.4.pair.16|rnn.8.gelu"):
        assert _init_global_features(_m2(spec)).shape == (N_INIT_GLOBAL,)
    assert N_INIT_GLOBAL == 15
