"""Tests for the structure-mirroring (tree) loss predictor."""

import numpy as np

from texmo.configuration import Configuration
from texmo.precision import Precision
from texmo.predict.loss_tree import (
    TreeLossModel,
    build_arrays,
    compile_conf,
    discover_simple_types,
    fit,
    predict,
)
from texmo.spec_parser import parse_model2


def _conf(spec):
    return Configuration(
        parse_model2(spec, Precision.FP32),
        lr=0.01, length=64, batch=16, steps=128, decay=1.0)


def _type_idx(confs):
    st = discover_simple_types([(c, 1.0) for c in confs])
    return st, {t: i for i, t in enumerate(st)}


def test_compile_plain_chain():
    c = _conf('bits.1+bp|dense.4.gelu-gru.8')
    st, ti = _type_idx([c])
    p, s1, s2, d, r = compile_conf(c, ti, len(st))
    assert p.shape[0] == 2 and r == 1
    np.testing.assert_array_equal(s1, [0, 0])
    np.testing.assert_array_equal(d, [0, 0])
    assert p[:, -1].sum() == 0  # no merges


def test_compile_split_forks_and_merges():
    c = _conf('bits.1+bp|split.mul(dense.4.gelu, dense.4)')
    st, ti = _type_idx([c])
    p, s1, s2, d, r = compile_conf(c, ti, len(st))
    # Two branch instructions off register 0, then the merge back
    # into register 0.
    assert r == 3
    np.testing.assert_array_equal(s1, [0, 0, 1])
    np.testing.assert_array_equal(s2, [0, 0, 2])
    np.testing.assert_array_equal(d, [1, 2, 0])
    np.testing.assert_array_equal(p[:, -1], [0, 0, 1])


def test_compile_nested_split_preserves_fork_values():
    """The fork register must stay unwritten until its merge reads it
    (the pass branch reads the pre-split value)."""
    c = _conf(
        'bytes|dense.16.gelu-split.add(dense.16.gelu'
        '-split.cat(dense.16.gelu, pass), pass)-dense.16.gelu')
    st, ti = _type_idx([c])
    p, s1, s2, d, r = compile_conf(c, ti, len(st))
    # The outer pass branch reads register 0 at the final merge; the
    # only earlier write to 0 is the pre-split dense.
    merge_ts = [t for t in range(p.shape[0]) if p[t, -1] > 0.5]
    outer = merge_ts[-1]
    assert s2[outer] == 0
    writes_to_0 = [t for t in range(outer) if d[t] == 0]
    assert writes_to_0 == [0]  # only the pre-split dense


def test_empty_chain_compiles():
    c = _conf('bits.1+bp|')
    st, ti = _type_idx([_conf('bits.1+bp|dense.4.gelu'), c])
    p, s1, s2, d, r = compile_conf(c, ti, len(st))
    assert p.shape[0] == 1 and r == 1


def test_build_arrays_pads_and_masks():
    confs = [_conf('bits.1+bp|dense.4.gelu'),
             _conf('bits.1+bp|dense.4.gelu-gru.8-dense.2.tanh')]
    st, ti = _type_idx(confs)
    globs, p, s1, s2, d, m, R = build_arrays(confs, ti, len(st))
    assert p.shape[:2] == (2, 3)
    np.testing.assert_array_equal(m, [[1, 0, 0], [1, 1, 1]])


def test_fit_predict_roundtrip():
    confs = [_conf('bits.1+bp|dense.4.gelu'),
             _conf('bytes|gru.8'),
             _conf('bits.1+bp|split.mul(dense.4.gelu, dense.4)')]
    data = [(c, l) for c, l in zip(confs, (1.0, 2.0, 4.0))] * 10
    st = discover_simple_types(data)
    params, trace = fit(data, st, steps=300, seed=0, batch_size=8)
    assert trace[-1] < trace[0]
    model = TreeLossModel(params=params, simple_types=st)
    preds = model.predict(confs)
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))
    # The fit had enough capacity/steps to order three distinct confs.
    assert preds[0] < preds[1] < preds[2]


def test_fit_predict_per_type_fwd():
    confs = [_conf('bits.1+bp|dense.4.gelu'),
             _conf('bytes|gru.8')]
    data = [(c, l) for c, l in zip(confs, (1.0, 4.0))] * 10
    st = discover_simple_types(data)
    params, _ = fit(data, st, steps=200, seed=0, batch_size=8,
                    per_type_fwd=True)
    assert params['W_step'].ndim == 3  # weight bank
    preds = predict(params, confs, st, per_type_fwd=True)
    assert np.all(np.isfinite(preds))
    assert preds[0] < preds[1]


def test_predict_handles_unseen_deeper_conf():
    """Predict-time confs may need more registers or longer tapes than
    training saw -- the register file sizes itself per batch."""
    train_confs = [_conf('bits.1+bp|dense.4.gelu')]
    data = [(c, 1.5) for c in train_confs] * 5
    st = discover_simple_types(data)
    params, _ = fit(data, st, steps=50, seed=0, batch_size=4)
    deep = _conf(
        'bits.1+bp|split.add(dense.4.gelu'
        '-split.add(dense.4.gelu, pass), pass)-dense.4.gelu')
    preds = predict(params, [deep], st)
    assert np.all(np.isfinite(preds))
