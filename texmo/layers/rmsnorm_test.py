"""Tests for the RMSNorm layer, including a numeric cross-check against
the HuggingFace RecurrentGemmaRMSNorm math:
    out = x * rsqrt(mean(x^2, axis=-1) + eps) * (1 + gamma), fp32 reduce.
"""
import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.rmsnorm import RmsNormDef, RmsNormJax, _EPS
from texmo.spec_parser import _build_layer_def


def _ref(x, gamma, eps=_EPS):
    x64 = x.astype(np.float64)
    ms = np.mean(x64 * x64, axis=-1, keepdims=True)
    normed = x64 / np.sqrt(ms + eps)
    return normed * (1.0 + gamma.astype(np.float64))


def _make(input_size=8, seed=0, gamma_scale=0.1):
    layer = RmsNormJax(input_size, dtype=jnp.float32)
    g = jax.random.normal(
        jax.random.PRNGKey(seed), (input_size,)) * gamma_scale
    return layer, {'gamma': g}


def test_init_gamma_zero():
    # gamma init 0 -> identity scale at start.
    w = RmsNormJax(8, jnp.float32).init_weights(jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(w['gamma']), 0.0)


def test_forward_matches_reference():
    layer, w = _make(8)
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 7, 8))
    out = np.asarray(layer.forward(w, x), dtype=np.float64)
    ref = _ref(np.asarray(x, np.float64), np.asarray(w['gamma'], np.float64))
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_gamma_zero_gives_unit_rms():
    # With gamma = 0 the output has rms ~ 1 (not L2 norm 1) regardless of
    # input scale.
    layer = RmsNormJax(16, jnp.float32)
    w = {'gamma': jnp.zeros(16)}
    x = jax.random.normal(jax.random.PRNGKey(2), (3, 5, 16)) * 7.0
    out = np.asarray(layer.forward(w, x))
    rms = np.sqrt(np.mean(out * out, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3)


def test_step_matches_forward():
    # Stateless: per-position step equals forward.
    layer, w = _make(8)
    x = jax.random.normal(jax.random.PRNGKey(3), (6, 8))  # one sample, T=6
    fwd = np.asarray(layer.forward(w, x[None]))[0]
    s = layer.init_state()
    outs = []
    for t in range(6):
        s, y = layer.step(w, s, x[t])
        outs.append(np.asarray(y))
    assert np.allclose(np.stack(outs), fwd, atol=1e-6)


def test_num_weights_and_validity():
    assert RmsNormDef(input_size=2560).num_weights == 2560
    layer = RmsNormJax(8, jnp.float32)
    w = layer.init_weights(jax.random.PRNGKey(0))
    assert int(np.asarray(w['gamma']).size) == RmsNormDef(input_size=8).num_weights
    assert RmsNormDef(input_size=8).is_valid()
    assert not RmsNormDef(input_size=1).is_valid()  # degenerate


def test_parse_and_model2_build():
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    ld = _build_layer_def("rmsnorm", input_size=8)
    assert isinstance(ld, RmsNormDef) and str(ld) == "rmsnorm"
    m = parse_model2("bytes|dense.16.gelu-rmsnorm", Precision.FP32)
    assert "rmsnorm" in [l.name for l in m.layer_seq.layers]
    m.build_jax()  # constructs without error


def test_norm_rmsnorm_neighbors_swap_insert_remove():
    from texmo.layers.norm import NormDef
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    # Def-level swap both directions.
    assert "rmsnorm" in list(NormDef(input_size=8).neighbors())
    assert "norm" in list(RmsNormDef(input_size=8).neighbors())
    # Model-level norm <-> rmsnorm swap.
    m = parse_model2("bytes|dense.8.gelu-norm-dense.8.gelu", Precision.FP32)
    assert "bytes|dense.8.gelu-rmsnorm-dense.8.gelu" in {
        nb.spec for nb in m.neighbors()}
    # Insert rmsnorm into a plain chain, and remove it again.
    m2 = parse_model2("bytes|dense.8.gelu-dense.8.gelu", Precision.FP32)
    assert any("rmsnorm" in nb.spec for nb in m2.neighbors())
    m3 = parse_model2("bytes|dense.8.gelu-rmsnorm-dense.8.gelu", Precision.FP32)
    specs3 = {nb.spec for nb in m3.neighbors()}
    assert "bytes|dense.8.gelu-dense.8.gelu" in specs3       # removed
    assert "bytes|dense.8.gelu-norm-dense.8.gelu" in specs3  # rmsnorm->norm


def test_rmsnorm_validity_rules():
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2

    def ok(spec):
        return parse_model2(spec, Precision.FP32).is_valid()
    assert ok("bytes|dense.8.gelu-rmsnorm-dense.8.gelu")
    assert not ok("bytes|rmsnorm-dense.8.gelu")          # first layer
    assert not ok("bytes|dense.8.gelu-rmsnorm-rmsnorm")  # adjacent
    assert not ok("bytes|dense.8.gelu-norm-rmsnorm")     # adjacent (mixed)
    assert not ok("bytes|suffix.2-rmsnorm")              # after suffix


def test_predictors_handle_rmsnorm():
    from texmo.configuration import Configuration
    from texmo.precision import Precision
    from texmo.predict import timing
    from texmo.predict.loss_rnn import _layer_features
    from texmo.predict.predict_common import model_layers
    from texmo.spec_parser import parse_model2
    model = parse_model2("bytes|dense.8.gelu-rmsnorm", Precision.FP32)
    conf = Configuration(model, lr=0.01, length=32, batch=4, steps=1,
                         decay=1.0)
    # Timing: rmsnorm gets a component (no unknown-layer error).
    assert any(c.type_id == "rmsnorm" for c in timing.featurize(conf))
    # Loss: rmsnorm is a simple type, and its weights (size) show up in
    # feat[0] = log2(num_weights) -- unlike the parameter-free norm (0).
    rm = next(l for l in model_layers(conf) if l.name == "rmsnorm")
    feat = _layer_features(rm, {"rmsnorm": 0}, n_simple=1)
    assert feat[0] > 0      # log2(size) > 0
    assert feat[3] == 1.0   # one-hot for the simple type "rmsnorm"
