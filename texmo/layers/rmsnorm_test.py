"""Tests for the RMSNorm layer, including a numeric cross-check against
the HuggingFace RecurrentGemmaRMSNorm math:
    out = x * rsqrt(mean(x^2, axis=-1) + eps) * (1 + gamma), fp32 reduce.
"""
import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.rmsnorm import RmsNormDef, RmsNormJax, _EPS
from texmo.model import _build_layer_def


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
