"""Tests for the RG-LRU layer, including a numeric cross-check against
a direct reimplementation of the HuggingFace `RecurrentGemmaRglru`
math (block-diagonal gates, a = exp(-8*r*softplus(Lambda)), the
sqrt(1-a^2) input multiplier, reset at the first position, fp32 scan)."""
import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.rglru import RglruDef, RglruJax
from texmo.model import _build_layer_def


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _ref_forward(x, w, blocks, reset_first):
    """Mirror of transformers RecurrentGemmaRglru.forward (+ scan)."""
    B, T, D = x.shape
    bw = D // blocks
    xb = x.reshape(B, T, blocks, bw)
    ig = np.einsum('btkc,kcd->btkd', xb, w['w_ig']) + w['b_ig']
    rg = np.einsum('btkc,kcd->btkd', xb, w['w_rg']) + w['b_rg']
    i = _sigmoid(ig).reshape(B, T, D)
    r = _sigmoid(rg).reshape(B, T, D)
    log_a = -8.0 * r * _softplus(w['lam'])
    a = np.exp(log_a)
    a_sq = np.exp(2.0 * log_a)
    gated = x * i
    mult = np.sqrt(np.clip(1.0 - a_sq, 1e-12, 1.0))
    a = a.copy()
    if reset_first:
        mult[:, 0, :] = 1.0
        a[:, 0, :] = 0.0
    nx = gated * mult
    h = np.zeros((B, D), dtype=np.float64)
    out = np.zeros((B, T, D), dtype=np.float64)
    for t in range(T):
        h = a[:, t] * h + nx[:, t]
        out[:, t] = h
    return out


def _make(input_size=8, blocks=2, seed=0):
    layer = RglruJax(input_size, blocks, dtype=jnp.float32)
    w = layer.init_weights(jax.random.PRNGKey(seed))
    return layer, w


def _np_weights(w):
    return {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}


def test_forward_shape():
    layer, w = _make(input_size=8, blocks=2)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 8))
    out = layer.forward(w, x)
    assert out.shape == (3, 5, 8)


def test_forward_matches_reference():
    layer, w = _make(input_size=8, blocks=2)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 7, 8))
    out = np.asarray(layer.forward(w, x), dtype=np.float64)
    ref = _ref_forward(np.asarray(x, np.float64), _np_weights(w),
                       blocks=2, reset_first=True)
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_step_matches_no_reset_reference():
    # `step` carries state with no first-position reset (its contract is
    # a pure recurrence from the given state).
    layer, w = _make(input_size=8, blocks=2)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 6, 8))
    # step is per-sample (size,) -> loop samples and time.
    outs = np.zeros((2, 6, 8))
    for b in range(2):
        s = layer.init_state()
        for t in range(6):
            s, y = layer.step(w, s, x[b, t])
            outs[b, t] = np.asarray(y)
    ref = _ref_forward(np.asarray(x, np.float64), _np_weights(w),
                       blocks=2, reset_first=False)
    assert np.allclose(outs, ref, atol=1e-5, rtol=1e-4)


def test_num_weights_matches_init():
    blocks = 2
    d = RglruDef(blocks, input_size=8)
    _, w = _make(input_size=8, blocks=blocks)
    total = sum(int(np.asarray(v).size) for v in w.values())
    assert d.num_weights == total


def test_validity_and_recurrentgemma_shape():
    # 2B config: lru_width=2560, num_heads=10 -> block_width=256. Runs
    # (2560 % 10 == 0) but is NOT search-valid: blocks=10 isn't a power
    # of 2. RecurrentGemma is loaded outside the search.
    rg = RglruDef(10, input_size=2560)
    assert rg.block_width == 256
    assert not rg.is_valid()
    # Power-of-2 blocks that divide the width are valid.
    assert RglruDef(8, input_size=2560).is_valid()
    assert RglruDef(1, input_size=8).is_valid()
    # Non-power-of-2 blocks -> invalid.
    assert not RglruDef(3, input_size=2560).is_valid()
    # Power-of-2 blocks that don't divide the width -> invalid.
    assert not RglruDef(16, input_size=8).is_valid()


def test_parse_roundtrip():
    layer_def = _build_layer_def("rglru.2", input_size=8)
    assert isinstance(layer_def, RglruDef)
    assert layer_def.blocks == 2
    assert str(layer_def) == "rglru.2"


def test_model2_build():
    # rglru is reachable through the Model2 path (parse_model2 uses the
    # same _build_layer_def factory).
    from texmo.precision import Precision
    from texmo.spec_parser import parse_model2
    m = parse_model2("bytes|dense.8.gelu-rglru.2", Precision.FP32)
    assert "rglru" in [l.name for l in m.layer_seq.layers]
    m.build_jax()  # constructs without error
