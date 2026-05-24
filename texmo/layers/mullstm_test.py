import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.mullstm import MulLstmDef


def _make_jax(size=4, input_size=4, seed=0, dtype=jnp.float32):
    d = MulLstmDef(size, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = MulLstmDef(8, input_size=4)
    assert d.size == 8
    assert str(d) == "mullstm.8"
    assert d.is_valid()
    # 5*8*(4+8) + 4*8 = 480 + 32 = 512
    assert d.num_weights == 512


def test_def_is_valid():
    assert MulLstmDef(1, input_size=4).is_valid()
    assert MulLstmDef(2, input_size=4).is_valid()
    assert MulLstmDef(16, input_size=4).is_valid()
    assert not MulLstmDef(3, input_size=4).is_valid()


def test_num_weights_is_lstm_plus_multiplicative():
    """num_weights should be LSTM + size*(input_size + size)."""
    size, input_size = 8, 6
    lstm_weights = 4 * size * (input_size + size) + 4 * size
    extra = size * (input_size + size)  # w_mx + w_mh, no biases
    d = MulLstmDef(size, input_size=input_size)
    assert d.num_weights == lstm_weights + extra


# -- JAX --

def test_jax_weight_count_matches_def():
    d, _, weights = _make_jax(size=8, input_size=4)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shapes():
    _, layer, _ = _make_jax(size=4, input_size=4)
    h, c = layer.init_state()
    assert h.shape == (4,)
    assert c.shape == (4,)
    assert h.sum() == 0
    assert c.sum() == 0


def test_jax_step_shapes():
    _, layer, weights = _make_jax(size=4, input_size=8)
    state = layer.init_state()
    new_state, out = layer.step(weights, state, jnp.ones(8))
    h, c = new_state
    assert h.shape == (4,)
    assert c.shape == (4,)
    assert out.shape == (4,)


def test_jax_forward_shape():
    _, layer, weights = _make_jax(size=4, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (2, 5, 8), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (2, 5, 4)


def test_jax_forward_matches_step():
    """Hoisted-projection scan and per-token step must agree."""
    _, layer, weights = _make_jax(size=4, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(42), (2, 6, 8), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_causality():
    """Perturbing token t+1 must not change output at position t."""
    _, layer, weights = _make_jax(size=4, input_size=4)
    a = jax.random.normal(
        jax.random.PRNGKey(7), (1, 8, 4), dtype=jnp.float32)
    b = a.at[0, 5:].set(jax.random.normal(
        jax.random.PRNGKey(8), (3, 4), dtype=jnp.float32))
    out_a = layer.forward(weights, a)
    out_b = layer.forward(weights, b)
    np.testing.assert_allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
    assert not np.allclose(
        np.asarray(out_a[0, 5:]), np.asarray(out_b[0, 5:]))
