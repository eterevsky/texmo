import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.matlstm import MatLstmDef


def _make_jax(size=4, input_size=4, seed=0, dtype=jnp.float32):
    d = MatLstmDef(size, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = MatLstmDef(4, input_size=8)
    assert d.size == 4
    assert str(d) == "matlstm.4"
    assert d.is_valid()
    # 3*4*8 (Q/K/V) + 3*8 (i/f/o input proj) + 3 (i/f/o bias) = 123
    assert d.num_weights == 123


def test_def_is_valid():
    assert MatLstmDef(2, input_size=4).is_valid()
    assert MatLstmDef(4, input_size=4).is_valid()
    # size must be a power of 2 and >= 2 (D=1 would degenerate to scalar).
    assert not MatLstmDef(1, input_size=4).is_valid()
    assert not MatLstmDef(3, input_size=4).is_valid()


# -- JAX --

def test_jax_weight_count_matches_def():
    d, _, weights = _make_jax(size=4, input_size=8)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shapes():
    _, layer, _ = _make_jax(size=4, input_size=4)
    C, n, m = layer.init_state()
    assert C.shape == (4, 4)
    assert n.shape == (4,)
    assert m.shape == ()
    # All zeros at init.
    assert C.sum() == 0
    assert n.sum() == 0
    assert m == 0


def test_jax_step_shapes():
    _, layer, weights = _make_jax(size=4, input_size=8)
    state = layer.init_state()
    new_state, out = layer.step(weights, state, jnp.ones(8))
    C, n, m = new_state
    assert C.shape == (4, 4)
    assert n.shape == (4,)
    assert m.shape == ()
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
    # Perturbation actually changes later outputs (sanity).
    assert not np.allclose(
        np.asarray(out_a[0, 5:]), np.asarray(out_b[0, 5:]))


def test_jax_long_sequence_numerically_stable():
    """The exponential input gate without stabilisation would overflow
    at long T; the m_t tracker keeps |C| bounded."""
    _, layer, weights = _make_jax(size=4, input_size=4)
    inputs = jax.random.normal(
        jax.random.PRNGKey(13), (1, 256, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert jnp.all(jnp.isfinite(out))
