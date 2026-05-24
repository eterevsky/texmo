import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.slstm import SLstmDef


def _make_jax(size=4, input_size=4, seed=0, dtype=jnp.float32):
    d = SLstmDef(size, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = SLstmDef(8, input_size=4)
    assert d.size == 8
    assert str(d) == "slstm.8"
    assert d.is_valid()
    # 4*8*(4+8) + 4*8 = 384 + 32 = 416 (identical to LSTM at same shape)
    assert d.num_weights == 416


def test_def_is_valid():
    assert SLstmDef(1, input_size=4).is_valid()
    assert SLstmDef(2, input_size=4).is_valid()
    assert SLstmDef(16, input_size=4).is_valid()
    assert not SLstmDef(3, input_size=4).is_valid()


# -- JAX --

def test_jax_weight_count_matches_def():
    d, _, weights = _make_jax(size=8, input_size=4)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shapes():
    _, layer, _ = _make_jax(size=4, input_size=4)
    h, c, n, m = layer.init_state()
    assert h.shape == (4,)
    assert c.shape == (4,)
    assert n.shape == (4,)
    assert m.shape == (4,)
    # All zeros at init.
    assert h.sum() == 0
    assert c.sum() == 0
    assert n.sum() == 0
    assert m.sum() == 0


def test_jax_step_shapes():
    _, layer, weights = _make_jax(size=4, input_size=8)
    state = layer.init_state()
    new_state, out = layer.step(weights, state, jnp.ones(8))
    h, c, n, m = new_state
    assert h.shape == (4,)
    assert c.shape == (4,)
    assert n.shape == (4,)
    assert m.shape == (4,)
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


# -- Neighbors --

def test_slstm_neighbors_size_mutations_and_family_swaps():
    d = SLstmDef(4, input_size=8)
    neighbors = list(d.neighbors())
    assert "slstm.2" in neighbors
    assert "slstm.8" in neighbors
    assert "lstm.4" in neighbors
    assert "matlstm.4" in neighbors
    assert "mullstm.4" in neighbors


def test_slstm_neighbors_at_size_1_skips_matlstm():
    """matlstm requires size >= 2; family swap must skip it at size 1."""
    d = SLstmDef(1, input_size=4)
    neighbors = list(d.neighbors())
    assert "slstm.2" in neighbors          # size mutation up
    assert "lstm.1" in neighbors           # family swap to lstm
    assert "mullstm.1" in neighbors        # family swap to mullstm
    assert "matlstm.1" not in neighbors    # matlstm needs size >= 2


def test_jax_long_sequence_numerically_stable():
    """The exponential input gate without stabilisation would overflow
    at long T; the m_t tracker keeps c and n bounded."""
    _, layer, weights = _make_jax(size=4, input_size=4)
    inputs = jax.random.normal(
        jax.random.PRNGKey(13), (1, 256, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert jnp.all(jnp.isfinite(out))
