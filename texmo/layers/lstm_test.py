import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.lstm import LstmDef


def test_def_properties():
    d = LstmDef(32, input_size=16)
    assert d.size == 32
    assert str(d) == "lstm.32"
    assert d.is_valid()
    # Single-bias per gate (matches LstmJax).
    assert d.num_weights == 4 * 32 * (16 + 32) + 4 * 32


def test_non_power2_invalid():
    assert not LstmDef(13, input_size=8).is_valid()


def test_neighbors():
    d = LstmDef(32, input_size=16)
    neighbors = list(d.neighbors())
    # Size mutations
    assert "lstm.16" in neighbors
    assert "lstm.64" in neighbors
    # Type swaps (gru and mgru only among the older recurrent layers)
    assert "gru.32" in neighbors
    assert "mgru.32" in neighbors
    # Not a neighbor of mingru or rnn
    assert "mingru.32" not in neighbors
    for act in ("relu", "gelu", "tanh"):
        assert f"rnn.32.{act}" not in neighbors
    # All-to-all within the LSTM family.
    assert "matlstm.32" in neighbors
    assert "slstm.32" in neighbors
    assert "mullstm.32" in neighbors


def test_neighbors_lstm_size_1_skips_matlstm():
    """matlstm needs size >= 2, so swapping from lstm.1 skips it."""
    d = LstmDef(1, input_size=4)
    neighbors = list(d.neighbors())
    assert "slstm.1" in neighbors
    assert "mullstm.1" in neighbors
    assert "matlstm.1" not in neighbors


def test_gru_has_lstm_neighbor():
    from texmo.layers.gru import GruDef, MgruDef, MinGruDef
    g = list(GruDef(32, input_size=16).neighbors())
    assert "lstm.32" in g

    m = list(MgruDef(32, input_size=16).neighbors())
    assert "lstm.32" in m

    mg = list(MinGruDef(32, input_size=16).neighbors())
    assert "lstm.32" not in mg


# -- JAX --

def test_jax_forward_and_step():
    d = LstmDef(8, input_size=4)
    layer = d.build_jax(jnp.float32)

    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    assert weights['w_ih'].shape == (32, 4)
    assert weights['w_hh'].shape == (32, 8)
    assert weights['b'].shape == (32,)

    inputs = jax.random.normal(rng, (2, 6, 4), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    assert fwd.shape == (2, 6, 8)

    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)
        # The emitted value is h, not the cell state c.
        h, c = state
        np.testing.assert_array_equal(out, h)
        assert c.shape == (8,)


def test_jax_weight_count_matches_def():
    d = LstmDef(16, input_size=8)
    layer = d.build_jax(jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shape():
    layer = LstmDef(8, input_size=4).build_jax(jnp.float32)
    h, c = layer.init_state()
    assert h.shape == (8,)
    assert c.shape == (8,)
    assert h.dtype == jnp.float32
    assert c.dtype == jnp.float32
