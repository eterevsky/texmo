import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.rnn import RnnDef


def test_def_properties_tanh():
    d = RnnDef(32, activation="tanh", input_size=16)
    assert d.size == 32
    assert str(d) == "rnn.32.tanh"
    assert d.is_valid()
    # Single-bias count (matches RnnJax).
    assert d.num_weights == 32 * 16 + 32 * 32 + 32


def test_def_properties_gelu():
    d = RnnDef(16, activation="gelu", input_size=8)
    assert str(d) == "rnn.16.gelu"
    assert d.is_valid()
    # Linear weights: W(input+size, size) + b(size)
    assert d.num_weights == 16 * (8 + 16) + 16


def test_def_properties_silu():
    d = RnnDef(16, activation="silu", input_size=8)
    assert str(d) == "rnn.16.silu"
    assert d.is_valid()


def test_def_no_activation_invalid():
    d = RnnDef(32, input_size=16)
    assert not d.is_valid()


def test_def_relu_retired_from_search():
    # Parses and runs, but no longer search-valid (see DenseDef).
    assert not RnnDef(32, activation="relu", input_size=16).is_valid()


def test_def_non_power2_invalid():
    d = RnnDef(13, activation="tanh", input_size=8)
    assert not d.is_valid()


def test_neighbors():
    d = RnnDef(32, activation="tanh", input_size=16)
    neighbors = list(d.neighbors())
    assert "rnn.16.tanh" in neighbors
    assert "rnn.64.tanh" in neighbors
    assert "dense.32.tanh" in neighbors
    # Recurrent family swaps
    assert "gru.32" in neighbors
    assert "mgru.32" in neighbors
    assert "mingru.32" in neighbors


def test_neighbors_activation_cycle_is_complete():
    # Every search-valid rnn activation is one mutation from every
    # other, so none is a dead end. rnn has no bare form.
    acts = ("tanh", "gelu", "silu")
    for act in acts:
        nbs = list(RnnDef(8, activation=act, input_size=4).neighbors())
        for other in acts:
            if other == act:
                continue
            assert f"rnn.8.{other}" in nbs, (act, other)
        assert f"rnn.8.{act}" not in nbs          # no self-edge
        assert "rnn.8" not in nbs                 # no bare rnn


def test_neighbors_size1():
    d = RnnDef(1, activation="gelu", input_size=4)
    neighbors = list(d.neighbors())
    assert "rnn.2.gelu" in neighbors
    assert "dense.1.gelu" in neighbors
    assert "gru.1" in neighbors


# -- JAX --

@pytest.mark.parametrize('activation', ['tanh', 'relu', 'gelu', 'silu'])
def test_jax_forward_and_step(activation):
    d = RnnDef(8, activation=activation, input_size=4)
    layer = d.build_jax(jnp.float32)

    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    assert weights['w_ih'].shape == (8, 4)
    assert weights['w_hh'].shape == (8, 8)
    assert weights['b'].shape == (8,)

    inputs = jax.random.normal(rng, (2, 6, 4), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    assert fwd.shape == (2, 6, 8)

    # step should match forward at each timestep
    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_weight_count_gelu_matches_def():
    """For gelu, JAX param count matches RnnDef.num_weights exactly."""
    d = RnnDef(16, activation='gelu', input_size=8)
    layer = d.build_jax(jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shape():
    layer = RnnDef(8, activation='tanh', input_size=4).build_jax(jnp.float32)
    state = layer.init_state()
    assert state.shape == (8,)
    assert state.dtype == jnp.float32


def test_jax_forward_batch_independent():
    """Samples in a batch don't leak state into each other."""
    d = RnnDef(8, activation='tanh', input_size=4)
    layer = d.build_jax(jnp.float32)
    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)

    inputs = jax.random.normal(rng, (3, 5, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    expected = jnp.concatenate(
        [layer.forward(weights, inputs[i:i + 1]) for i in range(3)], axis=0)
    np.testing.assert_allclose(out, expected, atol=1e-6)
