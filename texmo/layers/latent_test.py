import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.latent import LatentDef, LrnnDef

# -- LatentDef --

def test_latent_def_properties():
    d = LatentDef(16, 4, input_size=8)
    assert d.size == 16
    assert d.reps == 4
    assert str(d) == "latent.16.4"
    assert d.is_valid()
    # Wi(8,16) + bias + Wr(16,16)
    assert d.num_weights == 16 * 8 + 16 + 16 * 16


def test_latent_invalid_size1():
    assert not LatentDef(1, 2, input_size=8).is_valid()


def test_latent_invalid_reps1():
    assert not LatentDef(8, 1, input_size=4).is_valid()


def test_latent_invalid_non_power2():
    assert not LatentDef(13, 4, input_size=8).is_valid()
    assert not LatentDef(8, 3, input_size=8).is_valid()


def test_latent_neighbors():
    d = LatentDef(16, 4, input_size=8)
    neighbors = list(d.neighbors())
    # Size 2x
    assert "latent.8.4" in neighbors
    assert "latent.32.4" in neighbors
    # Reps 2x
    assert "latent.16.2" in neighbors
    assert "latent.16.8" in neighbors
    # Type swap
    assert "lrnn.16.4" in neighbors


def test_latent_neighbors_reps2_collapse():
    d = LatentDef(16, 2, input_size=8)
    neighbors = list(d.neighbors())
    # When reps=2, dense.X.tanh is a neighbor
    assert "dense.16.tanh" in neighbors
    # And we shouldn't drop reps below 2
    assert "latent.16.1" not in neighbors


def test_dense_tanh_has_latent_neighbor():
    from texmo.layers.dense import DenseDef
    d = DenseDef(16, input_size=8, activation="tanh")
    neighbors = list(d.neighbors())
    assert "latent.16.2" in neighbors


def test_dense_relu_no_latent_neighbor():
    from texmo.layers.dense import DenseDef
    d = DenseDef(16, input_size=8, activation="relu")
    neighbors = list(d.neighbors())
    assert "latent.16.2" not in neighbors


# -- LrnnDef --

def test_lrnn_def_properties():
    d = LrnnDef(16, 4, input_size=8)
    assert d.size == 16
    assert d.reps == 4
    assert str(d) == "lrnn.16.4"
    assert d.is_valid()
    # Wi(8,16) + bias + Wh(16,16) + Wr(16,16)
    assert d.num_weights == 16 * 8 + 16 + 16 * 16 + 16 * 16


def test_lrnn_neighbors():
    d = LrnnDef(16, 4, input_size=8)
    neighbors = list(d.neighbors())
    assert "lrnn.8.4" in neighbors
    assert "lrnn.32.4" in neighbors
    assert "lrnn.16.2" in neighbors
    assert "lrnn.16.8" in neighbors
    assert "latent.16.4" in neighbors


def test_lrnn_neighbors_reps2_collapse():
    d = LrnnDef(16, 2, input_size=8)
    neighbors = list(d.neighbors())
    assert "rnn.16.tanh" in neighbors


def test_rnn_tanh_has_lrnn_neighbor():
    from texmo.layers.rnn import RnnDef
    d = RnnDef(16, input_size=8, activation="tanh")
    neighbors = list(d.neighbors())
    assert "lrnn.16.2" in neighbors


def test_rnn_relu_no_lrnn_neighbor():
    from texmo.layers.rnn import RnnDef
    d = RnnDef(16, input_size=8, activation="relu")
    neighbors = list(d.neighbors())
    assert "lrnn.16.2" not in neighbors


# -- JAX Latent --

def test_jax_latent_forward_and_step():
    d = LatentDef(8, 4, input_size=4)
    layer = d.build_jax(jnp.float32)

    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    assert weights['w_i'].shape == (8, 4)
    assert weights['w_r'].shape == (8, 8)
    assert weights['b'].shape == (8,)

    inputs = jax.random.normal(rng, (2, 6, 4), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    assert fwd.shape == (2, 6, 8)

    # Step is stateless; each position is independent of the others.
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, None, inputs[0, t])
        assert state is None
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_latent_is_deterministic():
    """The JAX latent state is always zero-initialized (unlike the
    paper's random init), so repeated forwards on the same inputs
    agree exactly."""
    d = LatentDef(8, 4, input_size=4)
    layer = d.build_jax(jnp.float32)
    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    inputs = jax.random.normal(rng, (1, 3, 4), dtype=jnp.float32)
    np.testing.assert_array_equal(
        layer.forward(weights, inputs), layer.forward(weights, inputs))


def test_jax_latent_weight_count_matches_def():
    d = LatentDef(16, 4, input_size=8)
    layer = d.build_jax(jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


# -- JAX Lrnn --

def test_jax_lrnn_forward_and_step():
    d = LrnnDef(8, 4, input_size=4)
    layer = d.build_jax(jnp.float32)

    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    assert weights['w_i'].shape == (8, 4)
    assert weights['w_h'].shape == (8, 8)
    assert weights['w_r'].shape == (8, 8)
    assert weights['b'].shape == (8,)

    inputs = jax.random.normal(rng, (2, 6, 4), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    assert fwd.shape == (2, 6, 8)

    # step should match forward
    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_lrnn_weight_count_matches_def():
    d = LrnnDef(16, 4, input_size=8)
    layer = d.build_jax(jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_lrnn_init_state_shape():
    layer = LrnnDef(8, 2, input_size=4).build_jax(jnp.float32)
    state = layer.init_state()
    assert state.shape == (8,)
    assert state.dtype == jnp.float32
