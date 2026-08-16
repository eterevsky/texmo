import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.dense import DenseDef


def _fixed_weights():
    """Hand-picked weights for the known-value tests below."""
    return {
        'w': jnp.array([[2, 0, 1, 0], [0, 1, 0, 2]], dtype=jnp.float32),
        'b': jnp.array([10, 20], dtype=jnp.float32),
    }


def test_num_weights():
    layer_def = DenseDef(8, activation="gelu", input_size=4)
    assert layer_def.num_weights == 8 * 4 + 8


def test_unknown_activation_rejected():
    # `elu` is genuinely absent from _JAX_ACTIVATIONS (silu was the
    # example here until it was implemented, 2026-08).
    with pytest.raises(KeyError):
        DenseDef(8, activation="elu", input_size=4)


def test_is_valid_activation_whitelist():
    for act in ("tanh", "gelu", "silu"):
        assert DenseDef(8, activation=act, input_size=4).is_valid(), act
    # relu parses and runs but is retired from the search.
    assert not DenseDef(8, activation="relu", input_size=4).is_valid()
    # The bare form is valid only where a split branch opts in.
    assert not DenseDef(8, input_size=4).is_valid()
    assert DenseDef(8, input_size=4).is_valid(allow_bare=True)


def test_neighbors_activation_cycle_is_complete():
    # Every search-valid activation is one mutation from every other,
    # so no activation is a dead end. The bare form joins the cycle for
    # dense (it is the gate/linear path); size stays fixed throughout.
    acts = ("tanh", "gelu", "silu")
    for act in acts:
        nbs = list(DenseDef(8, activation=act, input_size=4).neighbors())
        for other in acts:
            if other == act:
                continue
            assert f"dense.8.{other}" in nbs, (act, other)
        assert "dense.8" in nbs                      # bare
        assert f"dense.8.{act}" not in nbs           # no self-edge
    # ...and the bare form reaches all three back.
    bare = list(DenseDef(8, input_size=4).neighbors())
    for act in acts:
        assert f"dense.8.{act}" in bare, act


def test_neighbors():
    d = DenseDef(32, activation="relu", input_size=16)
    neighbors = list(d.neighbors())
    assert "dense.16.relu" in neighbors
    assert "dense.64.relu" in neighbors
    assert "rnn.32.relu" in neighbors


def test_neighbors_size1():
    d = DenseDef(1, activation="tanh", input_size=8)
    neighbors = list(d.neighbors())
    assert "dense.2.tanh" in neighbors
    assert "rnn.1.tanh" in neighbors


# -- JAX --

def test_jax_step_known_values():
    layer = DenseDef(2, input_size=4).build_jax(jnp.float32)
    state, out = layer.step(
        _fixed_weights(), None, jnp.array([1, 2, 3, 4], dtype=jnp.float32))
    assert state is None
    # [2*1 + 1*3 + 10, 1*2 + 2*4 + 20]
    np.testing.assert_allclose(out, [15, 30], atol=1e-6)


def test_jax_forward_known_values():
    layer = DenseDef(2, input_size=4).build_jax(jnp.float32)
    inputs = jnp.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=jnp.float32)
    out = layer.forward(_fixed_weights(), inputs)
    assert out.shape == (1, 2, 2)
    np.testing.assert_allclose(out[0], [[15, 30], [27, 42]], atol=1e-6)


def test_jax_activation_relu():
    layer = DenseDef(2, activation="relu", input_size=4).build_jax(jnp.float32)
    weights = {
        'w': jnp.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32),
        'b': jnp.array([-5, -5], dtype=jnp.float32),
    }
    out = layer.forward(
        weights, jnp.array([[[3, 0, 0, 10]]], dtype=jnp.float32))
    # relu(-5 + 3) = 0, relu(-5 + 10) = 5
    np.testing.assert_allclose(out, [[[0, 5]]], atol=1e-6)


def test_jax_activation_silu():
    layer = DenseDef(2, activation="silu", input_size=4).build_jax(jnp.float32)
    weights = {
        'w': jnp.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32),
        'b': jnp.array([-5, -5], dtype=jnp.float32),
    }
    out = layer.forward(
        weights, jnp.array([[[3, 0, 0, 10]]], dtype=jnp.float32))
    # silu(x) = x * sigmoid(x); pre-activations are -2 and 5.
    expect = [x / (1.0 + np.exp(-x)) for x in (-2.0, 5.0)]
    np.testing.assert_allclose(out, [[expect]], atol=1e-6)


def test_jax_bare_dense_is_linear():
    """A bare dense (no activation) applies no nonlinearity."""
    layer = DenseDef(2, input_size=4).build_jax(jnp.float32)
    weights = {
        'w': jnp.array([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32),
        'b': jnp.array([-5, -5], dtype=jnp.float32),
    }
    out = layer.forward(
        weights, jnp.array([[[3, 0, 0, 10]]], dtype=jnp.float32))
    np.testing.assert_allclose(out, [[[-2, 5]]], atol=1e-6)


def test_jax_forward_batch():
    """Batched forward equals the per-sample forwards stacked."""
    layer = DenseDef(2, input_size=4).build_jax(jnp.float32)
    weights = _fixed_weights()
    inputs = jnp.array([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
    ], dtype=jnp.float32)

    out = layer.forward(weights, inputs)
    assert out.shape == (2, 2, 2)

    out0 = layer.forward(weights, inputs[0:1])
    out1 = layer.forward(weights, inputs[1:2])
    np.testing.assert_allclose(
        out, jnp.concatenate([out0, out1], axis=0), atol=1e-6)


def test_jax_forward_and_step():
    """Build DenseJax from DenseDef, run forward and step."""
    layer_def = DenseDef(size=16, input_size=8, activation="gelu")
    layer = layer_def.build_jax(jnp.float32)

    rng = jax.random.PRNGKey(0)
    weights = layer.init_weights(rng)
    assert weights['w'].shape == (16, 8)
    assert weights['b'].shape == (16,)

    inputs = jax.random.normal(rng, (4, 32, 8))
    outputs = layer.forward(weights, inputs)
    assert outputs.shape == (4, 32, 16)

    # step should match forward at each position (step is unbatched)
    x = inputs[0, 0]
    state, out = layer.step(weights, None, x)
    assert state is None
    assert jnp.allclose(out, outputs[0, 0], atol=1e-6)


def test_jax_weight_count():
    """Weight count from DenseDef matches actual init_weights."""
    layer_def = DenseDef(size=32, input_size=16, activation="relu")
    layer = layer_def.build_jax(jnp.float32)
    weights = layer.init_weights(jax.random.PRNGKey(0))
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == layer_def.num_weights
