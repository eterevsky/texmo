import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.lmgu import LmguDef


def _make_jax(size=4, reps=2, input_size=4, seed=0, dtype=jnp.float32):
    d = LmguDef(size, reps, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = LmguDef(4, 2, input_size=8)
    assert d.size == 4
    assert d.reps == 2
    assert str(d) == "lmgu.4.2"
    assert d.is_valid()
    # 2*4*8 + 4*4*4 + 2*4 = 64 + 64 + 8 = 136
    assert d.num_weights == 136


def test_def_is_valid():
    assert LmguDef(2, 2, input_size=4).is_valid()
    assert LmguDef(4, 8, input_size=4).is_valid()
    # size must be a power of 2 and > 1.
    assert not LmguDef(1, 2, input_size=4).is_valid()
    assert not LmguDef(3, 2, input_size=4).is_valid()
    # reps must be a power of 2 and >= 2.
    assert not LmguDef(4, 1, input_size=4).is_valid()
    assert not LmguDef(4, 3, input_size=4).is_valid()


def test_num_weights_formula():
    """6 weight matrices + 2 biases."""
    size, input_size = 8, 6
    d = LmguDef(size, 4, input_size=input_size)
    # 2*size*input_size (w_h_i + w_f_i)
    # + 4*size*size (w_h_h, w_h_s, w_f_h, w_f_s)
    # + 2*size (b_h, b_f)
    expected = 2 * size * input_size + 4 * size * size + 2 * size
    assert d.num_weights == expected


def test_num_mults_scales_with_reps():
    """num_mults = num_weights * reps (lrnn convention)."""
    d2 = LmguDef(4, 2, input_size=4)
    d4 = LmguDef(4, 4, input_size=4)
    assert d4.num_mults == 2 * d2.num_mults


# -- JAX --

def test_jax_weight_count_matches_def():
    d, _, weights = _make_jax(size=4, reps=2, input_size=8)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shape():
    _, layer, _ = _make_jax(size=4, reps=2, input_size=4)
    state = layer.init_state()
    assert state.shape == (4,)
    assert state.sum() == 0


def test_jax_step_shape():
    _, layer, weights = _make_jax(size=4, reps=2, input_size=8)
    state = layer.init_state()
    new_state, out = layer.step(
        weights, state, jnp.ones(8, dtype=jnp.float32))
    assert new_state.shape == (4,)
    assert out.shape == (4,)


def test_jax_forward_shape():
    _, layer, weights = _make_jax(size=4, reps=2, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (2, 5, 8), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (2, 5, 4)


def test_jax_forward_matches_step():
    """Hoisted-projection scan and per-token step must agree."""
    _, layer, weights = _make_jax(size=4, reps=4, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(42), (2, 6, 8), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_causality():
    """Perturbing token t+1 must not change output at position t."""
    _, layer, weights = _make_jax(size=4, reps=2, input_size=4)
    a = jax.random.normal(
        jax.random.PRNGKey(7), (1, 8, 4), dtype=jnp.float32)
    b = a.at[0, 5:].set(jax.random.normal(
        jax.random.PRNGKey(8), (3, 4), dtype=jnp.float32))
    out_a = layer.forward(weights, a)
    out_b = layer.forward(weights, b)
    np.testing.assert_allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
    assert not np.allclose(
        np.asarray(out_a[0, 5:]), np.asarray(out_b[0, 5:]))


def test_jax_higher_reps_runs_cleanly():
    """Iteration converges (or at least doesn't NaN) at higher reps."""
    _, layer, weights = _make_jax(size=4, reps=8, input_size=4)
    inputs = jax.random.normal(
        jax.random.PRNGKey(13), (1, 32, 4), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert jnp.all(jnp.isfinite(out))


# -- Neighbors --

def test_lmgu_neighbors_size_and_reps_mutations():
    d = LmguDef(4, 4, input_size=8)
    neighbors = list(d.neighbors())
    # 2x size (size > 1 constraint already met).
    assert "lmgu.2.4" in neighbors
    assert "lmgu.8.4" in neighbors
    # 2x reps (reps >= 2 constraint).
    assert "lmgu.4.2" in neighbors
    assert "lmgu.4.8" in neighbors
    # Family swap with lrnn at same (S, R).
    assert "lrnn.4.4" in neighbors


def test_lmgu_neighbors_size_clamped_at_2():
    """size > 1, so halving from size=2 doesn't fire."""
    d = LmguDef(2, 4, input_size=8)
    neighbors = list(d.neighbors())
    assert "lmgu.4.4" in neighbors
    assert "lmgu.1.4" not in neighbors


def test_lmgu_neighbors_reps_2_collapses_to_mgru():
    """At reps=2, lmgu swaps with mgru.X (the non-iterating cousin)."""
    d = LmguDef(4, 2, input_size=8)
    neighbors = list(d.neighbors())
    assert "mgru.4" in neighbors
    assert "lrnn.4.2" in neighbors
    # No mgru swap at reps != 2.
    d4 = LmguDef(4, 4, input_size=8)
    n4 = list(d4.neighbors())
    assert "mgru.4" not in n4
