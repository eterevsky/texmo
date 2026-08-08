import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.msr import MsrDef, _gammas, _rope_jax_step, _thetas


# -- Helpers --

def _make_jax(dim=2, heads=2, input_size=4, seed=0, dtype=jnp.float32):
    d = MsrDef(dim, heads, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = MsrDef(2, 4, input_size=8)
    assert d.dim == 2
    assert d.heads == 4
    assert d.size == 8
    assert str(d) == "msr.2.4"
    assert d.is_valid()
    # 3 projections, each (heads*dim) x input_size, no bias.
    assert d.num_weights == 3 * 4 * 2 * 8


def test_def_is_valid():
    assert MsrDef(2, 1, input_size=4).is_valid()
    assert MsrDef(4, 2, input_size=4).is_valid()
    assert MsrDef(8, 4, input_size=4).is_valid()
    # dim must be a power of 2 and >= 2 (need at least one pair).
    assert not MsrDef(1, 2, input_size=4).is_valid()
    assert not MsrDef(3, 2, input_size=4).is_valid()
    # heads must be a power of 2.
    assert not MsrDef(2, 3, input_size=4).is_valid()


def test_gammas_match_paper():
    g = _gammas(4)
    assert g[0] == 1 - 2 ** -5
    assert g[1] == 1 - 2 ** -6
    assert g[2] == 1 - 2 ** -7
    assert g[3] == 1 - 2 ** -8


def test_thetas_match_paper():
    # ROPE_BASE=10000, dim=4 -> two pairs at 10000^0=1 and 10000^-0.5.
    t = _thetas(4)
    assert t[0] == pytest.approx(1.0)
    assert t[1] == pytest.approx(0.01)


# -- RoPE properties --

def test_rope_preserves_norm_step():
    """RoPE is unitary: rotation preserves the L2 norm of each pair."""
    thetas = jnp.array(_thetas(4), dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (4,), dtype=jnp.float32)
    for pos in [0, 1, 7, 33]:
        x_rot = _rope_jax_step(x, jnp.float32(pos), thetas)
        # Each pair has the same norm before and after.
        for j in range(0, 4, 2):
            n_in = jnp.linalg.norm(x[j:j + 2])
            n_out = jnp.linalg.norm(x_rot[j:j + 2])
            np.testing.assert_allclose(n_in, n_out, atol=1e-6)


def test_rope_zero_position_is_identity():
    thetas = jnp.array(_thetas(4), dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(1), (4,), dtype=jnp.float32)
    np.testing.assert_allclose(
        _rope_jax_step(x, jnp.float32(0), thetas), x, atol=1e-7)


# -- JAX --

def test_jax_step_shapes():
    _, layer, weights = _make_jax(dim=2, heads=4, input_size=8)
    state = layer.init_state()
    S, pos = state
    assert S.shape == (4, 2, 2)
    assert pos == 0
    x = jax.random.normal(jax.random.PRNGKey(3), (8,), dtype=jnp.float32)
    (new_S, new_pos), out = layer.step(weights, state, x)
    assert new_S.shape == (4, 2, 2)
    assert new_pos == 1
    assert out.shape == (8,)


def test_jax_zero_input_zero_output():
    """With zeros in, the output stays zero."""
    _, layer, weights = _make_jax(dim=2, heads=2, input_size=4)
    out = layer.forward(weights, jnp.zeros((1, 4, 4), dtype=jnp.float32))
    np.testing.assert_allclose(out, jnp.zeros_like(out), atol=1e-7)

def test_jax_weight_count_matches_def():
    d, _, weights = _make_jax(dim=4, heads=2, input_size=8)
    actual = sum(w.size for w in jax.tree.leaves(weights))
    assert actual == d.num_weights


def test_jax_init_state_shapes():
    _, layer, _ = _make_jax(dim=2, heads=4, input_size=4)
    S, pos = layer.init_state()
    assert S.shape == (4, 2, 2)
    assert pos.shape == ()
    assert pos.dtype == jnp.int32


def test_jax_forward_shape():
    _, layer, weights = _make_jax(dim=2, heads=4, input_size=8)
    out = layer.forward(weights, jnp.zeros((2, 5, 8)))
    assert out.shape == (2, 5, 8)


def test_jax_forward_matches_step():
    _, layer, weights = _make_jax(dim=4, heads=2, input_size=8)
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.normal(rng, (2, 6, 8))
    fwd = layer.forward(weights, inputs)
    state = layer.init_state()
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        np.testing.assert_allclose(fwd[0, t], out, atol=1e-5)


def test_jax_causality():
    _, layer, weights = _make_jax(dim=4, heads=2, input_size=4)
    rng = jax.random.PRNGKey(7)
    a = jax.random.normal(rng, (1, 8, 4))
    b = a.at[0, 5:].set(jax.random.normal(jax.random.PRNGKey(8), (3, 4)))
    out_a = layer.forward(weights, a)
    out_b = layer.forward(weights, b)
    # Positions 0..4 must match exactly.
    np.testing.assert_allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
    # Position 5 onwards must differ -- sanity check that the
    # perturbation is actually doing something.
    assert not jnp.allclose(out_a[0, 5:], out_b[0, 5:])


# -- Neighbors --

def test_msr_neighbors_size_and_heads_mutations():
    d = MsrDef(4, 2, input_size=8)
    neighbors = list(d.neighbors())
    # 2x dim
    assert "msr.2.2" in neighbors
    assert "msr.8.2" in neighbors
    # 2x heads
    assert "msr.4.1" in neighbors
    assert "msr.4.4" in neighbors
    # No mgru swap when heads != 1.
    assert not any(n.startswith("mgru.") for n in neighbors)


def test_msr_neighbors_min_dim_clamped():
    """dim must stay >= 2 (RoPE needs at least one pair)."""
    d = MsrDef(2, 4, input_size=8)
    neighbors = list(d.neighbors())
    assert "msr.4.4" in neighbors          # double dim
    assert "msr.1.4" not in neighbors      # halve dim would be 1
    assert "msr.2.2" in neighbors          # halve heads
    assert "msr.2.8" in neighbors          # double heads


def test_msr_neighbors_swap_with_mgru_at_heads_1():
    d = MsrDef(8, 1, input_size=4)
    neighbors = list(d.neighbors())
    assert "mgru.8" in neighbors
    # heads 1 can still double; halve would be 0.5 → drops out via power2_neighbors.
    assert "msr.8.2" in neighbors


def test_mgru_includes_msr_neighbor():
    from texmo.layers.gru import MgruDef
    d = MgruDef(8, input_size=4)
    neighbors = list(d.neighbors())
    assert "msr.8.1" in neighbors
