import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from texmo.layers.msr import MsrDef, _gammas, _thetas


# -- Helpers --

def _make_torch(dim=2, heads=2, input_size=4, seed=0):
    torch.manual_seed(seed)
    return MsrDef(dim, heads, input_size=input_size).build_module()


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


# -- Torch step/forward --

def test_step_shapes():
    m = _make_torch(dim=2, heads=4, input_size=8)
    state = m.init_state()
    S, pos = state
    assert S.shape == (4, 2, 2)
    assert pos == 0
    new_state, out = m.step(state, torch.randn(8))
    new_S, new_pos = new_state
    assert new_S.shape == (4, 2, 2)
    assert new_pos == 1
    assert out.shape == (8,)


def test_forward_shape():
    m = _make_torch(dim=4, heads=2, input_size=8)
    out = m(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 8)


def test_forward_matches_step():
    """Recurrent and parallel forms must produce identical outputs."""
    m = _make_torch(dim=4, heads=2, input_size=8)
    m.eval()
    inputs = torch.randn(1, 6, 8)
    with torch.no_grad():
        fwd = m(inputs)
        state = m.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = m.step(state, inputs[0, t])
            assert torch.allclose(fwd[0, t], step_out, atol=1e-5)


def test_causality():
    """Perturbing token t+1 must not change output at position t."""
    m = _make_torch(dim=4, heads=2, input_size=4)
    m.eval()
    a = torch.randn(1, 8, 4)
    b = a.clone()
    b[0, 5:] = torch.randn(3, 4)  # only change positions 5..7
    with torch.no_grad():
        out_a = m(a)
        out_b = m(b)
        # Positions 0..4 must match exactly.
        assert torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
        # Position 5 onwards should differ (sanity check the perturbation
        # is actually doing something).
        assert not torch.allclose(out_a[0, 5:], out_b[0, 5:])


def test_zero_input_zero_state_zero_output():
    """With zeros in, output and new state stay zero."""
    m = _make_torch(dim=2, heads=2, input_size=4)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 4, 4))
        assert torch.allclose(out, torch.zeros_like(out))


def test_state_dict_roundtrip():
    d = MsrDef(2, 2, input_size=4)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())
    inputs = torch.randn(1, 3, 4)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


def test_param_count_matches_def():
    d = MsrDef(4, 2, input_size=6)
    m = d.build_module()
    # gammas/thetas are non-persistent buffers, not parameters.
    assert sum(p.numel() for p in m.parameters()) == d.num_weights


# -- RoPE properties --

def test_rope_preserves_norm_step():
    """RoPE is unitary: rotation preserves the L2 norm of each pair."""
    from texmo.layers.msr import _rope_torch_step
    thetas = torch.tensor(_thetas(4), dtype=torch.float32)
    x = torch.randn(4)
    for pos in [0, 1, 7, 33]:
        x_rot = _rope_torch_step(x, pos, thetas)
        # Each pair has the same norm before and after.
        for j in range(0, 4, 2):
            n_in = torch.linalg.norm(x[j:j+2])
            n_out = torch.linalg.norm(x_rot[j:j+2])
            assert torch.allclose(n_in, n_out, atol=1e-6)


def test_rope_zero_position_is_identity():
    from texmo.layers.msr import _rope_torch_step
    thetas = torch.tensor(_thetas(4), dtype=torch.float32)
    x = torch.randn(4)
    assert torch.allclose(_rope_torch_step(x, 0, thetas), x, atol=1e-7)


# -- JAX --

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
    np.testing.assert_allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)


# -- Cross-backend agreement --

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


def test_torch_matches_jax():
    """Torch and JAX implementations must agree on shared weights."""
    dim, heads, input_size = 4, 2, 8
    seed = 13

    d = MsrDef(dim, heads, input_size=input_size)
    j_layer = d.build_jax(jnp.float32)
    j_weights = j_layer.init_weights(jax.random.PRNGKey(seed))

    t_module = d.build_module()
    # Copy JAX weights into the Torch module. Torch's nn.Linear stores
    # weight as (out, in) — same shape as our stacked JAX array.
    with torch.no_grad():
        t_module.w_qkv.weight.copy_(
            torch.tensor(np.asarray(j_weights['w_qkv'])))
    t_module.eval()

    rng = jax.random.PRNGKey(seed + 1)
    inputs_j = jax.random.normal(rng, (1, 7, input_size), dtype=jnp.float32)
    inputs_t = torch.tensor(np.asarray(inputs_j))

    j_out = j_layer.forward(j_weights, inputs_j)
    with torch.no_grad():
        t_out = t_module(inputs_t)

    np.testing.assert_allclose(
        np.asarray(j_out), t_out.numpy(), atol=1e-5)
