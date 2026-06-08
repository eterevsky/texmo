import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.conv import ConvDef


def _make_jax(kernel=4, input_size=8, seed=0, dtype=jnp.float32):
    d = ConvDef(kernel, input_size=input_size)
    layer = d.build_jax(dtype)
    weights = layer.init_weights(jax.random.PRNGKey(seed))
    return d, layer, weights


# -- Def --

def test_def_properties():
    d = ConvDef(4, input_size=8)
    assert d.kernel == 4
    assert d.size == 8           # depthwise preserves channels
    assert d.length == 4         # suffix-like: reads 4 past positions
    assert str(d) == "conv.4"
    assert d.is_valid()
    # 8 channels x (4 kernel taps + 1 bias) = 40
    assert d.num_weights == 40


def test_def_is_valid():
    assert ConvDef(2, input_size=4).is_valid()
    assert ConvDef(4, input_size=4).is_valid()
    assert ConvDef(16, input_size=4).is_valid()
    # L must be >= 2 and power of 2.
    assert not ConvDef(1, input_size=4).is_valid()
    assert not ConvDef(3, input_size=4).is_valid()
    assert not ConvDef(6, input_size=4).is_valid()


def test_def_size_matches_input():
    """Depthwise conv preserves channel count regardless of input_size."""
    for d in (1, 2, 4, 16, 256):
        assert ConvDef(4, input_size=d).size == d


# -- JAX --

def test_jax_weight_shapes():
    d, _, weights = _make_jax(kernel=4, input_size=8)
    assert weights['w'].shape == (4, 8)
    assert weights['b'].shape == (8,)
    total = sum(w.size for w in jax.tree.leaves(weights))
    assert total == d.num_weights


def test_jax_init_state_shape():
    _, layer, _ = _make_jax(kernel=4, input_size=6)
    state = layer.init_state()
    assert state.shape == (3, 6)
    assert state.sum() == 0


def test_jax_forward_shape_drops_l_minus_1_positions():
    """Valid-conv convention: output sequence is shorter by L-1."""
    _, layer, weights = _make_jax(kernel=4, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1), (2, 7, 8), dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    assert out.shape == (2, 4, 8)  # 7 - 4 + 1


def test_jax_forward_matches_step():
    """Per-token `step` from a zero state, after L-1 warm-up steps,
    must agree with `forward` -- which doesn't emit the L-1 transients.

    For a sequence of length T input, forward emits T - L + 1 outputs
    aligned to input positions L-1 ... T-1.
    """
    L = 4
    _, layer, weights = _make_jax(kernel=L, input_size=8)
    inputs = jax.random.normal(
        jax.random.PRNGKey(42), (1, 8, 8), dtype=jnp.float32)
    fwd = layer.forward(weights, inputs)
    state = layer.init_state()
    # Drive step through every input position; the framework's
    # warm-up would have done the first L-1 of these on padding
    # tokens, so the first non-transient step output appears at
    # input position L-1 -- matching fwd[0, 0].
    step_outs = []
    for t in range(inputs.shape[1]):
        state, out = layer.step(weights, state, inputs[0, t])
        step_outs.append(out)
    for t in range(fwd.shape[1]):
        np.testing.assert_allclose(
            np.asarray(fwd[0, t]),
            np.asarray(step_outs[t + L - 1]),
            atol=1e-5)


def test_jax_causality():
    """Perturbing input position p must not change output positions
    whose receptive field doesn't reach p."""
    L = 4
    _, layer, weights = _make_jax(kernel=L, input_size=4)
    a = jax.random.normal(
        jax.random.PRNGKey(7), (1, 8, 4), dtype=jnp.float32)
    # Perturb input positions [5..8). Output position t corresponds
    # to input positions [t .. t+L-1]; perturbations at input >= 5
    # don't reach output t whose t+L-1 < 5, i.e. t < 5 - (L-1) = 2.
    b = a.at[0, 5:].set(jax.random.normal(
        jax.random.PRNGKey(8), (3, 4), dtype=jnp.float32))
    out_a = layer.forward(weights, a)
    out_b = layer.forward(weights, b)
    np.testing.assert_allclose(
        np.asarray(out_a[0, :2]), np.asarray(out_b[0, :2]), atol=1e-6)
    # Sanity: the perturbation does change at least one later output.
    assert not np.allclose(
        np.asarray(out_a[0, 2:]), np.asarray(out_b[0, 2:]))


def test_jax_forward_manual_conv_kernel_2():
    """Hand-computed expected outputs for a fixed kernel=2 setup."""
    # Skip random init -- set weights explicitly.
    layer = ConvDef(2, input_size=2).build_jax(jnp.float32)
    weights = {
        # w[0, c] applies to x[t], w[1, c] applies to x[t-1].
        # Channel 0: y = 2*x[t] + 3*x[t-1] (no bias).
        # Channel 1: y = -1*x[t] + 0.5*x[t-1] (bias 0.1).
        'w': jnp.array([[2.0, -1.0],
                        [3.0, 0.5]], dtype=jnp.float32),
        'b': jnp.array([0.0, 0.1], dtype=jnp.float32),
    }
    # inputs: 4 positions, valid-conv output is 4 - 2 + 1 = 3 positions
    # aligned to input positions 1, 2, 3.
    # t=0 -> [1, 4], t=1 -> [2, 5], t=2 -> [3, 6], t=3 -> [4, 7]
    inputs = jnp.array([[[1.0, 4.0], [2.0, 5.0],
                         [3.0, 6.0], [4.0, 7.0]]],
                       dtype=jnp.float32)
    out = layer.forward(weights, inputs)
    # Output index 0 <-> input index 1: y = w0*x[1] + w1*x[0]
    #   Channel 0: 2*2 + 3*1 = 7
    #   Channel 1: -1*5 + 0.5*4 + 0.1 = -2.9
    # Output index 1 <-> input index 2:
    #   Channel 0: 2*3 + 3*2 = 12
    #   Channel 1: -1*6 + 0.5*5 + 0.1 = -3.4
    # Output index 2 <-> input index 3:
    #   Channel 0: 2*4 + 3*3 = 17
    #   Channel 1: -1*7 + 0.5*6 + 0.1 = -3.9
    expected = jnp.array([[[7.0, -2.9],
                           [12.0, -3.4],
                           [17.0, -3.9]]], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(expected), atol=1e-5)


# -- Neighbors --

def test_neighbors_kernel_mutations():
    d = ConvDef(4, input_size=8)
    neighbors = list(d.neighbors())
    assert "conv.2" in neighbors
    assert "conv.8" in neighbors


def test_neighbors_swap_to_suffix():
    d = ConvDef(4, input_size=8)
    neighbors = list(d.neighbors())
    assert "suffix.4" in neighbors


def test_suffix_swap_to_conv():
    from texmo.layers.suffix import SuffixDef
    d = SuffixDef(4, input_size=8)
    neighbors = list(d.neighbors())
    assert "conv.4" in neighbors


def test_neighbors_min_kernel_filtered_by_model():
    # The Def yields conv.1 too (size mutation hits L=1) but ModelDef
    # filters it via `is_valid()`. Bare Def output includes the
    # invalid one -- that's how the framework's pattern works for
    # other layers (e.g. SuffixDef(2).neighbors() includes "suffix.1").
    d = ConvDef(2, input_size=4)
    neighbors = list(d.neighbors())
    assert "conv.1" in neighbors
    assert "conv.4" in neighbors
