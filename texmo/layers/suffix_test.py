import jax
import jax.numpy as jnp
import numpy as np

from texmo.layers.suffix import SuffixDef
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def test_neighbors():
    d = SuffixDef(4, input_size=8)
    neighbors = list(d.neighbors())
    # Size mutations + the conv.L type-swap + the single-head attn
    # swap over the same span, sized to the suffix's input width.
    assert neighbors == ["suffix.2", "suffix.8", "conv.4", "attn.8.1.4"]


def test_neighbors_min():
    d = SuffixDef(2, input_size=4)
    neighbors = list(d.neighbors())
    # suffix.1 is invalid but still generated (Model2Def filters);
    # conv.2 is the same-L type swap, attn.4.1.2 the soft-wrap swap.
    assert neighbors == ["suffix.1", "suffix.4", "conv.2", "attn.4.1.2"]


def test_def_properties():
    d = SuffixDef(4, input_size=8)
    assert d.length == 4
    assert d.size == 32
    assert d.num_weights == 0
    assert str(d) == 'suffix.4'
    assert d.is_valid()


def test_def_is_valid():
    assert SuffixDef(2, input_size=4).is_valid()
    assert SuffixDef(4, input_size=4).is_valid()
    assert not SuffixDef(1, input_size=4).is_valid()  # must be > 1
    assert not SuffixDef(3, input_size=4).is_valid()  # must be power of 2


# -- JAX --

def test_jax_init_state():
    layer = SuffixDef(2, input_size=4).build_jax(jnp.float32)
    state = layer.init_state()
    assert state.shape == (1, 4)
    assert state.sum() == 0


def test_jax_step_known_values():
    layer = SuffixDef(2, input_size=4).build_jax(jnp.float32)
    state = layer.init_state()

    v1 = jnp.array([1, 2, 3, 4], dtype=jnp.float32)
    state, out = layer.step(None, state, v1)
    # Window: [zeros, v1] → flattened
    assert out.shape == (8,)
    np.testing.assert_array_equal(out, [0, 0, 0, 0, 1, 2, 3, 4])

    v2 = jnp.array([5, 6, 7, 8], dtype=jnp.float32)
    state, out = layer.step(None, state, v2)
    # Window: [v1, v2]
    np.testing.assert_array_equal(out, [1, 2, 3, 4, 5, 6, 7, 8])

    v3 = jnp.array([9, 10, 11, 12], dtype=jnp.float32)
    state, out = layer.step(None, state, v3)
    # Window: [v2, v3]
    np.testing.assert_array_equal(out, [5, 6, 7, 8, 9, 10, 11, 12])


def test_jax_forward_batch():
    layer = SuffixDef(2, input_size=4).build_jax(jnp.float32)
    inputs = jnp.array([[[0, 0, 0, 0],
                         [1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]],
                        [[0, 0, 0, 0],
                         [10, 20, 30, 40],
                         [50, 60, 70, 80],
                         [90, 100, 110, 120]]], dtype=jnp.float32)
    out = layer.forward(None, inputs)
    assert out.shape == (2, 3, 8)
    np.testing.assert_array_equal(
        out[1, 0], [0, 0, 0, 0, 10, 20, 30, 40])


def test_jax_forward():
    layer = SuffixDef(2, input_size=4).build_jax(jnp.float32)
    inputs = jnp.array([[[0, 0, 0, 0],
                         [1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]]], dtype=jnp.float32)
    out = layer.forward(None, inputs)
    assert out.shape == (1, 3, 8)

    expected = jnp.array([[[0, 0, 0, 0, 1, 2, 3, 4],
                           [1, 2, 3, 4, 5, 6, 7, 8],
                           [5, 6, 7, 8, 9, 10, 11, 12]]], dtype=jnp.float32)
    np.testing.assert_array_equal(out, expected)


def test_jax_step_matches_forward():
    layer = SuffixDef(2, input_size=4).build_jax(jnp.float32)
    inputs = jnp.array([[[0, 0, 0, 0],
                         [1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12]]], dtype=jnp.float32)
    fwd = layer.forward(None, inputs)

    state = layer.init_state()
    step_outputs = []
    for t in range(inputs.shape[1]):
        state, out = layer.step(None, state, inputs[0, t])
        step_outputs.append(out)

    # forward skips the first position (consumed by padding)
    for i in range(fwd.shape[1]):
        np.testing.assert_allclose(fwd[0, i], step_outputs[i + 1])


def test_jax_suffix4_forward():
    layer = SuffixDef(4, input_size=2).build_jax(jnp.float32)
    inputs = jnp.arange(14, dtype=jnp.float32).reshape(1, 7, 2)
    out = layer.forward(None, inputs)
    assert out.shape == (1, 4, 8)
    np.testing.assert_array_equal(out[0, 0], inputs[0, 0:4].reshape(-1))
    np.testing.assert_array_equal(out[0, 3], inputs[0, 3:7].reshape(-1))


def test_jax_model_with_suffix():
    md = parse_model2("bits.1+bp|suffix.4-dense.4.gelu", Precision.FP32)
    assert md.total_padding == 4  # 1 base + (4-1)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))

    tokens = jnp.array([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=jnp.int32)
    logits = model.forward(weights, tokens)
    assert logits.shape == (1, 8, 2)


def test_jax_model_suffix_step_matches_forward():
    md = parse_model2("bits.1+bp|suffix.2-dense.8.tanh", Precision.FP32)
    assert md.total_padding == 2  # 1 base + (2-1)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))

    token_list = [0, 1, 1, 0, 1, 0]
    tokens = jnp.array([token_list], dtype=jnp.int32)
    fwd_logits = model.forward(weights, tokens)

    states, logits0 = model.initial_step(weights)
    step_logits = [logits0]
    for t in token_list[:-1]:
        states, logits_t = model.step(weights, states, t)
        step_logits.append(logits_t)

    for i in range(len(token_list)):
        np.testing.assert_allclose(
            fwd_logits[0, i], step_logits[i], atol=1e-5)
