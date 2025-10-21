import jax.numpy as jnp
import numpy as np

from .input_bits import InputBits


def test_bits1():
    layer = InputBits.from_spec('bits.1')

    assert layer.ntokens == 2
    assert layer.output_size == 1
    assert layer.tokens_name == 'bits.1'
    assert str(layer) == 'bits.1'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    _, encoding = layer.step(None, state, 0, jnp.float32)
    assert encoding == jnp.array([0])

    _, encoding = layer.step(None, state, 1, jnp.float32)
    assert encoding == jnp.array([1])


def test_bits1_batch():
    layer = InputBits.from_spec('bits.1')

    # batch: 2 len: 4
    input = jnp.array([[0, 1, 0, 1], [0, 1, 1, 0]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0], [0], [1], [0], [1]], [[0], [0], [1], [1], [0]]]))