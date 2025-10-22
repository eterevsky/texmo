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


def test_bits1_neighbors():
    layer = InputBits.from_spec('bits.1')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2', 'bits.1+pos'}


def test_bits1_pos():
    layer = InputBits.from_spec('bits.1+pos')

    assert layer.ntokens == 2
    assert layer.output_size == 9
    assert layer.tokens_name == 'bits.1'
    assert str(layer) == 'bits.1+pos'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 0, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))

    state, encoding = layer.step(None, state, 1, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([1, 0, 1, 0, 0, 0, 0, 0, 0]))


def test_bits1_pos_batch():
    layer = InputBits.from_spec('bits.1+pos')

    # batch: 2 len: 4
    input = jnp.array([[0, 1, 0, 1], [0, 1, 1, 0]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0, 0]],
                   [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0]]]))


def test_bits2():
    layer = InputBits.from_spec('bits.2')

    assert layer.ntokens == 4
    assert layer.output_size == 2
    assert layer.tokens_name == 'bits.2'
    assert str(layer) == 'bits.2'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    _, encoding = layer.step(None, state, 1, jnp.float32)
    np.testing.assert_array_equal(encoding, jnp.array([1, 0]))

    _, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(encoding, jnp.array([0, 1]))


def test_bits2_neighbors():
    layer = InputBits.from_spec('bits.2')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.1', 'bits.4', 'bits.2+pos', 'bits.2.oh'}


def test_bits2_batch():
    layer = InputBits.from_spec('bits.2')

    # batch: 2 len: 4
    input = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1]],
                   [[0, 0],
                    [0, 1],
                    [1, 1],
                    [0, 0],
                    [1, 0]]]))


def test_bits2_pos():
    layer = InputBits.from_spec('bits.2+pos')

    assert layer.ntokens == 4
    assert layer.output_size == 6
    assert layer.tokens_name == 'bits.2'
    assert str(layer) == 'bits.2+pos'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 1, 1, 0, 0, 0]))

    state, encoding = layer.step(None, state, 3, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([1, 1, 0, 1, 0, 0]))

def test_bits2_pos_neighbors():
    layer = InputBits.from_spec('bits.2+pos')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2', 'bits.2.oh+pos', 'bits.1+pos', 'bits.4+pos'}

def test_bits2_pos_batch():
    layer = InputBits.from_spec('bits.2+pos')

    # batch: 2 len: 4
    input = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0],
                    [1, 1, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1]]]))


