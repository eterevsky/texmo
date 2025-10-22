import jax.numpy as jnp
import numpy as np
from rich import print as pprint

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
    assert neighbors == {'bits.2', 'bits.1+pos', 'bits.1+bp'}


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

def test_bits1_bp():
    layer = InputBits.from_spec('bits.1+bp')

    assert layer.ntokens == 2
    assert layer.output_size == 4
    assert layer.tokens_name == 'bits.1'
    assert str(layer) == 'bits.1+bp'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 0, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 0, 0, 0]))

    state, encoding = layer.step(None, state, 1, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([1, 1, 0, 0]))


def test_bits1_bp_batch():
    layer = InputBits.from_spec('bits.1+bp')

    input = jnp.array([[0, 1, 0, 1], [0, 1, 1, 0]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 1, 1, 0]],
                   [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 1, 0]]]))


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
    assert neighbors == {'bits.1', 'bits.4', 'bits.2+pos', 'bits.2+bp', 'bits.2.oh'}


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
    assert neighbors == {'bits.2', 'bits.2.oh+pos', 'bits.1+pos', 'bits.4+pos', 'bits.2+bp'}

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


def test_bits2_bp():
    layer = InputBits.from_spec('bits.2+bp')

    assert layer.ntokens == 4
    assert layer.output_size == 4
    assert layer.tokens_name == 'bits.2'
    assert str(layer) == 'bits.2+bp'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 1, 0, 0]))

    state, encoding = layer.step(None, state, 3, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([1, 1, 1, 0]))

def test_bits2_bp_neighbors():
    layer = InputBits.from_spec('bits.2+bp')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2', 'bits.2.oh+bp', 'bits.1+bp', 'bits.4+bp', 'bits.2+pos'}

def test_bits2_bp_batch():
    layer = InputBits.from_spec('bits.2+bp')

    # batch: 2 len: 4
    input = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 1, 1, 1]],
                   [[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 1, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 1, 1]]]))


def test_bits2_oh_neighbors():
    layer = InputBits.from_spec('bits.2.oh')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2', 'bits.2.oh+pos', 'bits.2.oh+bp', 'bits.4.oh'}


def test_bits2_oh():
    layer = InputBits.from_spec('bits.2.oh')

    assert layer.ntokens == 4
    assert layer.output_size == 4
    assert layer.tokens_name == 'bits.2'
    assert str(layer) == 'bits.2.oh'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 0, 1, 0]))

    state, encoding = layer.step(None, state, 3, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 0, 0, 1]))

def test_bits2_oh_batch():
    layer = InputBits.from_spec('bits.2.oh')

    input = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]],
                   [[0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]]]))


def test_bits2_oh_pos_neighbors():
    layer = InputBits.from_spec('bits.2.oh+pos')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2.oh', 'bits.2.oh+bp', 'bits.2+pos', 'bits.4.oh+pos'}


def test_bits2_oh_pos():
    layer = InputBits.from_spec('bits.2.oh+pos')

    assert layer.ntokens == 4
    assert layer.output_size == 8
    assert layer.tokens_name == 'bits.2'
    assert str(layer) == 'bits.2.oh+pos'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 0, 1, 0, 1, 0, 0, 0]))

    state, encoding = layer.step(None, state, 3, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 0, 0, 1, 0, 1, 0, 0]))

def test_bits2_oh_pos_batch():
    layer = InputBits.from_spec('bits.2.oh+pos')

    input = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]])

    output = layer.forward_batch(None, input, 1, dtype=jnp.float32)

    np.testing.assert_array_equal(
        output,
        jnp.array([[[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1]]]))


def test_bits4_neighbors():
    layer = InputBits.from_spec('bits.4')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.2', 'bits.8', 'bits.4+pos', 'bits.4+bp', 'bits.4.oh'}


def test_bits4():
    layer = InputBits.from_spec('bits.4')

    assert layer.ntokens == 16
    assert layer.output_size == 4
    assert layer.tokens_name == 'bits.4'
    assert str(layer) == 'bits.4'
    assert layer.is_valid()
    state = layer.init_state(None, jnp.float32)

    state, encoding = layer.step(None, state, 2, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([0, 1, 0, 0]))

    state, encoding = layer.step(None, state, 3, jnp.float32)
    np.testing.assert_array_equal(
        encoding, jnp.array([1, 1, 0, 0]))


def test_bits8_neighbors():
    layer = InputBits.from_spec('bits.8')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.4', 'bytes'}


def test_bits8():
    layer = InputBits.from_spec('bits.8')

    assert layer.ntokens == 256
    assert layer.output_size == 8
    assert layer.tokens_name == 'bits.8'
    assert str(layer) == 'bits.8'
    assert layer.is_valid()


def test_bytes_neighbors():
    layer = InputBits.from_spec('bytes')
    neighbors = {str(n) for n in layer.neighbors()}
    assert neighbors == {'bits.8', 'bits.4.oh'}


def test_bytes():
    layer = InputBits.from_spec('bytes')

    assert layer.ntokens == 256
    assert layer.output_size == 256
    assert layer.tokens_name == 'bits.8'
    assert str(layer) == 'bytes'
    assert layer.is_valid()
