import jax.numpy as jnp
import numpy as np

from texmo.layers.input_bits import InputBitsDef
from texmo.precision import Precision

# -- bits.1 (raw bit, no position) --

def test_bits1_def():
    d = InputBitsDef.from_spec('bits.1')
    assert d.ntokens == 2
    assert d.size == 1
    assert d.tokens_name == 'bits.1'
    assert d.num_weights == 0
    assert str(d) == 'bits.1'
    assert d.is_valid()


def test_bits1_step():
    layer = InputBitsDef.from_spec('bits.1').build_jax()
    state = layer.init_state()
    assert state is None

    state, out = layer.step(None, state, 0)
    assert out.shape == (1,)
    assert out[0] == 0.0

    state, out = layer.step(None, state, 1)
    assert out[0] == 1.0


def test_bits1_forward():
    layer = InputBitsDef.from_spec('bits.1').build_jax()
    tokens = jnp.array([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 4, 1)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(
        out, [[[0], [1], [0], [1]],
              [[0], [1], [1], [0]]])


def test_bits1_forward_matches_step():
    layer = InputBitsDef.from_spec('bits.1').build_jax()
    tokens = jnp.array([[0, 1, 1, 0]], dtype=jnp.int32)
    fwd = layer.forward(None, tokens)

    state = layer.init_state()
    for t in range(tokens.shape[1]):
        state, step_out = layer.step(None, state, int(tokens[0, t]))
        np.testing.assert_array_equal(fwd[0, t], step_out)


# -- bits.1+bp (raw bit + binary position) --

def test_bits1_bp_def():
    d = InputBitsDef.from_spec('bits.1+bp')
    assert d.ntokens == 2
    assert d.size == 4  # 1 bit + 3 bp bits
    assert str(d) == 'bits.1+bp'
    assert d.is_valid()


def test_bits1_bp_step():
    layer = InputBitsDef.from_spec('bits.1+bp').build_jax()
    state = layer.init_state()
    assert state == 0

    # pos=0: bp encoding of 0 = [0,0,0]
    state, out = layer.step(None, state, 0)
    np.testing.assert_array_equal(out, [0, 0, 0, 0])

    # pos=1: bp encoding of 1 = [1,0,0]
    state, out = layer.step(None, state, 1)
    np.testing.assert_array_equal(out, [1, 1, 0, 0])


def test_bits1_bp_forward():
    layer = InputBitsDef.from_spec('bits.1+bp').build_jax()
    tokens = jnp.array([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 4, 4)
    np.testing.assert_array_equal(
        out,
        [[[0, 0, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 1, 0],
          [1, 1, 1, 0]],
         [[0, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 0, 1, 0],
          [0, 1, 1, 0]]])


def test_bits1_bp_forward_matches_step():
    layer = InputBitsDef.from_spec('bits.1+bp').build_jax()
    tokens = jnp.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=jnp.int32)
    fwd = layer.forward(None, tokens)

    state = layer.init_state()
    for t in range(tokens.shape[1]):
        state, step_out = layer.step(None, state, int(tokens[0, t]))
        np.testing.assert_array_equal(fwd[0, t], step_out)


# -- bits.2 (raw 2-bit, no position) --

def test_bits2_def():
    d = InputBitsDef.from_spec('bits.2')
    assert d.ntokens == 4
    assert d.size == 2
    assert str(d) == 'bits.2'
    assert d.is_valid()


def test_bits2_step():
    layer = InputBitsDef.from_spec('bits.2').build_jax()
    state = layer.init_state()

    _, out = layer.step(None, state, 1)  # binary: 01 → [1, 0]
    np.testing.assert_array_equal(out, [1, 0])

    _, out = layer.step(None, state, 2)  # binary: 10 → [0, 1]
    np.testing.assert_array_equal(out, [0, 1])


def test_bits2_forward():
    layer = InputBitsDef.from_spec('bits.2').build_jax()
    tokens = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 4, 2)
    np.testing.assert_array_equal(
        out,
        [[[0, 0], [1, 0], [0, 1], [1, 1]],
         [[0, 1], [1, 1], [0, 0], [1, 0]]])


# -- bits.2+bp --

def test_bits2_bp_def():
    d = InputBitsDef.from_spec('bits.2+bp')
    assert d.ntokens == 4
    assert d.size == 4  # 2 bits + 2 bp bits
    assert str(d) == 'bits.2+bp'
    assert d.is_valid()


def test_bits2_bp_step():
    layer = InputBitsDef.from_spec('bits.2+bp').build_jax()
    state = layer.init_state()

    # pos=0: bp of 0 = [0,0]; token 2 = [0,1]
    state, out = layer.step(None, state, 2)
    np.testing.assert_array_equal(out, [0, 1, 0, 0])

    # pos=1: bp of 1 = [1,0]; token 3 = [1,1]
    state, out = layer.step(None, state, 3)
    np.testing.assert_array_equal(out, [1, 1, 1, 0])


def test_bits2_bp_forward():
    layer = InputBitsDef.from_spec('bits.2+bp').build_jax()
    tokens = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 4, 4)
    np.testing.assert_array_equal(
        out,
        [[[0, 0, 0, 0],
          [1, 0, 1, 0],
          [0, 1, 0, 1],
          [1, 1, 1, 1]],
         [[0, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 0, 0, 1],
          [1, 0, 1, 1]]])


# -- bits.2.oh (one-hot 2-bit) --

def test_bits2_oh_def():
    d = InputBitsDef.from_spec('bits.2.oh')
    assert d.ntokens == 4
    assert d.size == 4
    assert str(d) == 'bits.2.oh'
    assert d.is_valid()


def test_bits2_oh_step():
    layer = InputBitsDef.from_spec('bits.2.oh').build_jax()
    state = layer.init_state()

    _, out = layer.step(None, state, 2)
    np.testing.assert_array_equal(out, [0, 0, 1, 0])

    _, out = layer.step(None, state, 3)
    np.testing.assert_array_equal(out, [0, 0, 0, 1])


def test_bits2_oh_forward():
    layer = InputBitsDef.from_spec('bits.2.oh').build_jax()
    assert layer.size == 4
    tokens = jnp.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 4, 4)
    # Verify one-hot
    np.testing.assert_array_equal(out.sum(axis=-1), jnp.ones((2, 4)))
    np.testing.assert_array_equal(out[0, 0], [1, 0, 0, 0])
    np.testing.assert_array_equal(out[0, 3], [0, 0, 0, 1])
    assert out[0, 2, 2] == 1.0
    assert out[1, 0, 2] == 1.0


# -- bits.2.oh+bp --

def test_bits2_oh_bp_def():
    d = InputBitsDef.from_spec('bits.2.oh+bp')
    assert d.ntokens == 4
    assert d.size == 6  # 4 one-hot + 2 bp
    assert str(d) == 'bits.2.oh+bp'
    assert d.is_valid()


# -- bits.4 --

def test_bits4_def():
    d = InputBitsDef.from_spec('bits.4')
    assert d.ntokens == 16
    assert d.size == 4
    assert str(d) == 'bits.4'
    assert d.is_valid()


def test_bits4_step():
    layer = InputBitsDef.from_spec('bits.4').build_jax()
    state = layer.init_state()

    # token 2 = binary 0010 → [0, 1, 0, 0]
    _, out = layer.step(None, state, 2)
    np.testing.assert_array_equal(out, [0, 1, 0, 0])

    # token 3 = binary 0011 → [1, 1, 0, 0]
    _, out = layer.step(None, state, 3)
    np.testing.assert_array_equal(out, [1, 1, 0, 0])


# -- bits.4.oh --

def test_bits4_oh_def():
    d = InputBitsDef.from_spec('bits.4.oh')
    assert d.ntokens == 16
    assert d.size == 16
    assert str(d) == 'bits.4.oh'
    assert d.is_valid()


def test_bits4_oh_forward():
    layer = InputBitsDef.from_spec('bits.4.oh').build_jax()
    tokens = jnp.array([[0, 5, 15]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (1, 3, 16)
    assert out[0, 0, 0] == 1.0
    assert out[0, 1, 5] == 1.0
    assert out[0, 2, 15] == 1.0
    np.testing.assert_array_equal(out.sum(axis=-1), jnp.ones((1, 3)))


# -- bits.4+bp --

def test_bits4_bp_def():
    d = InputBitsDef.from_spec('bits.4+bp')
    assert d.ntokens == 16
    assert d.size == 5  # 4 bits + 1 bp bit
    assert str(d) == 'bits.4+bp'
    assert d.is_valid()


def test_bits4_bp_forward_matches_step():
    layer = InputBitsDef.from_spec('bits.4+bp').build_jax()
    tokens = jnp.array([[0, 5]], dtype=jnp.int32)
    fwd = layer.forward(None, tokens)

    state = layer.init_state()
    for t in range(tokens.shape[1]):
        state, step_out = layer.step(None, state, int(tokens[0, t]))
        np.testing.assert_array_equal(fwd[0, t], step_out)


# -- bits.8 --

def test_bits8_def():
    d = InputBitsDef.from_spec('bits.8')
    assert d.ntokens == 256
    assert d.size == 8
    assert str(d) == 'bits.8'
    assert d.is_valid()


# -- bytes alias --

def test_bytes_alias():
    d = InputBitsDef.from_spec('bytes')
    assert d.ntokens == 256
    assert d.size == 256
    assert d.one_hot is True
    assert str(d) == 'bytes'
    assert d.is_valid()


# -- is_valid --

def test_is_valid():
    assert not InputBitsDef(1, True, False).is_valid()   # oh with 1 bit
    assert not InputBitsDef(8, False, True).is_valid()    # bp with 8 bits


# -- dtype --

def test_dtype_fp16():
    layer = InputBitsDef.from_spec(
        'bits.2', precision=Precision.FP16).build_jax()
    _, out = layer.step(None, None, 1)
    assert out.dtype == jnp.float16

    tokens = jnp.array([[0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens)
    assert out.dtype == jnp.float16


def test_dtype_bf16():
    layer = InputBitsDef.from_spec(
        'bits.2', precision=Precision.BF16).build_jax()
    out = layer.forward(None, jnp.array([[0, 1]], dtype=jnp.int32))
    assert out.dtype == jnp.bfloat16


# -- padding --

def test_jax_bits1_bp_forward_shape():
    layer = InputBitsDef.from_spec('bits.1+bp').build_jax()
    assert layer.size == 1 + 3  # 1 bit + 3 bp bits
    tokens = jnp.array([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens)
    assert out.shape == (1, 8, 4)


def test_jax_forward_with_padding():
    layer = InputBitsDef.from_spec('bits.1+bp').build_jax()
    tokens = jnp.array([[0, 1]], dtype=jnp.int32)
    out = layer.forward(None, tokens, padding=2)
    assert out.shape == (1, 4, 4)
    # Padding positions should have 0.5 for the bit value (max entropy)
    assert out[0, 0, 0] == 0.5
    assert out[0, 1, 0] == 0.5
