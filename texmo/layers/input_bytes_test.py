import jax.numpy as jnp
import numpy as np

from texmo.layers.input_bytes import InputBytesDef
from texmo.precision import Precision


def test_def_properties():
    d = InputBytesDef()
    assert d.ntokens == 256
    assert d.size == 256
    assert d.num_weights == 0
    assert str(d) == 'bytes'


def test_neighbors():
    assert InputBytesDef().neighbors() == ('bits.4.oh+bp',)


def test_build_jax_is_bits8_one_hot():
    """`bytes` is the bits.8.oh encoding under another name."""
    layer = InputBytesDef().build_jax()
    assert layer.ntokens == 256
    assert layer.size == 256
    assert layer.one_hot is True
    assert layer.bp is False


def test_step():
    layer = InputBytesDef().build_jax()
    state, out = layer.step(None, None, 65)  # ASCII 'A'
    assert state is None
    assert out.shape == (256,)
    assert out[65] == 1.0
    assert out.sum() == 1.0
    assert out.dtype == jnp.float32


def test_step_zero():
    layer = InputBytesDef().build_jax()
    _, out = layer.step(None, None, 0)
    assert out[0] == 1.0
    assert out.sum() == 1.0


def test_forward():
    layer = InputBytesDef().build_jax()
    tokens = jnp.array([[0, 1, 255], [10, 20, 30]], dtype=jnp.int32)
    out = layer.forward(None, tokens)

    assert out.shape == (2, 3, 256)
    assert out.dtype == jnp.float32

    # Check one-hot correctness
    assert out[0, 0, 0] == 1.0
    assert out[0, 1, 1] == 1.0
    assert out[0, 2, 255] == 1.0
    assert out[1, 0, 10] == 1.0

    # Each position sums to 1
    np.testing.assert_array_equal(out.sum(axis=-1), jnp.ones((2, 3)))


def test_forward_matches_step():
    layer = InputBytesDef().build_jax()
    tokens = jnp.array([[0, 65, 255]], dtype=jnp.int32)
    fwd_out = layer.forward(None, tokens)

    for t in range(tokens.shape[1]):
        _, step_out = layer.step(None, None, int(tokens[0, t]))
        np.testing.assert_array_equal(fwd_out[0, t], step_out)


def test_forward_with_padding():
    """Padding positions carry the uniform (max-entropy) vector."""
    layer = InputBytesDef().build_jax()
    tokens = jnp.array([[7, 8]], dtype=jnp.int32)
    out = layer.forward(None, tokens, padding=2)
    assert out.shape == (1, 4, 256)
    np.testing.assert_allclose(out[0, 0], 1.0 / 256, atol=1e-7)
    np.testing.assert_allclose(out[0, 1], 1.0 / 256, atol=1e-7)
    assert out[0, 2, 7] == 1.0


def test_dtype_fp16():
    layer = InputBytesDef(precision=Precision.FP16).build_jax()
    _, out = layer.step(None, None, 42)
    assert out.dtype == jnp.float16

    out = layer.forward(None, jnp.array([[1, 2]], dtype=jnp.int32))
    assert out.dtype == jnp.float16


def test_dtype_bf16():
    layer = InputBytesDef(precision=Precision.BF16).build_jax()
    out = layer.forward(None, jnp.array([[1, 2]], dtype=jnp.int32))
    assert out.dtype == jnp.bfloat16


def test_from_numpy():
    """Verify it works with numpy input, matching dataset usage."""
    layer = InputBytesDef().build_jax()
    data = np.array([[72, 101, 108, 108, 111]], dtype=np.int32)  # "Hello"
    out = layer.forward(None, jnp.asarray(data))
    assert out.shape == (1, 5, 256)
    assert out[0, 0, 72] == 1.0  # 'H'
