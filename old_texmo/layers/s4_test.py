import jax.numpy as jnp
import numpy as np

from .s4 import S4


def test_step():
    layer = S4(2, input_shape=(2,))

    state, out = layer.step(
        weights={'mix': np.array([[2, 1], [3, 4]], dtype=np.float32)},
        state=np.array([[3, 4], [5, 6]]),
        input=np.array([1, 2]),
        dtype=jnp.float32)
    assert out.dtype == jnp.float32
    assert (state == out).all()

    assert (out == np.array([[1, 2], [3, 4]] + np.array([[10, 6], [15, 24]]))).all()

def test_step_batch():
    layer = S4(2, input_shape=(2,))

    input = np.array([[1, 2], [3, 4]])
    state = np.array([[[3, 4], [5, 6]], [[7, 8], [9, 10]]])
    weights={'mix': np.array([[2, 1], [3, 4]], dtype=np.float32)}

    print('***', state.shape)

    _, out1 = layer.step_batch(weights, state, input, dtype=jnp.float32)
    _, out2 = layer._step_batch_from_step(weights, state, input, dtype=jnp.float32)

    assert (out1 == out2).all()