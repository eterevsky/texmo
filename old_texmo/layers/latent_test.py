import jax.numpy as jnp
import numpy as np

from .latent import Latent, _normalize


def test_step():
    layer = Latent(2, reps=2, input_shape=(1,))

    _, out = layer.step(
        weights={
            'wi': np.array([[1], [2]]),
            'wr': np.array([[3, 4], [5, 6]]),
            'b': np.array([7, 8])
        },
        _state=None,
        input=np.array([1], dtype=np.float32),
        dtype=jnp.float32)
    assert out.dtype == jnp.float32

    input_contrib = np.array([8, 10])
    state1 = np.tanh(input_contrib)
    s = _normalize(state1)
    out2 = np.tanh(input_contrib + np.array([[3, 4], [5, 6]]) @ s)

    assert np.linalg.norm(out - out2) < 1E-4


def test_forward_batch():
    layer = Latent(2, reps=4, input_shape=(1,))

    input = np.array([[[1], [2]], [[3], [4]]])
    weights = {
        'wi': np.array([[1], [2]]),
        'wr': np.array([[3, 4], [5, 6]]),
        'b': np.array([-1, -2])
    }

    _, out1 = layer.forward_batch(weights, input, dtype=jnp.float32)
    _, out2 = layer._forward_batch_from_step(weights, input, dtype=jnp.float32)

    assert (out1 == out2).all()
