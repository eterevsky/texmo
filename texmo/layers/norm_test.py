import jax.numpy as jnp
import numpy as np

from .norm import Norm


def test_step():
    layer = Norm(input_shape=(2,))

    state, out = layer.step(None, None, np.array([1, 2]), dtype=jnp.float32)
    assert out.dtype == jnp.float32

    assert np.linalg.norm(out - np.array([0.4472136, 0.89442719])) < 1E-5


def test_firward_batch():
    layer = Norm(input_shape=(2,))

    input = np.array([[[1, 2], [2, 3], [0, 0]], [[-1, 2], [2, -3], [0, 0]]])

    out = layer.forward_batch(None, input, dtype=jnp.float32)
    assert out.dtype == jnp.float32

    expected = np.array([[[0.4472136, 0.89442719], [0.5547002, 0.83205029], [0, 0]],
                         [[-0.4472136, 0.89442719], [0.5547002, -0.83205029], [0, 0]]])

    assert np.linalg.norm(out - expected) < 1E-5
