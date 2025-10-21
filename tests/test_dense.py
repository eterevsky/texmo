import logging
from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from texmo.layers.dense import Dense

logging.disable(level=logging.ERROR)


class DenseTest(TestCase):
    def test_step(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([1, 2, 3, 4])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        state, out = layer.step(weights, None, input, dtype=jnp.float32)
        self.assertIsNone(state)
        self.assertTrue(all(out == np.array([15, 30])))

    def test_forward1(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input, dtype=jnp.float32)
        _, out1 = layer.step(weights, None, input[0], dtype=jnp.float32)
        out_fs = layer._forward_from_step(weights, input, dtype=jnp.float32)

        self.assertTrue((out[0] == out1).all())
        self.assertTrue((out == out_fs).all())

    def test_forward2(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input, dtype=jnp.float32)
        _, out0 = layer.step(weights, None, input[0], dtype=jnp.float32)
        _, out1 = layer.step(weights, None, input[1], dtype=jnp.float32)
        out_fs = layer._forward_from_step(weights, input, dtype=jnp.float32)

        self.assertTrue((out == np.stack((out0, out1))).all())
        self.assertTrue((out == out_fs).all())

    def test_forward_batch1(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input[0], dtype=jnp.float32)
        out_fb = layer.forward_batch(weights, input, dtype=jnp.float32)
        out_fb_ff = layer._forward_batch_from_forward(weights, input, dtype=jnp.float32)

        self.assertTrue((out == out_fb[0]).all())
        self.assertTrue((out == out_fb_ff[0]).all())

    def test_forward_batch2(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
        )

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out0 = layer.forward(weights, input[0], dtype=jnp.float32)
        out1 = layer.forward(weights, input[1], dtype=jnp.float32)
        out = np.stack((out0, out1))
        out_fb = layer.forward_batch(weights, input, dtype=jnp.float32)
        out_fb_ff = layer._forward_batch_from_forward(weights, input, dtype=jnp.float32)
        out_fb_fs = layer._forward_batch_from_step(weights, input, dtype=jnp.float32)

        self.assertTrue((out == out_fb).all())
        self.assertTrue((out == out_fb_ff).all())
        self.assertTrue((out == out_fb_fs).all())
