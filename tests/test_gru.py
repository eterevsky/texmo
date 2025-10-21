import logging
from unittest import TestCase

from jax import numpy as jnp

from texmo.layers.gru import Gru
from texmo.prng import Rng

logging.disable(level=logging.ERROR)


class GruTest(TestCase):
    def test_forward(self):
        layer = Gru(2, input_shape=(4,))
        input = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        rng = Rng()
        weights = layer.init_weights(rng, 1.0, dtype=jnp.float32)

        state = layer.init_state(weights, dtype=jnp.float32)
        state0, out0 = layer.step(weights, state, input[0], dtype=jnp.float32)
        state1, out1 = layer.step(weights, state0, input[1], dtype=jnp.float32)
        _, out2 = layer.step(weights, state1, input[2], dtype=jnp.float32)

        out = jnp.stack((out0, out1, out2))
        out_fw = layer.forward(weights, input, dtype=jnp.float32)

        self.assertTrue(jnp.linalg.norm(out - out_fw) < 1e-6)

    def test_forward_batch(self):
        layer = Gru(2, input_shape=(4,))
        input = jnp.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ]
        )

        rng = Rng()
        weights = layer.init_weights(rng, 1.0, dtype=jnp.float32)

        out_ff = layer._forward_batch_from_forward(weights, input, dtype=jnp.float32)
        out_fs = layer._forward_batch_from_step(weights, input, dtype=jnp.float32)

        self.assertTrue(jnp.linalg.norm(out_ff - out_fs) < 1e-6)


class MinGruTest(TestCase):

    def test_forward_batch(self):
        layer = Gru(2, input_shape=(4,))
        input = jnp.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ],
            dtype=jnp.float32
        )

        rng = Rng()
        weights = layer.init_weights(rng, 1.0, dtype=jnp.float32)

        out_ff = layer._forward_batch_from_forward(weights, input, dtype=jnp.float32)
        out_fs = layer._forward_batch_from_step(weights, input, dtype=jnp.float32)
        out_fb = layer.forward_batch(weights, input, dtype=jnp.float32)

        self.assertTrue(jnp.linalg.norm(out_ff - out_fs) < 1e-6)
        self.assertTrue(jnp.linalg.norm(out_fb - out_fs) < 1e-6)
