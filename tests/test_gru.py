from jax import numpy as jnp
from unittest import TestCase

from texmo.layers.gru import Gru
from texmo.prng import Rng


class GruTest(TestCase):
    def test_forward(self):
        layer = Gru(2, input_shape=(4,))
        input = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        rng = Rng()
        weights = layer.init_weights(rng, 1.0)

        state = layer.init_state(weights)
        state0, out0 = layer.step(weights, state, input[0])
        state1, out1 = layer.step(weights, state0, input[1])
        _, out2 = layer.step(weights, state1, input[2])

        out = jnp.stack((out0, out1, out2))
        out_fw = layer.forward(weights, input)

        self.assertTrue(jnp.linalg.norm(out - out_fw) < 1E-6)

    def test_forward_batch(self):
        layer = Gru(2, input_shape=(4,))
        input = jnp.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ]
        )

        rng = Rng()
        weights = layer.init_weights(rng, 1.0)

        out_ff = layer._forward_batch_from_forward(weights, input)
        out_fs = layer._forward_batch_from_step(weights, input)

        self.assertTrue(jnp.linalg.norm(out_ff - out_fs) < 1E-6)
