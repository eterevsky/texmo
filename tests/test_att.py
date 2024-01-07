import logging
from unittest import TestCase

import numpy as np
from jax import numpy as jnp

from texmo.layers.att import Attention
from texmo.prng import Rng

logging.disable(level=logging.ERROR)


class AttTest(TestCase):
    def test_step(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        state = layer.init_state(weights)
        state, out1 = layer.step(weights, state, input=jnp.array([1, 2, 3]))
        np.testing.assert_array_almost_equal(out1, [1, 2, 2, 3])

        state, out2 = layer.step(weights, state, input=jnp.array([3, 2, 1]))

        layer_mk = Attention(length=4, heads=2, size=4, mk=True, input_shape=(3,))
        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [1, 0, 0]]]),
        }

        state = layer_mk.init_state(weights)
        state, out_mk1 = layer_mk.step(weights, state, input=jnp.array([1, 2, 3]))
        state, out_mk2 = layer_mk.step(weights, state, input=jnp.array([3, 2, 1]))

        np.testing.assert_array_almost_equal(out1, out_mk1)
        np.testing.assert_array_almost_equal(out2, out_mk2)

    def test_forward_batch_short1(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        input = jnp.array([[[1, 2, 3], [4, 5, 6]]])

        out_fb = layer.forward_batch_short(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)

    def test_forward_batch1(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        input = jnp.array([[[1, 2, 3], [4, 5, 6]]])

        out_fb = layer.forward_batch(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)

    def test_forward_batch2(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        input = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])

        out_fb = layer.forward_batch(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)


    def test_forward_batch_short(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        input = jnp.array([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [6, 5, 4]]])

        out_fb = layer.forward_batch_short(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)

    def test_forward_batch(self):
        layer = Attention(length=4, heads=2, size=4, mk=False, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array([[0, 1, 0], [1, 0, 0]]),
        }

        input = jnp.array([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [6, 5, 4]]])

        out_fb = layer.forward_batch(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)

    def test_forward_batch_mk(self):
        layer = Attention(length=4, heads=2, size=4, mk=True, input_shape=(3,))

        weights = {
            "wvalue": jnp.array(
                [
                    [[1, 0, 0], [0, 1, 0]],
                    [[0, 1, 0], [0, 0, 1]],
                ]
            ),
            "wquery": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
            "wkey": jnp.array(
                [
                    [[0, 1, 0], [1, 0, 0]],
                    [[0, 0, 1], [0, 1, 0]],
                ]
            ),
        }

        input = jnp.array([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [6, 5, 4]]])

        out_fb = layer.forward_batch(weights, input)
        out_step = layer._forward_batch_from_step_manual(weights, input)
        np.testing.assert_array_almost_equal(out_fb, out_step)
