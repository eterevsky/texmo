import logging
from unittest import TestCase
import os

import numpy as np
from jax import numpy as jnp

from texmo.layers.input import Input
from texmo.prng import Rng
from texmo.tokens import set_tokens_dir

logging.disable(level=logging.ERROR)


set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


class InputTest(TestCase):
    def test_parse1(self):
        input = Input.parse("tokens.256-pos.16-emb.128")
        self.assertEqual(input.ntokens, 256)
        self.assertFalse(input._emb_norm)
        self.assertEqual(input._emb_size, 128)
        self.assertEqual(input._positions, 16)
        self.assertEqual(input.weights, (256 + 1) * 128 + 16 * 128)
        self.assertEqual(input.output_size, 128)

    def test_parse2(self):
        input = Input.parse("tokens.256-emb.128.norm")
        self.assertEqual(input.ntokens, 256)
        self.assertTrue(input._emb_norm)
        self.assertEqual(input._emb_size, 128)
        self.assertIs(input._positions, None)
        self.assertEqual(input.weights, (256 + 1) * 128)
        self.assertEqual(input.output_size, 128)

    def test_parse3(self):
        input = Input.parse("bytes-onehot")
        self.assertEqual(input.ntokens, 256)
        self.assertFalse(input._emb_norm)
        self.assertIs(input._emb_size, None)
        self.assertIs(input._positions, None)
        self.assertEqual(input.weights, 0)
        self.assertEqual(input.output_size, 256)

    def test_parse4(self):
        input = Input.parse("bytes-pos.16-onehot")
        self.assertEqual(input.ntokens, 256)
        self.assertFalse(input._emb_norm)
        self.assertIs(input._emb_size, None)
        self.assertEqual(input._positions, 16)
        self.assertEqual(input.weights, 0)
        self.assertEqual(input.output_size, 256 + 16)

    def test_onehot_step(self):
        input = Input.parse("tokens.4-onehot")
        self.assertEqual(input.weights, 0)
        rng = Rng()
        self.assertFalse(input.init_weights(rng))

        _new_state, out = input.step({}, {}, 2)
        self.assertEqual(list(out), [0, 0, 1, 0])

    def test_onehot_forward(self):
        input = Input.parse("tokens.4-onehot")

        out = input.forward_batch(
            weights={}, input=jnp.array([[2, 3], [1, 0]]), padding_len=1
        )
        np.testing.assert_array_equal(
            out,
            [
                [[0.25, 0.25, 0.25, 0.25], [0, 0, 1, 0], [0, 0, 0, 1]],
                [[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [1, 0, 0, 0]],
            ],
        )

    def test_pos_onehot_step(self):
        input = Input.parse("tokens.4-pos.2-onehot")
        self.assertEqual(input.weights, 0)
        rng = Rng()
        self.assertFalse(input.init_weights(rng))
        state = input.init_state(rng)
        self.assertEqual(state["position"], 0)

        new_state, out = input.step({}, state, 2)
        self.assertEqual(list(out), [0, 0, 1, 0, 1, 0])
        self.assertEqual(new_state["position"], 1)

    def test_pos_onehot_forward(self):
        input = Input.parse("tokens.4-pos.2-onehot")

        out = input.forward_batch(
            weights={}, input=jnp.array([[2, 3], [1, 0]]), padding_len=1
        )
        np.testing.assert_array_equal(
            out,
            [
                [
                    [0.25, 0.25, 0.25, 0.25, 0, 1],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 0, 1],
                ],
                [
                    [0.25, 0.25, 0.25, 0.25, 0, 1],
                    [0, 1, 0, 0, 1, 0],
                    [1, 0, 0, 0, 0, 1],
                ],
            ],
        )

    def test_emb_step(self):
        input = Input.parse("tokens.4-pos.2-emb.2")
        rng = Rng()
        weights = input.init_weights(rng)
        self.assertEqual(weights["tokens"].shape, (5, 2))
        self.assertEqual(weights["positions"].shape, (2, 2))
        state = input.init_state(rng)

        new_state, out = input.step(
            {
                "tokens": np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
                "positions": np.array([[0, 1], [0, 2]]),
            },
            state,
            2,
        )
        self.assertEqual(list(out), [3, 1])
        self.assertEqual(new_state["position"], 1)

    def test_emb_forward(self):
        input = Input.parse("tokens.4-pos.2-emb.2")
        out = input.forward_batch(
            weights={
                "tokens": np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
                "positions": np.array([[0, 1], [0, 2]]),
            },
            input=jnp.array([[2, 3], [1, 0]]),
            padding_len=1,
        )
        np.testing.assert_array_equal(
            out,
            [
                [
                    [5, 2],
                    [3, 1],
                    [4, 2],
                ],
                [
                    [5, 2],
                    [2, 1],
                    [1, 2],
                ],
            ],
        )
