import numpy as np
from unittest import main, TestCase

from layers2 import Attn


class Layers2Test(TestCase):
    def test_attn(self):
        layer = Attn(4, 2, 8, input_shape=(4,))
        weights = {
            "wkey": np.zeros((2, 4, 4)),
            "wquery": np.array(
                [
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                ]
            ),
            "wvalue": np.zeros((2, 4, 4)),
            "bkey": np.zeros((4, 2, 4)),
            "bquery": np.zeros((2, 4)),
        }

        state = {
            "keys": np.array(
                [
                    [[1, 0, 0, 0], [0, 1, 0, 0]],
                    [[0, 1, 0, 0], [0, 0, 1, 0]],
                    [[0, 0, 1, 0], [0, 0, 0, 1]],
                    # [[0, 0, 0, 0], [0, 0, 0, 0]],
                ]
            ),
            "values": np.array(
                [
                    [[0, 0, 1, 0], [0, 0, 0, 2]],
                    [[0, 0, 0, 1], [2, 0, 0, 0]],
                    [[1, 0, 0, 0], [0, 2, 0, 0]],
                    # [[0, 0, 0, 0], [0, 0, 0, 0]],
                ]
            ),
        }

        input = np.array([1, 0, 0, 0])

        state, out = layer.step(weights, state, input)

        out0 = out[0:4]

        self.assertEqual(out0[1], 0)
        self.assertGt(out0[2], out0[0])
        self.assertGt(out0[2], out0[3])
