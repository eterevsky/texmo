import numpy as np
from unittest import TestCase

from texmo.layers.suffix import Suffix


class SuffixTest(TestCase):
    def test_step(self):
        layer = Suffix(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        state = layer.init_state(None)

        state, out0 = layer.step(None, state, input[0])
        state, out1 = layer.step(None, state, input[1])
        state, out2 = layer.step(None, state, input[2])
        self.assertTrue((out0[1] == np.array([1, 2, 3, 4])).all())
        self.assertTrue((out1 == np.array([[1, 2, 3, 4], [5, 6, 7, 8]])).all())
        self.assertTrue(
            (out2 == np.array([[5, 6, 7, 8], [9, 10, 11, 12]])).all()
        )

    def test_forward(self):
        layer = Suffix(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        out_fw = layer.forward(None, input)
        out_fw_fs = layer._forward_from_step(None, input)

        expected = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[5, 6, 7, 8], [9, 10, 11, 12]]]
        )

        self.assertTrue((expected == out_fw).all())
        self.assertTrue((expected == out_fw_fs).all())

    def test_forward_batch(self):
        layer = Suffix(2, input_shape=(4,))
        input = np.array(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
            ]
        )

        out = layer.forward_batch(None, input)
        out_ff = layer._forward_batch_from_forward(None, input)
        out_fs = layer._forward_batch_from_step(None, input)

        expected = np.array(
            [
                [[[1, 2, 3, 4], [5, 6, 7, 8]], [[5, 6, 7, 8], [9, 10, 11, 12]]],
                [
                    [[13, 14, 15, 16], [17, 18, 19, 20]],
                    [[17, 18, 19, 20], [21, 22, 23, 24]],
                ],
            ]
        )

        self.assertTrue((expected == out).all())
        self.assertTrue((expected == out_ff).all())
        self.assertTrue((expected == out_fs).all())
