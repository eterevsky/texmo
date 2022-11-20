import numpy as np
from unittest import TestCase

from texmo.layers.dense import Dense


class DenseTest(TestCase):
    def test_step(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([1, 2, 3, 4])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        state, out = layer.step(weights, None, input)
        self.assertIsNone(state)
        self.assertTrue(all(out == np.array([15, 30])))

    def test_forward1(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input)
        _, out1 = layer.step(weights, None, input[0])
        out_fs = layer._forward_from_step(weights, input)

        self.assertTrue((out[0] == out1).all())
        self.assertTrue((out == out_fs).all())

    def test_forward2(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input)
        _, out0 = layer.step(weights, None, input[0])
        _, out1 = layer.step(weights, None, input[1])
        out_fs = layer._forward_from_step(weights, input)

        self.assertTrue((out == np.stack((out0, out1))).all())
        self.assertTrue((out == out_fs).all())

    def test_forward_batch1(self):
        layer = Dense(2, input_shape=(4,))
        input = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])

        weights = {
            "w": np.array([[2, 0, 1, 0], [0, 1, 0, 2]]),
            "b": np.array([10, 20]),
        }

        out = layer.forward(weights, input[0])
        out_fb = layer.forward_batch(weights, input)
        out_fb_ff = layer._forward_batch_from_forward(weights, input)

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

        out0 = layer.forward(weights, input[0])
        out1 = layer.forward(weights, input[1])
        out = np.stack((out0, out1))
        out_fb = layer.forward_batch(weights, input)
        out_fb_ff = layer._forward_batch_from_forward(weights, input)
        out_fb_fs = layer._forward_batch_from_step(weights, input)

        self.assertTrue((out == out_fb).all())
        self.assertTrue((out == out_fb_ff).all())
        self.assertTrue((out == out_fb_fs).all())
