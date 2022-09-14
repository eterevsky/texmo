import unittest
from unittest import TestCase
import jax.numpy as jnp

import layers2


class ModelsTest(TestCase):
    def test_conv(self):
        input = []
        for i in range(4):
            b = []
            for j in range(10):
                b.append(0.1 * i + 0.01 * j)
            input.append(b)
        input = jnp.array(input)

        kernel = []
        for o in range(4):
            a = []
            for i in range(2):
                b = []
                for j in range(10):
                    b.append(0.3 * i + 0.02 * j + o)
                a.append(b)
            kernel.append(a)
        kernel = jnp.array(kernel)

        conv1 = layers2.Convolution(2, 4, input_shape=(4, 10), relu=True)
        conv2 = layers2.Convolution2(2, 4, input_shape=(4, 10), relu=True)

        b = jnp.zeros((4,))

        r1 = conv1.step({'kernel': kernel, 'b': b}, None, input)
        r2 = conv2.step({'kernel': kernel, 'b': b}, None, input)

        assert r1.shape == (3, 4)
        assert r2.shape == (3, 4)
        assert r1 == r2


if __name__ == '__main__':
    unittest.main()