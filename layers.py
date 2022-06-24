import jax
import jax.numpy as jnp
from jax.random import split, normal

class Layer(object):
    def __init__(self):
        pass

    def init_params(self, key, scale=1.0):
        return {}

    def step(self, params, input):
        return input

    def _scale(self, params, scale):
        for k in params.keys():
            params[k] = scale * params[k]
        return params


class FeedForward(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._input = input_size
        self._output = output_size

    def init_params(self, key, scale=0.1):
        key0, key1 = split(key)
        params = {
            'w': normal(key0, shape=(self._output, self._input)),
            'b': normal(key1, shape=(self._output,)),
        }
        return self._scale(params, scale)

    def step(self, params, input):
        return jnp.dot(params['w'], input.flatten()) + params['b']
