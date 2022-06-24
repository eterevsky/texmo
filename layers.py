import jax
import jax.numpy as jnp
from jax.random import split, normal


def scale_params(params, scale):
    for k in params.keys():
        params[k] = scale * params[k]
    return params


class Layer(object):
    def __init__(self):
        pass

    def init_params(self, key, scale=1.0):
        return {}

    def step(self, params, input):
        return input


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
        return scale_params(params, scale)

    def step(self, params, input):
        return jnp.dot(params['w'], input.flatten()) + params['b']


class Recurrent(Layer):
    def __init__(self, input_size, state_size, output_size):
        super().__init__()
        self._input = input_size
        self._state = state_size
        self._output = output_size

    def init_state(self):
        return jnp.zeros((self._state,))

    def init_params(self, key, scale=0.1):
        keys = split(key, 3)
        params = {
            'winput': normal(keys[0], shape=(self._state, self._input)),
            'wstate': normal(keys[1], shape=(self._state, self._state)),
            'b': normal(keys[2], shape=(self._state,)),
        }
        return scale_params(params, scale)

    def step(self, params, input, state):
        input = input.flatten()
        state = jnp.dot(params['winput'], input) + jnp.dot(params['wstate'], state) + params['b']
        return state

