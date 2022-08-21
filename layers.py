import math
import jax
import jax.numpy as jnp
from jax.random import split, normal

from model import NCHAR


def scale_weights(weights, scale):
    for k in weights.keys():
        weights[k] = scale * weights[k]
    return weights


class Layer(object):
    def __init__(self):
        pass

    def init_state(self, key):
        return None

    def init_weights(self, key, scale=1.0):
        return None

    def step(self, weights, input):
        raise NotImplementedError

    def step2(self, weights, state, input):
        """Returns: (state, output)"""
        raise NotImplementedError


class Suffix(Layer):
    name = 'suffix'

    def __init__(self, length, train_init=False):
        super().__init__()
        self._length = length
        self._train_init = train_init
        self.full_name = f'suffix.{length}'

    def init_weights(self, key, scale=0.1):
        if self._train_init:
            return scale * normal(key, shape=(self._length - 1, NCHAR))
        else:
            return None

    def init_state(self, params=None):
        if self._train_init:
            return params
        else:
            return jnp.ones((self._length - 1, NCHAR)) / NCHAR

    def step(self, weights, input, state):
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix

    def step2(self, weights, state, input):
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix


class FeedForward(Layer):
    name = 'dense'

    def __init__(self, input_size, output_size, activation=None, sigmoid=False, relu=False):
        super().__init__()
        self._input = input_size
        self._output = output_size
        suffix = ''
        if sigmoid:
            activation = jax.nn.sigmoid
            suffix = '.sigmoid'
        elif relu:
            activation = jax.nn.relu
            suffix = '.relu'
        self._activation = activation
        self.full_name = f'dense.{input_size}.{output_size}{suffix}'

    def init_weights(self, key, scale=0.1):
        key0, key1 = split(key)
        weights = {
            'w': normal(key0, shape=(self._output, self._input)),
            'b': normal(key1, shape=(self._output,)),
        }
        weights = scale_weights(weights, scale)
        return weights

    def init_state(self):
        return None

    def step(self, weights, input):
        out = jnp.dot(weights['w'], input.flatten()) + weights['b']
        if self._activation is not None:
            out = self._activation(out)
        return out

    def step2(self, weights, state, input):
        out = jnp.dot(weights['w'], input.flatten()) + weights['b']
        if self._activation is not None:
            out = self._activation(out)
        return state, out


class Convolution(Layer):
    name = 'conv'

    def __init__(self, input_size, kernel_size, output_size,
                 activation=jnp.tanh, tanh=False, sigmoid=False, relu=False):
        """Transforms [X, input_size] into [X - kernel_size + 1, output_size]"""
        super().__init__()
        self._input = input_size
        self._kernel = kernel_size
        self._output = output_size
        suffix = ''
        if sigmoid:
            activation = jax.nn.sigmoid
            suffix = '.sigmoid'
        elif relu:
            activation = jax.nn.relu
            suffix = '.relu'
        elif tanh:
            activation = jnp.tanh
            suffix = '.tanh'
        self._activation = activation
        self.full_name = f'conv.{input_size}.{kernel_size}.{output_size}{suffix}'

    def init_state(self):
        return None

    def init_weights(self, key, scale=0.1):
        key0, key1 = split(key)
        weights = {
            'kernel': normal(key0, shape=(self._output, self._kernel, self._input)),
            'b': normal(key1, shape=(self._output,))
        }
        return scale_weights(weights, scale)

    def step(self, weights, input):
        kernel = weights['kernel']
        input = jnp.expand_dims(input, axis=0)
        dn = jax.lax.conv_dimension_numbers(
            input.shape, kernel.shape, ('NWC', 'OWI', 'NWC'))
        out = jax.lax.conv_general_dilated(
            input, kernel, (1,), 'VALID', (1,), (1,), dn)
        out = jnp.squeeze(out)
        out += jnp.expand_dims(weights['b'], axis=0)

        out = self._activation(out)

        return out

    def step2(self, weights, state, input):
        kernel = weights['kernel']
        input = jnp.expand_dims(input, axis=0)
        dn = jax.lax.conv_dimension_numbers(
            input.shape, kernel.shape, ('NWC', 'OWI', 'NWC'))
        out = jax.lax.conv_general_dilated(
            input, kernel, (1,), 'VALID', (1,), (1,), dn)
        out = jnp.squeeze(out)
        out += jnp.expand_dims(weights['b'], axis=0)

        out = self._activation(out)

        return state, out


class Recurrent(Layer):
    name = 'rec'

    def __init__(self, input_size, state_size, sigmoid=False, relu=False,
                 tanh=False):
        super().__init__()
        self._input = input_size
        self._state = state_size
        if sigmoid:
            self._activation = jax.nn.sigmoid
            suffix = '.sigmoid'
        elif relu:
            self._activation = jax.nn.relu
            suffix = '.relu'
        elif tanh:
            self._activation = jnp.tanh
            suffix = '.tanh'
        else:
            self._activation = None
            suffix = ''
        self.full_name = f'rec.{input_size}.{state_size}{suffix}'

    def init_state(self):
        return jnp.zeros((self._state,))

    def init_weights(self, key, scale=0.1):
        keys = split(key, 3)
        weights = {
            'winput': normal(keys[0], shape=(self._state, self._input)),
            'wstate': normal(keys[1], shape=(self._state, self._state)),
            'b': normal(keys[2], shape=(self._state,)),
        }
        total_size = self._input + self._state
        scale = math.sqrt(2 / total_size)
        return scale_weights(weights, scale)

    def step(self, weights, input, state):
        input = input.flatten()
        state = jnp.dot(weights['winput'], input) + \
            jnp.dot(weights['wstate'], state) + weights['b']
        if self._activation is not None:
            state = self._activation(state)
        return state

    def step2(self, weights, state, input):
        input = input.flatten()
        state = jnp.dot(weights['winput'], input) + \
            jnp.dot(weights['wstate'], state) + weights['b']
        if self._activation is not None:
            state = self._activation(state)
        return state, state


class Gru(Layer):
    name = 'gru'

    def __init__(self, input_size, state_size, relu=False):
        super().__init__()
        self._input = input_size
        self._state = state_size
        if relu:
            self._activation = jax.nn.relu
            suffix = '.relu'
        else:
            self._activation = jnp.tanh
            suffix = '.tanh'
        self.full_name = f'gru.{input_size}.{state_size}{suffix}'

    def init_state(self):
        return jnp.zeros((self._state,))

    def init_weights(self, key, scale=0.1):
        keys = split(key, 9)
        weights = {
            'wz': jax.random.normal(keys[0], shape=(self._state, self._input)),
            'uz': jax.random.normal(keys[1], shape=(self._state, self._state)),
            'bz': jax.random.normal(keys[2], shape=(self._state,)),

            'wr': jax.random.normal(keys[3], shape=(self._state, self._input)),
            'ur': jax.random.normal(keys[4], shape=(self._state, self._state)),
            'br': jax.random.normal(keys[5], shape=(self._state,)),

            'wh': jax.random.normal(keys[6], shape=(self._state, self._input)),
            'uh': jax.random.normal(keys[7], shape=(self._state, self._state)),
            'bh': jax.random.normal(keys[8], shape=(self._state,)),
        }
        return scale_weights(weights, scale)

    def step(self, weights, input, state):
        input = input.flatten()
        z = jnp.dot(weights['wz'], input) + \
            jnp.dot(weights['uz'], state) + weights['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(weights['wr'], input) + \
            jnp.dot(weights['ur'], state) + weights['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(weights['wh'], input) + \
            jnp.dot(weights['uh'], r * state) + weights['bh']
        hc = self._activation(hc)

        return (1 - z) * state + z * hc

    def step2(self, weights, state, input):
        input = input.flatten()
        z = jnp.dot(weights['wz'], input) + \
            jnp.dot(weights['uz'], state) + weights['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(weights['wr'], input) + \
            jnp.dot(weights['ur'], state) + weights['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(weights['wh'], input) + \
            jnp.dot(weights['uh'], r * state) + weights['bh']
        hc = self._activation(hc)

        state = (1 - z) * state + z * hc
        return state, state


class Lstm(Layer):
    name = 'lstm'

    def __init__(self, input_size, state_size):
        super().__init__()
        self._input = input_size
        self._state = state_size
        self.full_name = f'lstm.{input_size}.{state_size}'

    def init_state(self):
        return {'h': jnp.zeros((self._state,)), 'c': jnp.zeros((self._state,))}

    def init_weights(self, key, scale=0.01):
        keys = split(key, 9)
        weights = {
            'wf': jax.random.normal(keys[0], shape=(self._state, self._input)),
            'uf': jax.random.normal(keys[1], shape=(self._state, self._state)),
            'bf': jax.random.normal(keys[2], shape=(self._state,)),

            'wi': jax.random.normal(keys[3], shape=(self._state, self._input)),
            'ui': jax.random.normal(keys[4], shape=(self._state, self._state)),
            'bi': jax.random.normal(keys[5], shape=(self._state,)),

            'wo': jax.random.normal(keys[6], shape=(self._state, self._input)),
            'uo': jax.random.normal(keys[7], shape=(self._state, self._state)),
            'bo': jax.random.normal(keys[8], shape=(self._state,)),

            'wc': jax.random.normal(keys[9], shape=(self._state, self._input)),
            'uc': jax.random.normal(keys[10], shape=(self._state, self._state)),
            'bc': jax.random.normal(keys[11], shape=(self._state,)),
        }
        return scale_weights(weights, scale)

    def step(self, weights, input, state):
        input = input.flatten()

        h = state['h']
        c = state['c']

        f = jnp.dot(weights['wf'], input) + \
            jnp.dot(weights['uf'], h) + weights['bf']
        f = jax.nn.sigmoid(f)

        i = jnp.dot(weights['wi'], input) + \
            jnp.dot(weights['ui'], h) + weights['bi']
        i = jax.nn.sigmoid(i)

        o = jnp.dot(weights['wo'], input) + \
            jnp.dot(weights['uo'], h) + weights['bo']
        o = jax.nn.sigmoid(o)

        cn = jnp.dot(weights['wc'], input) + \
            jnp.dot(weights['uc'], h) + weights['bc']
        cn = jax.nn.tanh(cn)

        c = f * c + i * cn
        h = o * c

        return {'h': h, 'c': c}

    def step2(self, weights, state, input):
        state = self.step(weights, input, state)
        return state, state['h']


LAYERS_BY_NAME = {
    'suffix': Suffix,
    'dense': FeedForward,
    'conv': Convolution,
    'rec': Recurrent,
    'gru': Gru,
    'lstm': Lstm,
}
