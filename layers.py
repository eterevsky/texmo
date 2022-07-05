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
        params = scale_params(params, scale)
        return params

    def step(self, params, input):
        return jnp.dot(params['w'], input.flatten()) + params['b']


class Recurrent(Layer):
    def __init__(self, input_size, state_size):
        super().__init__()
        self._input = input_size
        self._state = state_size

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


class Convolution(Layer):
    def __init__(self, input_size, kernel_size, output_size):
        super().__init__()
        self._input = input_size
        self._kernel = kernel_size
        self._output = output_size

    def init_params(self, key, scale=0.1):
        key0, key1 = split(key)
        params = {
            'kernel': normal(key0, shape=(self._output, self._kernel, self._input)),
            'b': normal(key1, shape=(self._output,))
        }
        return scale_params(params, scale)

    def step(self, params, input):
        kernel = params['kernel']
        input = jnp.expand_dims(input, axis=0)
        dn = jax.lax.conv_dimension_numbers(input.shape, kernel.shape, ('NWC', 'OWI', 'NWC'))
        out = jax.lax.conv_general_dilated(input, kernel, (1,), 'VALID', (1,), (1,), dn)
        out = jnp.squeeze(out)
        out += jnp.expand_dims(params['b'], axis=0)

        return out


class Gru(Layer):
    def __init__(self, input_size, state_size):
        super().__init__()
        self._input = input_size
        self._state = state_size

    def init_state(self):
        return jnp.zeros((self._state,))

    def init_params(self, key, scale=0.1):
        keys = split(key, 9)
        params = {
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
        return scale_params(params, scale)

    def step(self, params, input, state):
        input = input.flatten()
        z = jnp.dot(params['wz'], input) + jnp.dot(params['uz'], state) + params['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(params['wr'], input) + jnp.dot(params['ur'], state) + params['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(params['wh'], input) + jnp.dot(params['uh'], r * state) + params['bh']
        hc = jax.nn.tanh(hc)

        return (1 - z) * state + z * hc


class Lstm(Layer):
    def __init__(self, input_size, state_size):
        super().__init__()
        self._input = input_size
        self._state = state_size

    def init_state(self):
        return {'h': jnp.zeros((self._state,)), 'c': jnp.zeros((self._state,))}

    def init_params(self, key, scale=0.1):
        keys = split(key, 9)
        params = {
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
        return scale_params(params, scale)

    def step(self, params, input, state):
        input = input.flatten()

        h = state['h']
        c = state['c']

        f = jnp.dot(params['wf'], input) + jnp.dot(params['uf'], h) + params['bf']
        f = jax.nn.sigmoid(f)

        i = jnp.dot(params['wi'], input) + jnp.dot(params['ui'], h) + params['bi']
        i = jax.nn.sigmoid(i)

        o = jnp.dot(params['wo'], input) + jnp.dot(params['uo'], h) + params['bo']
        o = jax.nn.sigmoid(o)

        cn = jnp.dot(params['wc'], input) + jnp.dot(params['uc'], h) + params['bc']
        cn = jax.nn.tanh(cn)

        c = f * c + i * cn
        h = o * c

        return {'h': h, 'c': c}
