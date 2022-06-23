import jax
import jax.numpy as jnp
from jax.experimental import sparse
import optax

from model import Model, NCHAR


class Equal(Model):
    def serialize(self):
        return {'name': 'equal'}

    def step(self, params, state, c):
        return state, jnp.ones_like(c)


class Freq(Model):
    def serialize(self):
        return {'name': 'freq'}

    def init_params(self, key):
        return {'b': jax.random.normal(key, shape=(NCHAR,))}

    def step(self, params, state, c):
        out = params['b']
        return state, out


class Markov1(Model):
    """Logistic regression on the previous character."""

    def serialize(self):
        return {'name': 'markov1'}

    def init_params(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'w': jax.random.normal(key0, shape=(NCHAR, NCHAR)),
            'b': jax.random.normal(key1, shape=(NCHAR,)),
        }

    def step(self, params, state, c):
        out = jnp.dot(params['w'], c) + params['b']
        return state, out


class Markov2True(Model):
    """Logistic regression on the previous character."""

    def serialize(self):
        return {'name': 'markov2true'}

    def init_state(self):
        return jnp.ones((NCHAR,)) / NCHAR

    def init_params(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'w': jax.random.normal(key0, shape=(NCHAR, NCHAR * NCHAR)),
            'b': jax.random.normal(key1, shape=(NCHAR,)),
        }

    def step(self, params, state, c):
        prev = jnp.expand_dims(state, 1)
        cexp = jnp.expand_dims(c, 0)
        cross = prev * cexp
        cross = cross.flatten()
        out = jnp.dot(params['w'], cross) + params['b']
        return c, out


class Markov2(Model):
    """Feed forward based on two previous characeters with a hidden layer.

    prev + current
         V
       hidden
         V
        out
    """

    def __init__(self, hidden=128):
        self._hidden = hidden

    def serialize(self):
        return {'name': 'markov2', 'hidden': self._hidden}

    def init_state(self):
        return jnp.zeros((NCHAR,))

    def init_params(self, key):
        key = jax.random.split(key, 5)
        return {
            'w': jax.random.normal(key[0], shape=(self._hidden, NCHAR)),
            'wprev': jax.random.normal(key[1], shape=(self._hidden, NCHAR)),
            'b': jax.random.normal(key[2], shape=(self._hidden,)),

            'wout': jax.random.normal(key[3], shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(key[4], shape=(NCHAR,)),
        }

    def step(self, params, state, c):
        hidden = jnp.dot(params['w'], c) + jnp.dot(params['wprev'], state) + params['b']
        hidden = jax.nn.tanh(hidden)
        out = jnp.dot(params['wout'], hidden) + params['bout']
        return c, out


class MarkovFlex(Model):
    def __init__(self, hidden=128, state_size=256):
        self._hidden = 128
        self._state_size = 256

    def serialize(self):
        return {'name': 'markov-flex', 'hidden': self._hidden, 'state_size': self._state_size}

    def init_state(self):
        return jnp.zeros((self._state_size,))

    def init_params(self, key):
        key = jax.random.split(key, 8)
        return {
            'wstate_in': jax.random.normal(key[0], shape=(self._state_size, NCHAR)),
            'wstate_prev': jax.random.normal(key[1], shape=(self._state_size, self._state_size)),
            'bstate': jax.random.normal(key[2], shape=(self._state_size,)),

            'w': jax.random.normal(key[3], shape=(self._hidden, NCHAR)),
            'wprev': jax.random.normal(key[4], shape=(self._hidden, self._state_size)),
            'b': jax.random.normal(key[5], shape=(self._hidden,)),

            'wout': jax.random.normal(key[6], shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(key[7], shape=(NCHAR,)),
        }

    def step(self, params, state, c):
        new_state = (jnp.dot(params['wstate_in'], c) +
                     jnp.dot(params['wstate_prev'], state) +
                     params['bstate'])
        new_state = jax.nn.tanh(new_state)
        hidden = jnp.dot(params['w'], c) + jnp.dot(params['wprev'], state) + params['b']
        hidden = jax.nn.tanh(hidden)
        out = jnp.dot(params['wout'], hidden) + params['bout']
        return new_state, out


class RecurrentL1(Model):
    def __init__(self, hidden, activation=jax.nn.sigmoid):
        self._hidden = hidden
        self._activation = activation

    def serialize(self):
        return {'name': 'recurrent-l1', 'hidden': self._hidden}

    def init_params(self, key):
        key0, key1, key2, key3, key4 = jax.random.split(key, 5)
        return {
            'winp1': jax.random.normal(key0, shape=(self._hidden, NCHAR)),
            'b1': jax.random.normal(key1, shape=(self._hidden,)),
            'wstate1': jax.random.normal(key2, shape=(self._hidden, self._hidden)),
            'wout': jax.random.normal(key3, shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(key4, shape=(NCHAR,))
        }

    def init_state(self):
        return jnp.zeros((self._hidden,))

    def step(self, params, state, c):
        """One forward step in the recurrent network.
        Args:
            params: dictionary with model parameters
            state: array with the internal state, initially zero. shape = (256,)
            c: a single character represented as a one-of. shape = (256,)
        """
        state = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state) + params['b1']
        state = self._activation(state)
        # state = jax.nn.normalize(state)
        out = jnp.dot(params['wout'], state) + params['bout']
        return state, out

    def step_batch(self, params, sbatch, cbatch):
        sbatch = jnp.expand_dims(sbatch, 2)
        cbatch = jnp.expand_dims(cbatch, 2)
        b1 = jnp.expand_dims(params['b1'], 1)
        b1 = jnp.expand_dims(b1, 0)
        bout = jnp.expand_dims(params['bout'], 1)
        bout = jnp.expand_dims(bout, 0)
        winp1 = jnp.expand_dims(params['winp1'], 0)
        wstate1 = jnp.expand_dims(params['wstate1'], 0)
        sbatch = jnp.matmul(winp1, cbatch) + jnp.matmul(wstate1, sbatch) + b1
        sbatch = self._activation(sbatch)
        out = jnp.matmul(params['wout'], sbatch) + bout
        sbatch = sbatch.squeeze(axis=2)
        out = out.squeeze(axis=2)
        return sbatch, out


class RecurrentConv2(Model):
    """One layer with last two characters as inputs + recurrent layer."""

    def __init__(self, conv, hidden, activation=jax.nn.sigmoid):
        self._conv = conv
        self._hidden = hidden
        self._activation = activation

    def serialize(self):
        return {'name': 'recurrent-conv2', 'hidden': self._hidden, 'conv': self._conv}

    def init_params(self, key):
        keys = jax.random.split(key, 8)
        return {
            'winp': 0.1 * jax.random.normal(keys[0], shape=(self._conv, NCHAR)),
            'wprev': 0.1 * jax.random.normal(keys[1], shape=(self._conv, NCHAR)),
            'bconv': 0.1 * jax.random.normal(keys[2], shape=(self._conv,)),
            'wconv': 0.1 * jax.random.normal(keys[3], shape=(self._hidden, self._conv)),
            'whidden': 0.1 * jax.random.normal(keys[4], shape=(self._hidden, self._hidden)),
            'bhidden': 0.1 * jax.random.normal(keys[5], shape=(self._hidden,)),
            'wout': 0.1 * jax.random.normal(keys[6], shape=(NCHAR, self._hidden)),
            'bout': 0.1 * jax.random.normal(keys[7], shape=(NCHAR,))
        }

    def init_state(self):
        return {'h': jnp.zeros((self._hidden,)), 'prev': jnp.zeros((NCHAR,))}

    def step(self, params, state, c):
        prev = state['prev']
        hidden = state['h']

        conv = jnp.dot(params['winp'], c) + jnp.dot(params['wprev'], prev) + params['bconv']
        conv = jax.nn.tanh(conv)
        # conv /= jnp.sum(conv)

        new_hidden = jnp.dot(params['wconv'], conv) + jnp.dot(params['whidden'], hidden) + params['bhidden']
        new_hidden = jax.nn.tanh(new_hidden)

        out = jnp.dot(params['wout'], new_hidden) + params['bout']
        return {'h': new_hidden, 'prev': c}, out


class RecurrentL2(Model):
    def __init__(self, hidden, activation):
        self._hidden = hidden
        self._activation = activation

    def init_params(self, key):
        key0, key1, key2, key3, key4, key5, key6 = jax.random.split(key, 7)
        return {
            'winp1': jax.random.normal(key0, shape=(self._hidden, NCHAR)),
            'b1': jax.random.normal(key1, shape=(self._hidden,)),
            'wstate1': jax.random.normal(key2, shape=(self._hidden, self._hidden)),

            'b2': jax.random.normal(key3, shape=(self._hidden,)),
            'wstate2': jax.random.normal(key4, shape=(self._hidden, self._hidden)),

            'wout': jax.random.normal(key5, shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(key6, shape=(NCHAR,))
        }

    def init_state(self):
        return jnp.zeros((self._hidden,))

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with model parameters
            state: array with the internal state, initially zero. shape = (self._hidden,)
            c: a single character represented as a one-of. shape = (NCHAR,)
        """
        hidden = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state) + params['b1']
        hidden = jax.nn.sigmoid(hidden)

        state = jnp.dot(params['wstate2'], hidden) + params['b2']
        state = jax.nn.sigmoid(state)

        out = jnp.dot(params['wout'], state) + params['bout']
        return state, out


class RecurrentGRU(Model):
    def __init__(self, hidden, **kwargs):
        self._hidden = hidden

    def serialize(self):
        return {'name': 'recurrent-gru', 'hidden': self._hidden}

    def init_params(self, key):
        keys = jax.random.split(key, 16)
        params = {
            'wz': jax.random.normal(keys[0], shape=(self._hidden, NCHAR)),
            'pz': jax.random.normal(keys[11], shape=(self._hidden, NCHAR)),
            'uz': jax.random.normal(keys[1], shape=(self._hidden, self._hidden)),
            'bz': jax.random.normal(keys[2], shape=(self._hidden,)),

            'wr': jax.random.normal(keys[3], shape=(self._hidden, NCHAR)),
            'pr': jax.random.normal(keys[12], shape=(self._hidden, NCHAR)),
            'ur': jax.random.normal(keys[4], shape=(self._hidden, self._hidden)),
            'br': jax.random.normal(keys[5], shape=(self._hidden,)),

            'wh': jax.random.normal(keys[6], shape=(self._hidden, NCHAR)),
            'ph': jax.random.normal(keys[13], shape=(self._hidden, NCHAR)),
            'uh': jax.random.normal(keys[7], shape=(self._hidden, self._hidden)),
            'bh': jax.random.normal(keys[8], shape=(self._hidden,)),

            'w2': jax.random.normal(keys[14], shape=(self._hidden, self._hidden)),
            'b2': jax.random.normal(keys[15], shape=(self._hidden,)),

            'wout': jax.random.normal(keys[9], shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(keys[10], shape=(NCHAR,))
        }

        for k in params.keys():
            params[k] = 0.1 * params[k]

        return params

    def init_state(self):
        return {'h': jnp.zeros((self._hidden,)), 'prev': jnp.zeros((NCHAR,))}

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with model parameters
            state: array with the internal state, initially zero. shape = (256,)
            c: a single character represented as a one-of. shape = (256,)
        """
        h = state['h']
        prev = state['prev']

        z = jnp.dot(params['wz'], c) + jnp.dot(params['uz'], h) + jnp.dot(params['pz'], prev) + params['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(params['wr'], c) + jnp.dot(params['ur'], h) + jnp.dot(params['pr'], prev) + params['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(params['wh'], c) + jnp.dot(params['uh'], r * h) + jnp.dot(params['ph'], prev) + params['bh']
        hc = jax.nn.tanh(hc)

        hn = (1 - z) * h + z * hc

        t = jnp.dot(params['w2'], hn) + params['b2']
        t = jax.nn.sigmoid(t)

        out = jnp.dot(params['wout'], t) + params['bout']

        return {'h': hn, 'prev': c}, out

class ConvGru(Model):
    def __init__(self, hidden, conv, **kwargs):
        self._hidden = hidden
        self._conv = conv

    def serialize(self):
        return {'name': 'gru-conv', 'hidden': self._hidden, 'conv': self._conv}

    def init_params(self, key):
        keys = jax.random.split(key, 16)
        params = {
            'winp': 0.1 * jax.random.normal(keys[0], shape=(self._conv, NCHAR)),
            'wprev': 0.1 * jax.random.normal(keys[1], shape=(self._conv, NCHAR)),
            'bconv': 0.1 * jax.random.normal(keys[2], shape=(self._conv,)),

            'wz': jax.random.normal(keys[0], shape=(self._hidden, self._conv)),
            'pz': jax.random.normal(keys[11], shape=(self._hidden, self._conv)),
            'uz': jax.random.normal(keys[1], shape=(self._hidden, self._hidden)),
            'bz': jax.random.normal(keys[2], shape=(self._hidden,)),

            'wr': jax.random.normal(keys[3], shape=(self._hidden, self._conv)),
            'pr': jax.random.normal(keys[12], shape=(self._hidden, self._conv)),
            'ur': jax.random.normal(keys[4], shape=(self._hidden, self._hidden)),
            'br': jax.random.normal(keys[5], shape=(self._hidden,)),

            'wh': jax.random.normal(keys[6], shape=(self._hidden, self._conv)),
            'ph': jax.random.normal(keys[13], shape=(self._hidden, self._conv)),
            'uh': jax.random.normal(keys[7], shape=(self._hidden, self._hidden)),
            'bh': jax.random.normal(keys[8], shape=(self._hidden,)),

            'w2': jax.random.normal(keys[14], shape=(self._hidden, self._hidden)),
            'b2': jax.random.normal(keys[15], shape=(self._hidden,)),

            'wout': jax.random.normal(keys[9], shape=(NCHAR, self._hidden)),
            'bout': jax.random.normal(keys[10], shape=(NCHAR,))
        }

        for k in params.keys():
            params[k] = 0.1 * params[k]

        return params

    def init_state(self):
        return {'h': jnp.zeros((self._hidden,)), 'prev': jnp.zeros((NCHAR,)), 'prev2': jnp.zeros((NCHAR,))}

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with model parameters
            state: array with the internal state, initially zero. shape = (256,)
            c: a single character represented as a one-of. shape = (256,)
        """
        h = state['h']
        prev = state['prev']
        prev2 = state['prev2']

        conv = jnp.dot(params['winp'], c) + jnp.dot(params['wprev'], prev) + params['bconv']
        conv = jax.nn.tanh(conv)

        conv_prev = jnp.dot(params['winp'], prev) + jnp.dot(params['wprev'], prev2) + params['bconv']
        conv_prev = jax.nn.tanh(conv_prev)

        z = jnp.dot(params['wz'], conv) + jnp.dot(params['uz'], h) + jnp.dot(params['pz'], conv_prev) + params['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(params['wr'], conv) + jnp.dot(params['ur'], h) + jnp.dot(params['pr'], conv_prev) + params['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(params['wh'], conv) + jnp.dot(params['uh'], r * h) + jnp.dot(params['ph'], conv_prev) + params['bh']
        hc = jax.nn.tanh(hc)

        hn = (1 - z) * h + z * hc

        t = jnp.dot(params['w2'], hn) + params['b2']
        t = jax.nn.sigmoid(t)

        out = jnp.dot(params['wout'], t) + params['bout']

        return {'h': hn, 'prev': c, 'prev2': prev}, out

MODELS = {
    'equal': Equal,
    'freq': Freq,
    'markov1': Markov1,
    'markov2': Markov2,
    'markov2true': Markov2True,
    'markov-flex': MarkovFlex,
    'recurrent-l1': RecurrentL1,
    'recurrent-conv2': RecurrentConv2,
    'recurrent-gru': RecurrentGRU,
}

def build(spec):
    cls = MODELS[spec['name']]
    return cls(**spec)

