import jax
import jax.numpy as jnp

import model
from model import Model, NCHAR
import layers

class Equal(Model):
    def name(self):
        return 'equal'

    def serialize(self):
        return {'name': 'equal'}

    def step(self, params, state, c):
        return state, jnp.ones_like(c)


class Freq(Model):
    def name(self):
        return 'freq'

    def serialize(self):
        return {'name': 'freq'}

    def init_params(self, key):
        return {'b': jax.random.normal(key, shape=(NCHAR,))}

    def step(self, params, state, c):
        out = params['b']
        return state, out


class Markov1(Model):
    """Logistic regression on the previous character."""

    def __init__(self):
        super().__init__()
        self.layer = layers.FeedForward(NCHAR, NCHAR)

    def name(self):
        return 'markov1'

    def serialize(self):
        return {'name': 'markov1'}

    def init_state(self):
        return None

    def init_params(self, key):
        return self.layer.init_params(key)

    def step(self, params, state, c):
        out = self.layer.step(params, c)
        return state, out


class Markov(Model):
    """Logistic regression on a number of previous characters."""

    def __init__(self, suffix):
        super().__init__()
        assert suffix > 1
        self._suffix = suffix
        self.layer = layers.FeedForward(self._suffix * NCHAR, NCHAR)

    def name(self):
        return f'markov-{self._suffix}'

    def serialize(self):
        return {'name': 'markov', 'suffix': self._suffix}

    def init_state(self):
        return model.init_suffix(self._suffix)

    def init_params(self, key):
        return self.layer.init_params(key)

    def step(self, params, state, c):
        suffix = model.stack_suffix(state, c)
        out = self.layer.step(params, suffix)
        return suffix[1:], out


class Forward1(Model):
    def __init__(self, suffix, hidden=128):
        super().__init__()
        self._suffix = suffix
        self._hidden = hidden
        self.in_layer = layers.FeedForward(self._suffix * NCHAR, self._hidden)
        self.out_layer = layers.FeedForward(self._hidden, NCHAR)

    def name(self):
        return f'forward1-{self._suffix}-{self._hidden}'

    def serialize(self):
        return {'name': 'forward1', 'suffix': self._suffix, 'hidden': self._hidden}

    def init_state(self):
        return model.init_suffix(self._suffix)

    def init_params(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'input': self.in_layer.init_params(key0),
            'output': self.out_layer.init_params(key1),
        }

    def step(self, params, state, c):
        suffix = model.stack_suffix(state, c)
        hidden = self.in_layer.step(params['input'], suffix)
        out = self.out_layer.step(params['output'], hidden)
        return suffix[1:], out


class Recurrent1(Model):
    def __init__(self, hidden):
        self._hidden = hidden
        self.recurrent_layer = layers.Recurrent(NCHAR, self._hidden)
        self.out_layer = layers.FeedForward(self._hidden, NCHAR)
        self.name = f'recurrent1-{self._hidden}'

    def serialize(self):
        return {'name': 'recurrent1', 'hidden': self._hidden}

    def init_state(self):
        return self.recurrent_layer.init_state()

    def init_params(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'recurrent': self.recurrent_layer.init_params(key0),
            'out': self.out_layer.init_params(key1)
        }

    def step(self, params, state, c):
        state = self.recurrent_layer.step(params['recurrent'], c, state)
        state = jax.nn.sigmoid(state)
        out = self.out_layer.step(params['out'], state)
        return state, out


class Recurrent2(Model):
    """Recurrent layer + extra output layer."""

    def __init__(self, hidden, out):
        self._hidden = hidden
        self._out = out
        self.recurrent_layer = layers.Recurrent(NCHAR, self._hidden)
        self.out1_layer = layers.FeedForward(self._hidden, out)
        self.out2_layer = layers.FeedForward(out, NCHAR)
        self.name = f'recurrent2-{self._hidden}-{self._out}'

    def serialize(self):
        return {'name': 'recurrent2', 'hidden': self._hidden, 'out': self._out}

    def init_state(self):
        return self.recurrent_layer.init_state()

    def init_params(self, key):
        keys = jax.random.split(key)
        return {
            'recurrent': self.recurrent_layer.init_params(keys[0]),
            'out1': self.out1_layer.init_params(keys[1]),
            'out2': self.out2_layer.init_params(keys[2]),
        }

    def step(self, params, state, c):
        state = self.recurrent_layer.step(params['recurrent'], c, state)
        state = jax.nn.sigmoid(state)
        out1 = self.out1_layer.step(params['out1'], state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(params['out2'], out1)
        return state, out2


class Recurrent3(Model):
    """Recurrent layer + extra output layer."""

    def __init__(self, suffix, input, hidden, output):
        self._suffix = suffix
        self._input = input
        self._hidden = hidden
        self._out = output
        self.in_layer = layers.FeedForward(NCHAR*suffix, self._input)
        self.recurrent_layer = layers.Recurrent(self._input, self._hidden)
        self.out1_layer = layers.FeedForward(self._hidden, output)
        self.out2_layer = layers.FeedForward(output, NCHAR)

    def name(self):
        return f'recurrent3-{self._input}-{self._hidden}-{self._out}'

    def serialize(self):
        return {
            'name': 'recurrent3',
            'suffix': self._suffix,
            'input': self._input,
            'hidden': self._hidden,
            'output': self._out,
        }

    def init_state(self):
        return {
            'suffix': model.init_suffix(self._suffix),
            'state': self.recurrent_layer.init_state(),
        }

    def init_params(self, key):
        keys = jax.random.split(key, 4)
        return {
            'in': self.in_layer.init_params(keys[0]),
            'recurrent': self.recurrent_layer.init_params(keys[1]),
            'out1': self.out1_layer.init_params(keys[2]),
            'out2': self.out2_layer.init_params(keys[3]),
        }

    def step(self, params, state, c):
        suffix = model.stack_suffix(state['suffix'], c)
        input = self.in_layer.step(params['in'], suffix)
        input = jax.nn.sigmoid(input)
        state = self.recurrent_layer.step(params['recurrent'], input, state['state'])
        state = jax.nn.sigmoid(state)
        out1 = self.out1_layer.step(params['out1'], state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(params['out2'], out1)
        new_state = {
            'suffix': suffix[1:],
            'state': state,
        }
        return new_state, out2


class Recurrent4(Model):
    """Recurrent layer + extra output layer.

    Two convolution-like input layers.
    """

    def __init__(self, input1, input2, hidden, output):
        self._input1 = input1
        self._input2 = input2
        self._hidden = hidden
        self._output = output
        self.conv1_layer = layers.Convolution(NCHAR, 2, input1)
        self.conv2_layer = layers.Convolution(input1, 2, input2)
        self.recurrent_layer = layers.Recurrent(input2, self._hidden)
        self.out1_layer = layers.FeedForward(self._hidden, output)
        self.out2_layer = layers.FeedForward(output, NCHAR)
        self.name = f'recurrent4-{self._input1}-{self._input2}-{self._hidden}-{self._output}'

    def serialize(self):
        return {
            'name': 'recurrent4',
            'input1': self._input1,
            'input2': self._input2,
            'hidden': self._hidden,
            'output': self._output,
        }

    def init_state(self):
        return {
            'suffix': model.init_suffix(3),
            'state': self.recurrent_layer.init_state(),
        }

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'conv1': self.conv1_layer.init_params(keys[0]),
            'conv2': self.conv2_layer.init_params(keys[1]),
            'recurrent': self.recurrent_layer.init_params(keys[2]),
            'out1': self.out1_layer.init_params(keys[3]),
            'out2': self.out2_layer.init_params(keys[4]),
        }

    def step(self, params, state, c):
        suffix = model.stack_suffix(state['suffix'], c)
        in1 = self.conv1_layer.step(params['conv1'], suffix)
        in1 = jax.nn.sigmoid(in1)
        in2 = self.conv2_layer.step(params['conv2'], in1)
        in2 = jax.nn.sigmoid(in2)
        state = self.recurrent_layer.step(params['recurrent'], in2, state['state'])
        state = jax.nn.sigmoid(state)
        out1 = self.out1_layer.step(params['out1'], state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(params['out2'], out1)
        new_state = {
            'suffix': suffix[1:],
            'state': state,
        }
        return new_state, out2


class RecGru(Model):
    def __init__(self, conv, rec, gru, skip_rec=False):
        self._conv = conv
        self._rec = rec
        self._gru = gru
        self._skip_rec = skip_rec

        self.conv_layer = layers.FeedForward(2*NCHAR, self._conv)
        self.recurrent_layer = layers.Recurrent(self._conv, self._rec)
        gru_input = self._rec + self._conv if skip_rec else self._rec
        self.gru_layer = layers.Gru(gru_input, self._gru)
        self.out1_layer = layers.FeedForward(self._gru, NCHAR)
        # self.out2_layer = layers.FeedForward(out, NCHAR)

        skip = 's' if self._skip_rec else ''

        self.name = f'rec-gru-{conv}-{rec}{skip}-{gru}'

    def serialize(self):
        return {'name': 'rec-gru', 'conv': self._conv, 'rec': self._rec, 'gru': self. _gru}

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'conv': self.conv_layer.init_params(keys[0]),
            'rec': self.recurrent_layer.init_params(keys[1]),
            'gru': self.gru_layer.init_params(keys[2]),
            'out1': self.out1_layer.init_params(keys[3]),
            # 'out2': self.out2_layer.init_params(keys[4]),
        }

    def init_state(self):
        return {
            'suffix': model.init_suffix(2),
            'rec': self.recurrent_layer.init_state(),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, params, state, c):
        suffix = model.stack_suffix(state['suffix'], c)
        inp = self.conv_layer.step(params['conv'], suffix)
        inp = jax.nn.sigmoid(inp)
        rec_state = self.recurrent_layer.step(params['rec'], inp, state['rec'])
        rec_state = jax.nn.sigmoid(rec_state)
        if self._skip_rec:
            gru_input = jnp.concatenate([inp, rec_state])
        else:
            gru_input = rec_state
        gru_state = self.gru_layer.step(params['gru'], gru_input, state['gru'])
        out1 = self.out1_layer.step(params['out1'], gru_state)
        # out1 = jax.nn.sigmoid(out1)
        # out2 = self.out2_layer.step(params['out2'], out1)
        new_state = {
            'suffix': suffix[1:],
            'rec': rec_state,
            'gru': gru_state,
        }
        return new_state, out1


class RecGru2(Model):
    def __init__(self, rec, gru, out, skip_rec=False):
        self._rec = rec
        self._gru = gru
        self._out = out
        self._skip_rec = skip_rec

        self.recurrent_layer = layers.Recurrent(NCHAR, self._rec)
        gru_input = (self._rec + NCHAR) if skip_rec else self._rec
        self.gru_layer = layers.Gru(gru_input, self._gru)
        self.out1_layer = layers.FeedForward(self._gru, out)
        self.out2_layer = layers.FeedForward(out, NCHAR)

        skip = 's' if self._skip_rec else ''

        self.name = f'rec-gru2-{rec}{skip}-{gru}-{out}'

    def serialize(self):
        return {'name': 'rec-gru2', 'out': self._out, 'rec': self._rec, 'gru': self. _gru, 'skip_rec': self._skip_rec}

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'rec': self.recurrent_layer.init_params(keys[1]),
            'gru': self.gru_layer.init_params(keys[2]),
            'out1': self.out1_layer.init_params(keys[3]),
            'out2': self.out2_layer.init_params(keys[4]),
        }

    def init_state(self):
        return {
            'rec': self.recurrent_layer.init_state(),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, params, state, c):
        rec_state = self.recurrent_layer.step(params['rec'], c, state['rec'])
        rec_state = jax.nn.sigmoid(rec_state)
        if self._skip_rec:
            gru_input = jnp.concatenate([c, rec_state])
        else:
            gru_input = rec_state
        gru_state = self.gru_layer.step(params['gru'], gru_input, state['gru'])
        out1 = self.out1_layer.step(params['out1'], gru_state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(params['out2'], out1)
        new_state = {
            'rec': rec_state,
            'gru': gru_state,
        }
        return new_state, out2


class ConvGru2(Model):
    def __init__(self, conv, gru):
        self._conv = conv
        self._gru = gru

        self.conv_layer = layers.FeedForward(2*NCHAR, conv)
        self.gru_layer = layers.Gru(2*conv, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'conv-gru2-{conv}-{gru}'

    def serialize(self):
        return {'name': 'conv-gru2', 'conv': self._conv, 'gru': self. _gru}

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'conv': self.conv_layer.init_params(keys[0]),
            'gru': self.gru_layer.init_params(keys[1]),
            'out': self.out_layer.init_params(keys[2]),
        }

    def init_state(self):
        return {
            'suffix': model.init_suffix(3),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, params, state, c):
        suffix = model.stack_suffix(state['suffix'], c)

        conv_prev = self.conv_layer.step(params['conv'], jnp.concatenate(suffix[:-1]))
        conv = self.conv_layer.step(params['conv'], jnp.concatenate(suffix[1:]))
        conv_all = jnp.concatenate([conv_prev, conv])
        conv_all = jax.nn.relu(conv_all)

        gru_state = self.gru_layer.step(params['gru'], conv_all, state['gru'])

        out = self.out_layer.step(params['out'], gru_state)
        new_state = {
            'suffix': suffix[1:],
            'gru': gru_state,
        }
        return new_state, out


class ConvGru(Model):
    def __init__(self, hidden, conv, **kwargs):
        self._hidden = hidden
        self._conv = conv

    def serialize(self):
        return {'name': 'conv-gru', 'hidden': self._hidden, 'conv': self._conv}

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


class Conv3Gru(Model):
    def __init__(self, hidden, conv, **kwargs):
        self._hidden = hidden
        self._conv = conv

    def serialize(self):
        return {'name': 'conv3-gru', 'hidden': self._hidden, 'conv': self._conv}

    def init_params(self, key):
        keys = jax.random.split(key, 16)
        params = {
            'winp': 0.1 * jax.random.normal(keys[0], shape=(self._conv, 3*NCHAR)),
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
        return {'h': jnp.zeros((self._hidden,)), 'suffix': jnp.zeros((3, NCHAR))}

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with model parameters
            state: state dictionary
            c: a single character represented as a one-of. shape = (256,)
        """
        h = state['h']
        suffix = state['suffix']
        suffix = jnp.vstack((suffix, c.reshape((1, -1))))

        suffix1 = suffix[0:3]
        suffix2 = suffix[1:4]

        conv1 = jnp.dot(params['winp'], suffix1.flatten()) + params['bconv']
        conv1 = jax.nn.tanh(conv1)
        conv2 = jnp.dot(params['winp'], suffix2.flatten()) + params['bconv']
        conv2 = jax.nn.tanh(conv2)

        z = jnp.dot(params['wz'], conv1) + jnp.dot(params['uz'], h) + jnp.dot(params['pz'], conv2) + params['bz']
        z = jax.nn.sigmoid(z)

        r = jnp.dot(params['wr'], conv1) + jnp.dot(params['ur'], h) + jnp.dot(params['pr'], conv2) + params['br']
        r = jax.nn.sigmoid(r)

        hc = jnp.dot(params['wh'], conv1) + jnp.dot(params['uh'], r * h) + jnp.dot(params['ph'], conv2) + params['bh']
        hc = jax.nn.tanh(hc)

        hn = (1 - z) * h + z * hc

        t = jnp.dot(params['w2'], hn) + params['b2']
        t = jax.nn.sigmoid(t)

        out = jnp.dot(params['wout'], t) + params['bout']

        return {'h': hn, 'suffix': suffix2}, out


class Gru2(Model):
    def __init__(self, gru):
        self._gru = gru

        self.gru_layer = layers.Gru(2*NCHAR, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'gru2-{gru}'

    def serialize(self):
        return {'name': 'gru2', 'gru': self. _gru}

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'gru': self.gru_layer.init_params(keys[1]),
            'out': self.out_layer.init_params(keys[2]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, params, state, c):
        prev = state['prev']

        gru_input = jnp.concatenate([prev, c])
        gru_state = self.gru_layer.step(params['gru'], gru_input, state['gru'])
        out = self.out_layer.step(params['out'], gru_state)

        new_state = {
            'prev': c,
            'gru': gru_state,
        }
        return new_state, out


class GruGru(Model):
    def __init__(self, gru1, gru2):
        self._gru1 = gru1
        self._gru2 = gru2

        self.gru1_layer = layers.Gru(2*NCHAR, gru1)
        self.gru2_layer = layers.Gru(gru1, gru2)
        self.out_layer = layers.FeedForward(gru2, NCHAR)

        self.name = f'gru-gru-{gru1}-{gru2}'

    def serialize(self):
        return {'name': 'gru-gru', 'gru1': self._gru1, 'gru2': self._gru2}

    def init_params(self, key):
        keys = jax.random.split(key, 5)
        return {
            'gru1': self.gru1_layer.init_params(keys[1]),
            'gru2': self.gru2_layer.init_params(keys[2]),
            'out': self.out_layer.init_params(keys[3]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'gru1': self.gru1_layer.init_state(),
            'gru2': self.gru2_layer.init_state(),
        }

    def step(self, params, state, c):
        prev = state['prev']

        gru1_input = jnp.concatenate([prev, c])
        gru1_state = self.gru1_layer.step(params['gru1'], gru1_input, state['gru1'])

        gru2_state = self.gru2_layer.step(params['gru2'], gru1_state, state['gru2'])

        out = self.out_layer.step(params['out'], gru2_state)

        new_state = {
            'prev': c,
            'gru1': gru1_state,
            'gru2': gru2_state,
        }
        return new_state, out


MODELS = {
    'equal': Equal,
    'freq': Freq,
    'markov1': Markov1,
    'markov': Markov,
    'recurrent1': Recurrent1,
    'recurrent2': Recurrent2,
    'recurrent3': Recurrent3,
    'conv-gru': ConvGru,
    'conv3-gru': ConvGru,
    'conv-gru2': ConvGru2,
}

def build(spec):
    cls = MODELS[spec['name']]
    del spec['name']
    return cls(**spec)

