import jax
import jax.numpy as jnp

import model
from model import Model, NCHAR
import layers


class Equal(Model):
    name = 'equal'

    def step(self, weights, state, c):
        return state, jnp.ones_like(c)


class Freq(Model):
    name = 'freq'

    def init_weights(self, key):
        return {'b': jax.random.normal(key, shape=(NCHAR,))}

    def step(self, weights, state, c):
        out = weights['b']
        return state, out


class Markov1(Model):
    """Logistic regression on the previous character."""
    name = 'markov1'

    def __init__(self):
        super().__init__()
        self.layer = layers.FeedForward(NCHAR, NCHAR)

    def init_weights(self, key):
        return self.layer.init_weights(key)

    def step(self, weights, state, c):
        out = self.layer.step(weights, c)
        return state, out


class Markov(Model):
    """Logistic regression from the suffix of a given length."""

    def __init__(self, suffix):
        assert suffix > 1
        super().__init__(suffix=suffix)
        self.suffix_layer = layers.Suffix(NCHAR, self.suffix)
        self.layer = layers.FeedForward(self.suffix * NCHAR, NCHAR)
        self.name = f'markov-{suffix}'

    def init_state(self):
        return self.suffix_layer.init_state()

    def init_weights(self, key):
        return self.layer.init_weights(key)

    def step(self, weights, state, c):
        suffix_state, suffix = self.suffix_layer.step(None, c, state)
        out = self.layer.step(weights, suffix)
        return suffix_state, out


class Forward(Model):
    def __init__(self, suffix, hidden=128):
        super().__init__()
        self._suffix = suffix
        self._hidden = hidden
        self.in_layer = layers.FeedForward(self._suffix * NCHAR, self._hidden)
        self.out_layer = layers.FeedForward(self._hidden, NCHAR)

        self.name = f'forward-{self._suffix}-{self._hidden}'

    def serialize(self):
        return {'name': 'forward', 'suffix': self._suffix, 'hidden': self._hidden}

    def init_state(self):
        return model.init_suffix(self._suffix)

    def init_weights(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'input': self.in_layer.init_weights(key0),
            'output': self.out_layer.init_weights(key1),
        }

    def step(self, weights, state, c):
        suffix = model.stack_suffix(state, c)
        inp = self.in_layer.step(weights['input'], suffix)
        inp = jnp.tanh(inp)
        out = self.out_layer.step(weights['output'], inp)
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

    def init_weights(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'recurrent': self.recurrent_layer.init_weights(key0),
            'out': self.out_layer.init_weights(key1)
        }

    def step(self, weights, state, c):
        state = self.recurrent_layer.step(weights['recurrent'], c, state)
        state = jax.nn.sigmoid(state)
        out = self.out_layer.step(weights['out'], state)
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

    def init_weights(self, key):
        keys = jax.random.split(key)
        return {
            'recurrent': self.recurrent_layer.init_weights(keys[0]),
            'out1': self.out1_layer.init_weights(keys[1]),
            'out2': self.out2_layer.init_weights(keys[2]),
        }

    def step(self, weights, state, c):
        state = self.recurrent_layer.step(weights['recurrent'], c, state)
        state = jax.nn.sigmoid(state)
        out1 = self.out1_layer.step(weights['out1'], state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(weights['out2'], out1)
        return state, out2


class Recurrent3(Model):
    """Suffix + recurrent layer + extra output layer."""

    def __init__(self, suffix, input, hidden, output):
        self._suffix = suffix
        self._input = input
        self._hidden = hidden
        self._out = output
        self.in_layer = layers.FeedForward(NCHAR*suffix, self._input, activation=jax.nn.sigmoid)
        self.recurrent_layer = layers.Recurrent(self._input, self._hidden, activation=jax.nn.sigmoid)
        self.out1_layer = layers.FeedForward(self._hidden, output, activation=jax.nn.sigmoid)
        self.out2_layer = layers.FeedForward(self._out, NCHAR)
        self.name = f'recurrent3-{self._input}-{self._hidden}-{self._out}'

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
            'recurrent': self.recurrent_layer.init_state(),
        }

    def init_weights(self, key):
        keys = jax.random.split(key, 4)
        return {
            'in': self.in_layer.init_weights(keys[0]),
            'recurrent': self.recurrent_layer.init_weights(keys[1]),
            'out1': self.out1_layer.init_weights(keys[2]),
            'out2': self.out2_layer.init_weights(keys[3]),
        }

    def step(self, weights, state, c):
        suffix = model.stack_suffix(state['suffix'], c)
        input = self.in_layer.step(weights['in'], suffix)
        state = self.recurrent_layer.step(weights['recurrent'], input, state['recurrent'])
        out1 = self.out1_layer.step(weights['out1'], state)
        out2 = self.out2_layer.step(weights['out2'], out1)
        new_state = {
            'suffix': suffix[1:],
            'recurrent': state,
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

    def init_weights(self, key):
        keys = jax.random.split(key, 5)
        return {
            'conv1': self.conv1_layer.init_weights(keys[0]),
            'conv2': self.conv2_layer.init_weights(keys[1]),
            'recurrent': self.recurrent_layer.init_weights(keys[2]),
            'out1': self.out1_layer.init_weights(keys[3]),
            'out2': self.out2_layer.init_weights(keys[4]),
        }

    def step(self, weights, state, c):
        suffix = model.stack_suffix(state['suffix'], c)
        in1 = self.conv1_layer.step(weights['conv1'], suffix)
        in1 = jax.nn.sigmoid(in1)
        in2 = self.conv2_layer.step(weights['conv2'], in1)
        in2 = jax.nn.sigmoid(in2)
        state = self.recurrent_layer.step(weights['recurrent'], in2, state['state'])
        state = jax.nn.sigmoid(state)
        out1 = self.out1_layer.step(weights['out1'], state)
        out1 = jax.nn.sigmoid(out1)
        out2 = self.out2_layer.step(weights['out2'], out1)
        new_state = {
            'suffix': suffix[1:],
            'state': state,
        }
        return new_state, out2


class ConvGru1(Model):
    def __init__(self, conv, gru):
        """2-convolution over 3-char suffix + GRU"""
        self._conv = conv
        self._gru = gru

        self.suffix_layer = layers.Suffix(NCHAR, 3)
        self.conv_layer = layers.FeedForward(2*NCHAR, conv)
        self.gru_layer = layers.Gru(2*conv, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'convgru1-{conv}-{gru}'

    def serialize(self):
        return {'name': 'convgru1', 'conv': self._conv, 'gru': self._gru}

    def init_weights(self, key):
        keys = jax.random.split(key, 3)
        return {
            'conv': self.conv_layer.init_weights(keys[0]),
            'gru': self.gru_layer.init_weights(keys[1]),
            'out': self.out_layer.init_weights(keys[2]),
        }

    def init_state(self):
        return {
            'suffix': self.suffix_layer.init_state(),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, weights, state, c):
        suffix_state, suffix = self.suffix_layer.step(None, c, state['suffix'])

        gru_input1 = self.conv_layer.step(weights['conv'], suffix[0:2])
        gru_input2 = self.conv_layer.step(weights['conv'], suffix[1:3])
        gru_input = jnp.concatenate([gru_input1, gru_input2])
        gru_input = jnp.tanh(gru_input)

        gru_out = self.gru_layer.step(weights['gru'], gru_input, state['gru'])

        out = self.out_layer.step(weights['out'], gru_out)
        new_state = {'suffix': suffix_state, 'gru': gru_out}
        return new_state, out


class ConvGru3(Model):
    def __init__(self, suffix, conv, gru):
        """Convolution over X-char suffix + GRU"""
        self._suffix = suffix
        self._conv = conv
        self._gru = gru

        self.suffix_layer = layers.Suffix(NCHAR, suffix)
        self.conv_layer = layers.Convolution(NCHAR, 2, conv, activation=jax.nn.sigmoid)
        self.gru_layer = layers.Gru((suffix - 1)*conv, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'convgru3-{suffix}-{conv}-{gru}'

    def serialize(self):
        return {'name': 'convgru3', 'suffix': self._suffix, 'conv': self._conv, 'gru': self._gru}

    def init_weights(self, key):
        keys = jax.random.split(key, 3)
        return {
            'conv': self.conv_layer.init_weights(keys[0]),
            'gru': self.gru_layer.init_weights(keys[1]),
            'out': self.out_layer.init_weights(keys[2]),
        }

    def init_state(self):
        return {
            'suffix': self.suffix_layer.init_state(),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, weights, state, c):
        suffix_state, suffix = self.suffix_layer.step(None, c, state['suffix'])

        conv = self.conv_layer.step(weights['conv'], suffix)
        conv = conv.flatten()

        gru_out = self.gru_layer.step(weights['gru'], conv, state['gru'])

        out = self.out_layer.step(weights['out'], gru_out)
        new_state = {'suffix': suffix_state, 'gru': gru_out}
        return new_state, out


class Conv3Gru(Model):
    def __init__(self, inp, gru):
        """Fully connected over the last 3 chars + GRU"""
        self._inp = inp
        self._gru = gru

        self.suffix_layer = layers.Suffix(NCHAR, 3)
        self.inp_layer = layers.FeedForward(3*NCHAR, inp, activation=jax.nn.sigmoid)
        self.gru_layer = layers.Gru(inp, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'conv3gru-{inp}-{gru}'

    def serialize(self):
        return {'name': 'conv3gru', 'inp': self._inp, 'gru': self._gru}

    def init_weights(self, key):
        keys = jax.random.split(key, 3)
        return {
            'inp': self.inp_layer.init_weights(keys[0]),
            'gru': self.gru_layer.init_weights(keys[1]),
            'out': self.out_layer.init_weights(keys[2]),
        }

    def init_state(self):
        return {
            'suffix': self.suffix_layer.init_state(),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, weights, state, c):
        suffix_state, suffix = self.suffix_layer.step(None, c, state['suffix'])

        inp = self.inp_layer.step(weights['inp'], suffix)

        gru_out = self.gru_layer.step(weights['gru'], inp, state['gru'])

        out = self.out_layer.step(weights['out'], gru_out)
        new_state = {'suffix': suffix_state, 'gru': gru_out}
        return new_state, out


class Gru2(Model):
    def __init__(self, gru):
        self._gru = gru

        self.gru_layer = layers.Gru(2*NCHAR, gru)
        self.out_layer = layers.FeedForward(gru, NCHAR)

        self.name = f'gru2-{gru}'

    def serialize(self):
        return {'name': 'gru2', 'gru': self. _gru}

    def init_weights(self, key):
        keys = jax.random.split(key, 5)
        return {
            'gru': self.gru_layer.init_weights(keys[1]),
            'out': self.out_layer.init_weights(keys[2]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'gru': self.gru_layer.init_state(),
        }

    def step(self, weights, state, c):
        prev = state['prev']

        gru_input = jnp.concatenate([prev, c])
        gru_state = self.gru_layer.step(weights['gru'], gru_input, state['gru'])
        out = self.out_layer.step(weights['out'], gru_state)

        new_state = {
            'prev': c,
            'gru': gru_state,
        }
        return new_state, out


class GruGru(Model):
    def __init__(self, gru1, gru2, skip=False):
        self._gru1 = gru1
        self._gru2 = gru2
        self._skip = skip

        self.gru1_layer = layers.Gru(2*NCHAR, gru1)
        self.gru2_layer = layers.Gru(gru1, gru2)
        self.out_layer = layers.FeedForward(gru2, NCHAR)

        self.name = f'grugru-{gru1}-{gru2}{s}'

    def serialize(self):
        return {'name': 'grugru', 'gru1': self._gru1, 'gru2': self._gru2, 'skip': self._skip}

    def init_weights(self, key):
        keys = jax.random.split(key, 5)
        return {
            'gru1': self.gru1_layer.init_weights(keys[1]),
            'gru2': self.gru2_layer.init_weights(keys[2]),
            'out': self.out_layer.init_weights(keys[3]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'gru1': self.gru1_layer.init_state(),
            'gru2': self.gru2_layer.init_state(),
        }

    def step(self, weights, state, c):
        prev = state['prev']

        gru1_input = jnp.concatenate([prev, c])
        gru1_state = self.gru1_layer.step(weights['gru1'], gru1_input, state['gru1'])

        if self._skip:
            gru2_input = gru1_state + gru1_input
        else:
            gru2_input = gru1_state

        gru2_state = self.gru2_layer.step(weights['gru2'], gru2_input, state['gru2'])

        if self._skip:
            out_input = gru2_state + gru2_input
        else:
            out_input = gru2_state

        out = self.out_layer.step(weights['out'], out_input)

        new_state = {
            'prev': c,
            'gru1': gru1_state,
            'gru2': gru2_state,
        }
        return new_state, out


class Lstm2(Model):
    def __init__(self, lstm):
        self._lstm = lstm

        self.lstm_layer = layers.Lstm(2*NCHAR, lstm)
        self.out_layer = layers.FeedForward(lstm, NCHAR)

        self.name = f'lstm2-{lstm}'

    def serialize(self):
        return {'name': 'lstm2', 'lstm': self._lstm}

    def init_weights(self, key):
        keys = jax.random.split(key, 2)
        return {
            'lstm': self.lstm_layer.init_weights(keys[0]),
            'out': self.out_layer.init_weights(keys[1]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'lstm': self.lstm_layer.init_state(),
        }

    def step(self, weights, state, c):
        prev = state['prev']

        lstm_input = jnp.concatenate([prev, c])
        lstm_state = self.lstm_layer.step(weights['lstm'], lstm_input, state['lstm'])
        out = self.out_layer.step(weights['out'], lstm_state['h'])

        new_state = {
            'prev': c,
            'lstm': lstm_state,
        }
        return new_state, out


class LLstm(Model):
    def __init__(self, lstm1, lstm2=512, skip=False):
        self._lstm1 = lstm1
        self._lstm2 = lstm2

        self.lstm1_layer = layers.Lstm(2*NCHAR, lstm1)
        self.lstm2_layer = layers.Lstm(lstm1, lstm2)
        self.out_layer = layers.FeedForward(lstm2, NCHAR)
        self.skip = skip

        if self.skip:
            self.name = f'llstm-{lstm1}-{lstm2}-skip'
        else:
            self.name = f'llstm-{lstm1}-{lstm2}'

    def serialize(self):
        return {'name': 'llstm', 'lstm1': self._lstm1, 'lstm2': self._lstm2}

    def init_weights(self, key):
        keys = jax.random.split(key, 3)
        return {
            'lstm1': self.lstm1_layer.init_weights(keys[0]),
            'lstm2': self.lstm2_layer.init_weights(keys[1]),
            'out': self.out_layer.init_weights(keys[2]),
        }

    def init_state(self):
        return {
            'prev': jnp.zeros((NCHAR,)),
            'lstm1': self.lstm1_layer.init_state(),
            'lstm2': self.lstm2_layer.init_state(),
        }

    def step(self, weights, state, c):
        prev = state['prev']

        lstm1_input = jnp.concatenate([prev, c])
        lstm1_state = self.lstm1_layer.step(weights['lstm1'], lstm1_input, state['lstm1'])

        if self.skip:
            lstm2_input = lstm1_state['h'] + lstm1_input
        else:
            lstm2_input = lstm1_state['h']

        lstm2_state = self.lstm2_layer.step(weights['lstm2'], lstm2_input, state['lstm2'])

        out = self.out_layer.step(weights['out'], lstm2_state['h'])

        new_state = {
            'prev': c,
            'lstm1': lstm1_state,
            'lstm2': lstm2_state,
        }
        return new_state, out


MODELS = {
    'equal': Equal,
    'freq': Freq,
    'markov1': Markov1,
    'markov': Markov,
    'forward': Forward,
    'recurrent1': Recurrent1,
    'recurrent2': Recurrent2,
    'recurrent3': Recurrent3,
    'lstm2': Lstm2,
    'llstm': LLstm,
    'grugru': GruGru,
    'convgru1': ConvGru1,
    'convgru3': ConvGru3,
    'conv3gru': Conv3Gru,
}

def build(spec):
    cls = MODELS[spec['name']]
    del spec['name']
    return cls(**spec)


def parse(full_name):
    parts = full_name.split('-')
    cls = MODELS[parts[0]]
    args = []
    kwargs = {}
    for part in parts[1:]:
        try:
            p = int(part)
            args.append(p)
        except ValueError:
            kwargs[part] = True

    return cls(*args, **kwargs)
