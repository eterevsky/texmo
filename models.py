import jax
import jax.numpy as jnp

from model import Model, NCHAR


class Equal(Model):
    def step(self, params, state, c):
        return state, jnp.ones_like(c)


class Freq(Model):
    def init_params(self, key):
        return {'b': jax.random.normal(key, shape=(NCHAR,))}

    def step(self, params, state, c):
        out = params['b']
        return state, out


class Markov(Model):
    def init_params(self, key):
        key0, key1 = jax.random.split(key)
        return {
            'w': jax.random.normal(key0, shape=(NCHAR, NCHAR)),
            'b': jax.random.normal(key1, shape=(NCHAR,)),
        }

    def step(self, params, state, c):
        out = jnp.dot(params['w'], c) + params['b']
        return state, out


class Markov2(Model):
    def __init__(self):
        self._hidden = 128

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
        new_state = jnp.dot(params['wstate_in'], c) + jnp.dot(params['wstate_prev'], state) + params['bstate']
        new_state = jax.nn.tanh(state)
        hidden = jnp.dot(params['w'], c) + jnp.dot(params['wprev'], state) + params['b']
        hidden = jax.nn.tanh(hidden)
        out = jnp.dot(params['wout'], hidden) + params['bout']
        return new_state, out


class RecurrentL1(Model):
    def __init__(self, hidden, activation):
        self._hidden = hidden
        self._activation = activation

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

    def _step_batch(self, params, sbatch, cbatch):
        sbatch = jnp.expand_dims(sbatch, 2)
        cbatch = jnp.expand_dims(cbatch, 2)
        b1 = jnp.expand_dims(params['b1'], 1)
        bout = jnp.expand_dims(params['bout'], 1)
        # print('_step_batch', params['winp1'].shape, cbatch.shape, sbatch.shape, b1.shape)
        sbatch = jnp.matmul(params['winp1'], cbatch) + jnp.matmul(params['wstate1'], sbatch) + b1
        sbatch = jnp.tanh(sbatch)
        out = jnp.matmul(params['wout'], sbatch) + bout
        # print('_step_batch2', sbatch.shape, out.shape)
        sbatch = sbatch.squeeze(axis=2)
        out = out.squeeze(axis=2)
        return sbatch, out


    def _loss_batch(self, params, xbatch):
        sbatch = jnp.zeros((xbatch.shape[0], 256))
        xbatch = jnp.swapaxes(xbatch, 0, 1)  # (element, batch, one-hot)
        _, ybatch = jax.lax.scan(lambda s, c: self._step_batch(params, s, c), sbatch, xbatch)
        entropy = optax.softmax_cross_entropy(ybatch[:,:-1,:], xbatch[:,1:,:])
        return jnp.average(entropy)


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
        hidden = self._activation(hidden)

        state = jnp.dot(params['wstate2'], hidden) + params['b2']
        state = self._activation(state)

        out = jnp.dot(params['wout'], state) + params['bout']
        return state, out


# class RecurrentL2(Model):
#     def __init__(self, hidden1, hidden2, activation):
#         self._hidden1 = hidden1
#         self._hidden2 = hidden2
#         self._activation = activation

#     def init_params(self, key):
#         keys = jax.random.split(key, 8)
#         return {
#             'winp1': jax.random.normal(keys[0], shape=(self._hidden1, NCHAR)),
#             'wstate1': jax.random.normal(keys[1], shape=(self._hidden1, self._hidden1)),
#             'b1': jax.random.normal(keys[2], shape=(self._hidden1,)),

#             'w2': jax.random.normal(keys[3], shape=(self._hidden2, self._hidden1)),
#             'wstate2': jax.random.normal(keys[4], shape=(self._hidden2, self._hidden2)),
#             'b2': jax.random.normal(keys[5], shape=(self._hidden2,)),

#             'wout': jax.random.normal(keys[6], shape=(NCHAR, self._hidden2)),
#             'bout': jax.random.normal(keys[7], shape=(NCHAR,))
#         }

#     def init_state(self):
#         return {
#             'state1': jnp.zeros((self._hidden1,)),
#             'state2': jnp.zeros((self._hidden2,)),
#         }

#     def step(self, params, state, c):
#         """One forward step in the recurrent network.

#         Args:
#             params: dictionary with model parameters
#             state: array with the internal state, initially zero. shape = (256,)
#             c: a single character represented as a one-of. shape = (256,)
#         """
#         state1 = state['state1']
#         state1 = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state1) + params['b1']
#         state1 = self._activation(state1)

#         state2 = state['state2']
#         state2 = jnp.dot(params['w2'], state1) + jnp.dot(params['wstate2'], state2) + params['b2']
#         state2 = self._activation(state2)

#         out = jnp.dot(params['wout'], state2) + params['bout']
#         return {'state1': state1, 'state2': state2}, out


class RecurrentGRU(Model):
    def __init__(self, hidden):
        self._hidden = hidden

    def init_params(self, key):
        keys = jax.random.split(key, 16)
        return {
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
        t = jnp.dot(params['wout'], hn) + params['bout']
        t = jax.nn.sigmoid(t)

        out = jnp.dot(params['w2'], t) + params['b2']

        return {'h': hn, 'prev': c}, out


class Recurrent2(Model):
    def __init__(self, hidden, activation):
        self._hidden = hidden
        self._activation = activation

    def init_params(self, key):
        key0, key1, key2, key3, key4, key5 = jax.random.split(key, 6)
        return {
            'winp1': jax.random.normal(key0, shape=(self._hidden, NCHAR)),
            'b1': jax.random.normal(key1, shape=(self._hidden,)),
            'wstate1': jax.random.normal(key2, shape=(self._hidden, self._hidden)),
            'wstateout': jax.random.normal(key3, shape=(NCHAR, self._hidden)),
            'winout': jax.random.normal(key4, shape=(NCHAR, NCHAR)),
            'bout': jax.random.normal(key5, shape=(NCHAR,))
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
        new_state = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state) + params['b1']
        new_state = self._activation(state)
        # state = jax.nn.normalize(state)
        out = jnp.dot(params['wstateout'], state) + jnp.dot(params['winout'], c) + params['bout']
        return new_state, out
