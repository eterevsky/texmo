import argparse
import jax
import jax.numpy as jnp
import math
import numpy as np
import optax
import os
import random
import time


NCHAR = 256  # Considering characters with codes 0..NCHAR


class TrainSet(object):
    def __init__(self, dir):
        self.texts = []
        self.cum_weights = []
        total = 0
        for dir, _, files in os.walk(dir):
            for filename in files:
                path = os.path.join(dir, filename)
                print(f'Loading {path}')
                with open(path, 'rb') as f:
                    text = f.read()
                self.texts.append(text)
                total += len(text)
                self.cum_weights.append(total)

    def sample(self, length, batch_size):
        texts_batch = random.choices(self.texts, cum_weights=self.cum_weights, k=batch_size)
        batch = []
        for text in texts_batch:
            start = random.randrange(len(text))
            sample = text[start:start + length]
            sample += b'\0' * (length - len(sample))
            batch.append(list(sample))
        batch = np.array(batch, dtype=np.ubyte)
        return batch

    def all(self):
        return b'\n'.join(self.texts)


class Model(object):
    def init_params(self, key):
        """Initializes the params object.

        Args:
            key: JAX pseudorandom key, PRNGKey
        """
        return {}

    def init_state(self):
        """Initialize the state object."""
        return None

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with the model parameters
            state: array with the internal state (initialized by init_state or returned from the
                previous step)
            x: a single character represented as a one-of. shape = (256,)

        Returns:
            (new state, an output 1D vector of size NCHARS that after softmax will produce character
            probabilities.)
        """
        pass

    def step_prob(self, params, state, c, temperature=1.0):
        """Runs one step of the model and calculate the final probabilities.

        Arguments are the same as in step().

        Returns:
            (new state, a 1D vector of size NCHAR with probabilities of characters)
        """
        state, out = self.step(params, state, c)
        out_softmax = jax.nn.softmax(out / temperature)
        return state, out_softmax

    def loss(self, params, x):
        _, y = jax.lax.scan(lambda s, c: self.step(params, s, c), self.init_state(), x)
        # print('loss', x.shape, y.shape)
        return optax.softmax_cross_entropy(y[:-1,:], x[1:,:])

    def params_loss(self, params):
        if not params: return 0
        s = 0
        for v in params.values():
            s += jnp.average(v * v)

        return s


class Equal(Model):
    def init_params(self, key):
        return {}

    def init_state(self):
        return jnp.zeros((self._hidden,))

    def step(self, params, state, c):
        return state, jnp.ones_like(c)


class Freq(Model):
    def init_params(self, key):
        return {
            'b': jax.random.normal(key, shape=(NCHAR,)),
        }

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


class RealMarkov(Model):
    def init_params(self, key):
        return {
            'w': jnp.zeros((NCHAR, NCHAR)),
        }

    def step(self, params, state, c):
        out = jnp.matmul(params['w'], c)
        return state, out

    def loss(self, params, x):
        _, y = jax.lax.scan(lambda s, c: self.step(params, s, c), self.init_state(), x)
        # return optax.softmax_cross_entropy(y[:-1,:], x[1:,:])
        logits = y[:-1,:]
        labels = x[1:,:]
        return -jnp.sum(labels * jnp.log(logits), axis=1)

    def train(self, text):
        w = np.ones((NCHAR, NCHAR))
        for a, b in zip(text[:-1], text[1:]):
            w[b][a] += 1
        print(w[97:102,97:102].astype(np.int32))
        w /= jnp.sum(w, axis=0)  #.reshape((NCHAR, 1))
        return {'w': w}


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
    def __init__(self, hidden1, hidden2, activation):
        self._hidden1 = hidden1
        self._hidden2 = hidden2
        self._activation = activation

    def init_params(self, key):
        keys = jax.random.split(key, 8)
        return {
            'winp1': jax.random.normal(keys[0], shape=(self._hidden1, NCHAR)),
            'wstate1': jax.random.normal(keys[1], shape=(self._hidden1, self._hidden1)),
            'b1': jax.random.normal(keys[2], shape=(self._hidden1,)),

            'w2': jax.random.normal(keys[3], shape=(self._hidden2, self._hidden1)),
            'wstate2': jax.random.normal(keys[4], shape=(self._hidden2, self._hidden2)),
            'b2': jax.random.normal(keys[5], shape=(self._hidden2,)),

            'wout': jax.random.normal(keys[6], shape=(NCHAR, self._hidden2)),
            'bout': jax.random.normal(keys[7], shape=(NCHAR,))
        }

    def init_state(self):
        return {
            'state1': jnp.zeros((self._hidden1,)),
            'state2': jnp.zeros((self._hidden2,)),
        }

    def step(self, params, state, c):
        """One forward step in the recurrent network.

        Args:
            params: dictionary with model parameters
            state: array with the internal state, initially zero. shape = (256,)
            c: a single character represented as a one-of. shape = (256,)
        """
        state1 = state['state1']
        state1 = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state1) + params['b1']
        state1 = self._activation(state1)

        state2 = state['state2']
        state2 = jnp.dot(params['w2'], state1) + jnp.dot(params['wstate2'], state2) + params['b2']
        state2 = self._activation(state2)

        out = jnp.dot(params['wout'], state2) + params['bout']
        return {'state1': state1, 'state2': state2}, out


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


class Manager(object):
    def __init__(self, model, learning_rate):
        print('Creating Model')
        self._key = jax.random.PRNGKey(42)
        self.model = model
        self.params = self.model.init_params(self.key())

        print('Creating batch loss')
        self._loss_batch = jax.vmap(model.loss, in_axes=(None, 0), out_axes=0)
        print('Creating loss_avg')
        self._loss_avg = lambda params, xs: jnp.average(self._loss_batch(params, xs))
        self._loss_avg = jax.jit(self._loss_avg)
        self._total_loss = lambda params, xs: jnp.average(self._loss_batch(params, xs)) + 0.1 * model.params_loss(params)
        # self._loss_grad = jax.jit(jax.value_and_grad(_loss_batch))
        self._loss_grad = jax.jit(jax.value_and_grad(self._total_loss))
        self.optimizer = optax.adamw(learning_rate)
        # self.optimizer = optax.sgd(0.001)
        self.opt_state = self.optimizer.init(self.params)

    def key(self):
        self._key, key = jax.random.split(self._key)
        return key

    # def sample_batch(self, prefix, l, temperature=0.05):
    #     print(prefix)
    #     prefix = jnp.array(list(prefix))
    #     c_selected = prefix[-1]
    #     prefix_batch = jnp.expand_dims(jax.nn.one_hot(prefix, 256), 1)
    #     c = prefix_batch[-1,:,:]

    #     state = jnp.zeros((1, 512))
    #     print('sample', state.shape, prefix_batch[:-1,:,:].shape)
    #     state, _ = jax.lax.scan(
    #         lambda s, c: _step_batch(self.params, s, c), state, prefix_batch[:-1,:,:])

    #     out = []
    #     while len(out) < l and c_selected != 0:
    #         state, c = _step_batch(self.params, state, c)
    #         c_softmax = jax.nn.softmax(c.squeeze() / temperature)
    #         c_selected = jax.random.choice(self.key, 256, p=c_softmax)
    #         out.append(c_selected)
    #     return bytes(out)

    def sample_loss(self, s):
        """Loss of the model for a given string."""

        sarray = jnp.array(list(s))
        soh = jax.nn.one_hot(sarray, NCHAR)

        state = self.model.init_state()
        _, r = self.model.step(self.params, state, soh[0])
        print('prediction: ', r)

        loss = self.model.loss(self.params, soh) / math.log(2)
        print(chr(s[0]))
        for c, p in zip(s[1:], loss):
            if c == ord('\n'): c = ord('\\')
            print(chr(c), p)

        print('Sample loss:', jnp.average(loss))

    def batch_loss(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg(self.params, xs) / math.log(2)

    def sample(self, prefix, l, temperature=0.05):
        prefix = jnp.array(list(prefix))
        c_selected = prefix[-1]
        prefix = jax.nn.one_hot(prefix, 256)
        c = prefix[-1,:]

        state = self.model.init_state()
        state, _ = jax.lax.scan(lambda s, c: self.model.step(self.params, s, c), state, prefix)

        out = []
        while len(out) < l and c_selected != 0:
            state, c = self.model.step_prob(self.params, state, c)
            c_selected = jax.random.choice(self.key(), NCHAR, p=c)
            c = jax.nn.one_hot(c_selected, NCHAR)
            out.append(c_selected)
        return bytes(out)

    def train(self, xs, calc_real_loss=False):
        xs = jax.nn.one_hot(xs, NCHAR)
        loss, grads = self._loss_grad(self.params, xs)
        if calc_real_loss:
            real_loss = self._loss_avg(self.params, xs)
        else:
            real_loss = None
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

        return loss, real_loss


def main(dir, steps, learning_rate):
    print(f'Training data: {dir}')
    train_set = TrainSet(dir)

    # model = Equal()
    # model = Freq()
    model = Markov2()
    # model = RecurrentL1(hidden=128, activation=jax.nn.hard_sigmoid)
    # model = Recurrent2(hidden=128, activation=jax.nn.hard_sigmoid)
    # model = RealMarkov()
    # model = RecurrentL2(hidden1=128, hidden2=128, activation=jax.nn.relu6)

    manager = Manager(model, learning_rate)

    start = time.time()

    for i in range(steps):
        batch = train_set.sample(64, 64)
        # print(batch.shape)
        loss, real_loss = manager.train(batch, i % 10 == 0)
        if real_loss is not None:
            print(i, loss, real_loss)

    print(f'Training time: {time.time() - start}')

    manager.sample_loss(b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(64, 32)
    print('Batch loss:', manager.batch_loss(batch))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)
    print(prefix + out)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate', default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir, args.steps, args.learning_rate)
