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


class Model(object):
    def init_params(self, key):
        """Initializes the params object.

        Args:
            key: JAX pseudorandom key, PRNGKey
        """
        return None

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
        """
        pass

    def loss(self, params, x):
        _, y = jax.lax.scan(lambda s, c: self.step(params, s, c), self.init_state(), x)
        print(y[:-1,:].shape)
        print(x[1:,:].shape)
        return optax.softmax_cross_entropy(y[:-1,:], x[1:,:])


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
    def __init__(self, model):
        print('Creating Model')
        self._key = jax.random.PRNGKey(42)
        self.model = model
        self.params = self.model.init_params(self.key())

        print('Creating batch loss')
        self._loss_batch = jax.vmap(model.loss, in_axes=(None, 1), out_axes=0)
        print('Creating loss_avg')
        self._loss_avg = lambda params, xs: jnp.average(self._loss_batch(params, xs))
        # self._loss_grad = jax.jit(jax.value_and_grad(_loss_batch))
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss_avg))
        self.optimizer = optax.adam(0.001)
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
            state, c = self.model.step(self.params, state, c)
            c_softmax = jax.nn.softmax(c / temperature)
            c_selected = jax.random.choice(self.key(), 256, p=c_softmax)
            c = jax.nn.one_hot(c_selected, NCHAR)
            out.append(c_selected)
        return bytes(out)

    def train(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        # print('shape:', xs.shape)
        loss, grads = self._loss_grad(self.params, xs)
        # print('params:')
        # print(self.params)
        # print('loss:')
        # print(loss)
        # print('grads:')
        # print(grads)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

        return loss / math.log(2)


def main(dir, steps):
    print(f'Training data: {dir}')
    train_set = TrainSet(dir)

    # model = Equal()
    # model = Freq()
    # model = Markov()
    # model = RecurrentL1(hidden=128, activation=jax.nn.hard_sigmoid)
    model = Recurrent2(hidden=128, activation=jax.nn.hard_sigmoid)

    manager = Manager(model)

    start = time.time()

    for i in range(steps):
        batch = train_set.sample(64, 128)
        loss = manager.train(batch)
        print(loss)

    print(f'Training time: {time.time() - start}')

    manager.sample_loss(b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(32, 128)
    print('Batch loss:', manager.batch_loss(batch))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)
    print(prefix + out)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir, args.steps)
