import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import random
import time


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


def _step(params, state, c):
    """One forward step in the recurrent network.

    Args:
        params: dictionary with model parameters
        state: array with the internal state, initially zero. shape = (256,)
        x: a single character represented as a one-of. shape = (256,)
    """
    # state1 = state[256:]
    # state2 = state[:256]
    state = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state) + params['b1']
    state = jnp.tanh(state)
    # state2 = jnp.dot(params['winp2'], state1) + jnp.dot(params['wstate2'], state2) + params['b2']
    # state2 = jnp.tanh(state2)
    # state = jnp.concatenate((state1, state2))
    out = jnp.dot(params['wout'], state) + params['bout']
    return state, out


def _loss(params, x):
    """Calculate loss for a single sample string.

    Args:
        params: dictionary with model parameters
        x: a string represented as one-of. shape = (L, 256)
    """
    _, y = jax.lax.scan(lambda s, c: _step(params, s, c), jnp.zeros((256,)), x)
    return optax.softmax_cross_entropy(y[:-1,:], x[1:,:]) / (x.shape[0] - 1)


def _step_batch(params, sbatch, cbatch):
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


def _loss_batch(params, xbatch):
    sbatch = jnp.zeros((xbatch.shape[0], 256))
    xbatch = jnp.swapaxes(xbatch, 0, 1)  # (element, batch, one-hot)
    _, ybatch = jax.lax.scan(lambda s, c: _step_batch(params, s, c), sbatch, xbatch)
    entropy = optax.softmax_cross_entropy(ybatch[:,:-1,:], xbatch[:,1:,:])
    return jnp.average(entropy)


class Model(object):
    def __init__(self):
        print('Creating Model')
        self.key = jax.random.PRNGKey(42)
        self.params = {
            'wstate1': jax.random.normal(self.key, shape=(256, 256)),
            'winp1': jax.random.normal(self.key, shape=(256, 256)),
            # 'wstate2': jax.random.normal(self.key, shape=(256, 256)),
            # 'winp2': jax.random.normal(self.key, shape=(256, 256)),
            'wout': jax.random.normal(self.key, shape=(256, 256)),
            'b1': jax.random.normal(self.key, shape=(256,)),
            # 'b2': jax.random.normal(self.key, shape=(256,)),
            'bout': jax.random.normal(self.key, shape=(256,))
        }
        print('Creating batch loss')
        loss_batch = jax.vmap(_loss, in_axes=(None, 1), out_axes=0)
        print('Creating loss_avg')
        loss_avg = lambda params, xs: jnp.average(loss_batch(params, xs))
        print('loss_avg')
        # self._loss_grad = jax.jit(jax.value_and_grad(_loss_batch))
        self._loss_grad = jax.jit(jax.value_and_grad(loss_avg))
        print('_loss_grad')
        self.optimizer = optax.adam(0.1)
        self.opt_state = self.optimizer.init(self.params)

    def sample_batch(self, prefix, l, temperature=0.05):
        print(prefix)
        prefix = jnp.array(list(prefix))
        c_selected = prefix[-1]
        prefix_batch = jnp.expand_dims(jax.nn.one_hot(prefix, 256), 1)
        c = prefix_batch[-1,:,:]

        state = jnp.zeros((1, 512))
        print('sample', state.shape, prefix_batch[:-1,:,:].shape)
        state, _ = jax.lax.scan(lambda s, c: _step_batch(self.params, s, c), state, prefix_batch[:-1,:,:])

        out = []
        while len(out) < l and c_selected != 0:
            state, c = _step_batch(self.params, state, c)
            c_softmax = jax.nn.softmax(c.squeeze() / temperature)
            c_selected = jax.random.choice(self.key, 256, p=c_softmax)
            out.append(c_selected)
        return bytes(out)

    def sample(self, prefix, l, temperature=0.05):
        print(prefix)
        prefix = jnp.array(list(prefix))
        c_selected = prefix[-1]
        prefix = jax.nn.one_hot(prefix, 256)
        c = prefix[-1,:]

        state = jnp.zeros((256,))
        print('sample', state.shape, prefix[:-1,:].shape)
        state, _ = jax.lax.scan(lambda s, c: _step(self.params, s, c), state, prefix[:-1,:])

        out = []
        while len(out) < l and c_selected != 0:
            state, c = _step(self.params, state, c)
            c_softmax = jax.nn.softmax(c / temperature)
            c_selected = jax.random.choice(self.key, 256, p=c_softmax)
            c = jax.nn.one_hot(c_selected, 256)
            out.append(c_selected)
        return bytes(out)

    def train(self, xs):
        xs = jax.nn.one_hot(xs, 256)
        loss, grads = self._loss_grad(self.params, xs)
        print(loss)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)


def main(dir, steps):
    print(f'Training data: {dir}')
    train_set = TrainSet(dir)

    model = Model()

    start = time.time()

    for i in range(steps):
        batch = train_set.sample(16, 64)
        # print(batch)
        model.train(batch)

    print(f'Training time: {time.time() - start}')

    out = model.sample(b'Roses are red\nViolets are blue,\n', 256)
    print(out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir, args.steps)
