import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import random
import time

from dataset import DataSet

NCHARS = 256


def softmax_cross_entropy(logits, labels_oh):
    print(logits)
    print(labels_oh)
    return -NCHARS * jnp.mean(jax.nn.log_softmax(logits) * labels_oh)


def equal_prob_step(_params, state, c):
    return state, jnp.ones((NCHARS,))

def train_noop(params, opt_state, xs):
    return None, None

EQUAL_PROB = {
    'step_func': equal_prob_step,
    'train_func': train_noop,
    'param_shapes': {},
    'init_state': None,
    'init_opt_state': None
}


def two_layer_step(params, state, c):
    """One forward step in the recurrent network.

    Args:
        params: dictionary with model parameters
        state: array with the internal state, initially zero. shape = (256,)
        x: a single character represented as a one-of. shape = (256,)
    """
    state1 = state[256:]
    state2 = state[:256]
    state1 = jnp.dot(params['winp1'], c) + jnp.dot(params['wstate1'], state1) + params['b1']
    state = jax.nn.leaky_relu(state)
    # state1 = jnp.tanh(state1)
    state2 = jnp.dot(params['winp2'], state1) + jnp.dot(params['wstate2'], state2) + params['b2']
    state2 = jnp.tanh(state2)
    state = jnp.concatenate((state1, state2))
    out = jnp.dot(params['wout'], state2) + params['bout']
    # out = jnp.tanh(out)
    return state, out

def two_layer_loss(params, x):
    """Calculate loss for a single sample string.

    Args:
        params: dictionary with model parameters
        x: a string represented as one-of. shape = (L, 256)
    """
    _, y = jax.lax.scan(lambda s, c: two_layer_step(params, s, c), jnp.zeros((512,)), x)
    return softmax_cross_entropy(y[:-1,:], x[1:,:])


TWO_LAYERS = {
    'step_func': two_layer_step,
    'param_shapes': {
        'winp1': (256, NCHARS),
        'wstate1': (256, 256),
        'b1': (256,),

        'winp2': (256, 256),
        'wstate2': (256, 256),
        'b2': (256,),

        'wout': (NCHARS, 256),
        'bout': (NCHARS,)
    },
    'init_state': np.zeros(512,)
}


# def _step_batch(params, sbatch, cbatch):
#     sbatch = jnp.expand_dims(sbatch, 2)
#     cbatch = jnp.expand_dims(cbatch, 2)
#     b1 = jnp.expand_dims(params['b1'], 1)
#     bout = jnp.expand_dims(params['bout'], 1)
#     # print('_step_batch', params['winp1'].shape, cbatch.shape, sbatch.shape, b1.shape)
#     sbatch = jnp.matmul(params['winp1'], cbatch) + jnp.matmul(params['wstate1'], sbatch) + b1
#     sbatch = jnp.tanh(sbatch)
#     out = jnp.matmul(params['wout'], sbatch) + bout
#     # print('_step_batch2', sbatch.shape, out.shape)
#     sbatch = sbatch.squeeze(axis=2)
#     out = out.squeeze(axis=2)
#     return sbatch, out


# def _loss_batch(params, xbatch):
#     sbatch = jnp.zeros((xbatch.shape[0], 256))
#     xbatch = jnp.swapaxes(xbatch, 0, 1)  # (element, batch, one-hot)
#     _, ybatch = jax.lax.scan(lambda s, c: _step_batch(params, s, c), sbatch, xbatch)
#     entropy = optax.softmax_cross_entropy(ybatch[:,:-1,:], xbatch[:,1:,:])
#     return jnp.average(entropy)


class Model(object):
    def __init__(self, model_def=None):
        print('Creating model')
        self.key = jax.random.PRNGKey(random.randrange(2**31))
        self.params = {}
        self.step = model_def['step_func']

        print('Initializing parameters')
        for param, shape in model_def['param_shapes'].items():
            self.params[param] = jax.random.normal(self.key, shape=shape) * 0.1

        print('Creating loss')
        def loss(params, x):
            init_state = model_def['init_state']
            _, y = jax.lax.scan(lambda s, c: self.step(params, s, c), init_state, x)
            print('x:\n', x)
            print('y:\n', y)
            return softmax_cross_entropy(y[:-1,:], x[1:,:])

        print('Creating batch loss')
        loss_batch = jax.vmap(loss, in_axes=(None, 1), out_axes=0)

        print('Creating loss_avg')
        self.loss_avg = lambda params, xs: jnp.mean(loss_batch(params, xs))

        self.train_func = model_def.get('train_func')
        self.opt_state = model_def.get('init_opt_state')
        if self.train_func is None:
            print('Creating loss_grad')
            # self._loss_grad = jax.value_and_grad(loss_avg)
            self.loss_grad = jax.jit(jax.value_and_grad(self.loss_avg), backend='cpu')

            print('Initializing optimizer')
            self.optimizer = optax.adam(0.01)
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

    def sample(self, prefix, l, temperature=0.01):
        print(prefix)
        prefix = jnp.array(list(prefix))
        c_selected = prefix[-1]
        prefix = jax.nn.one_hot(prefix, NCHARS)
        c = prefix[-1,:]

        state = jnp.zeros((512,))
        print('sample', state.shape, prefix[:-1,:].shape)
        state, _ = jax.lax.scan(lambda s, c: self.step(self.params, s, c), state, prefix[:-1,:])

        out = []
        while len(out) < l and c_selected != 0:
            state, c = self.step(self.params, state, c)
            c_softmax = jax.nn.softmax(c / temperature)
            key = jax.random.PRNGKey(random.randrange(2**31))
            c_selected = jax.random.choice(key, NCHARS, p=c_softmax)
            c = jax.nn.one_hot(c_selected, NCHARS)
            out.append(c_selected)
        return bytes(out)

    def train(self, xs):
        xs = jax.nn.one_hot(xs, NCHARS)

        if self.train_func is not None:
            self.params, self.opt_state = self.train_func(self.params, self.opt_state, xs)
        else:
            loss, grads = self.loss_grad(self.params, xs)
            print(loss)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

    def evaluate(self, xs):
        xs = jax.nn.one_hot(xs, NCHARS)
        print(xs)
        return self.loss_avg(self.params, xs)


def train(dir, steps):
    jax.config.update("jax_debug_nans", True)

    print(f'Training data: {dir}')
    dataset = DataSet(dir)

    # model = Model(TWO_LAYERS)
    model = Model(EQUAL_PROB)

    start = time.time()

    for i in range(steps):
        batch = dataset.sample(32, 128)
        # print(batch)
        model.train(batch)

    print(f'Training time: {time.time() - start}')

    out = model.sample(b'Roses are red\nViolets are blue,\n', 256, temperature=0.2)
    # out = model.sample_batch(b'Roses are red\nViolets are blue,\n', 256)
    print(out)

    # eval_batch = dataset.sample(length=1024, size=32)
    eval_batch = np.array([list(b'abc')], dtype=np.ubyte)
    print(eval_batch)
    print(model.evaluate(eval_batch))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.dir, args.steps)
