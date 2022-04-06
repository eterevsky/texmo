import argparse
import jax
import jax.numpy as jnp
import math
import numpy as np
import optax
import time

from dataset import DataSet
from model import Model, NCHAR
import models


LOG2 = 1 / math.log(2)


def additive_weight_decay(weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Add a delta emulating L2 regularization to all parameters except biases."""

    def init_fn(_):
        return optax.AdditiveWeightDecayState()

    def update_fn(updates, state, params):
        updates = jax.tree_multimap(lambda g, p: g + weight_decay * p * (len(g.shape) > 1), updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class Manager(object):
    def __init__(self, model, learning_rate, regularization):
        print('Creating Model')
        self._key = jax.random.PRNGKey(42)
        self.model = model
        self.params = self.model.init_params(self.key())
        self._regularization = regularization

        print('Creating batch loss')
        self._loss_batch = jax.vmap(model.loss, in_axes=(None, 0), out_axes=0)
        print('Creating loss_avg')
        self._loss_avg = lambda params, xs: jnp.average(self._loss_batch(params, xs)) * LOG2
        self._loss_avg = jax.jit(self._loss_avg)
        self._total_loss = lambda params, xs: jnp.average(self._loss_batch(params, xs)) * LOG2
        self._loss_grad = jax.jit(jax.value_and_grad(self._total_loss))
        self.optimizer = optax.chain(
            optax.scale_by_param_block_rms(),
            optax.scale_by_adam(),
            additive_weight_decay(regularization),
            optax.scale(-learning_rate),
        )

        # self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    def key(self):
        self._key, key = jax.random.split(self._key)
        return key

    def sample_loss(self, s):
        """Loss of the model for a given string."""

        sarray = jnp.array(list(s))
        soh = jax.nn.one_hot(sarray, NCHAR)

        state = self.model.init_state()
        _, r = self.model.step(self.params, state, soh[0])

        loss = self.model.loss(self.params, soh)
        print(chr(s[0]))
        for c, p in zip(s[1:], loss):
            if c == ord('\n'): c = ord('\\')
            print(chr(c), p)

        print('Sample loss:', jnp.average(loss))

    def batch_loss(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg(self.params, xs)

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


def main(dir, steps, learning_rate, regularization):
    print(f'Training data: {dir}')
    train_set = DataSet(dir)

    # model = models.Equal()
    # model = models.Freq()
    # model = models.Markov2()
    model = models.MarkovFlex()
    # model = RecurrentL1(hidden=128, activation=jax.nn.hard_sigmoid)
    # model = Recurrent2(hidden=128, activation=jax.nn.hard_sigmoid)
    # model = RecurrentL2(hidden1=256, hidden2=256, activation=jax.nn.sigmoid)
    # model = RecurrentGRU(256)

    manager = Manager(model, learning_rate, regularization)

    start = time.time()

    for i in range(steps):
        batch = train_set.sample(32, 32)
        # print(batch.shape)
        loss, real_loss = manager.train(batch, i % 10 == 0)
        if real_loss is not None:
            print(i, loss, real_loss)

    print(f'Training time: {time.time() - start}')

    manager.sample_loss(b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(128, 128)
    print('Batch loss:', manager.batch_loss(batch))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)
    print(prefix + out)
    s = (prefix + out).decode('utf-8')
    print()
    print(s)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('-r', '--regularization', type=float, help='L2 regularization coefficient',
                        default=0.001)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir, args.steps, args.learning_rate, args.regularization)
