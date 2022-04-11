import argparse
import re
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import time

from dataset import DataSet
from model import Model, NCHAR
import models


LOG2 = 1 / math.log(2)


def global_norm(updates):
    pre_sqrt = sum([jnp.sum(jnp.square(x)) for x in jax.tree_leaves(updates)])
    return jnp.sqrt(pre_sqrt)


def clip_by_global_norm(max_norm) -> optax.GradientTransformation:
    """Clip updates using their global norm.
    References:
      [Pascanu et al, 2012](https://arxiv.org/abs/1211.5063)
    Args:
      max_norm: the maximum global norm for an update.
    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = global_norm(updates)
        trigger = g_norm < max_norm
        updates = jax.tree_map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def additive_weight_decay(weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Add a delta emulating L2 regularization to all parameters except biases."""

    def init_fn(_):
        return optax.AdditiveWeightDecayState()

    def update_fn(updates, state, params):
        updates = jax.tree_multimap(lambda g, p: g + weight_decay * p * (len(g.shape) > 1),
                                    updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def gpt3_schedule(warmup_steps,
                  anneal_steps,
                  peak_lr,
                  end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = jnp.clip(step - warmup_steps, 0, anneal_steps) / anneal_steps

        return warmup_pct * peak_lr - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2

    return sch


def exp_schedule(initial_steps, total_steps, initial_lr, final_lr):
    log_scale = math.log(final_lr / initial_lr)

    def sch(step):
        if step < initial_steps:
            return initial_lr

        t = (step - initial_steps) / (total_steps - initial_steps)
        return initial_lr * math.exp(t * log_scale)

    return sch


class Manager(object):
    def __init__(self, model, learning_rate, regularization, steps):
        print('Creating Model')
        self._key = jax.random.PRNGKey(42)
        self.model = model
        self.params = self.model.init_params(self.key())
        self._regularization = regularization

        print('Creating batch loss')
        self._loss_batch = jax.vmap(model.loss, in_axes=(None, 0), out_axes=0)
        # self._loss_batch = model.loss_batch
        print('Creating loss_avg')
        self._loss_avg = lambda params, xs: jnp.average(self._loss_batch(params, xs)) * LOG2
        # self._loss_avg = jax.jit(self._loss_avg)
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss_avg))
        self.optimizer = optax.chain(
            clip_by_global_norm(1),
            optax.scale_by_adam(),
            additive_weight_decay(regularization),
            optax.scale(-learning_rate),
            optax.scale_by_schedule(exp_schedule(steps//10, steps, learning_rate, learning_rate/10))
        )

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
        # print(chr(s[0]))
        # for c, p in zip(s[1:], loss):
        #     if c == ord('\n'): c = ord('\\')
        #     print(chr(c), p)

        print('Sample loss:', jnp.average(loss))

    def batch_loss(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg(self.params, xs)

    def batch_loss1(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg1(self.params, xs)

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

    def train(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        loss, grads = self._loss_grad(self.params, xs)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)

        return loss


def main(dir, steps, learning_rate, regularization):
    print(f'Training data: {dir}')
    train_set = DataSet(dir)

    # model = models.Equal()
    # model = models.Freq()
    # model = models.Markov2()
    # model = models.MarkovFlex()
    model = models.RecurrentL1(hidden=128, activation=jax.nn.sigmoid)
    # model = models.RecurrentL2(hidden=256, activation=jax.nn.sigmoid)
    # model = models.RecurrentGRU(256)

    manager = Manager(model, learning_rate, regularization, steps)

    start = time.time()

    losses = []
    recent_losses = []

    for i in range(steps):
        batch = train_set.sample(length=128, batch_size=32)
        loss = manager.train(batch)
        if i % 10 == 0 and i > 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            losses.append(avg_loss)
            recent_losses = []
            print(i, avg_loss)
        else:
            recent_losses.append(loss)

    print(f'Training time: {time.time() - start}')

    manager.sample_loss(b'Roses are red\nViolets are blue,\nSugar is sweet\nAnd so are you.')

    batch = train_set.sample(128, 128)
    print('Batch loss:', manager.batch_loss(batch))

    prefix = b'Roses are red\nViolets are blu'
    out = manager.sample(prefix, 256)
    print(prefix + out)

    try:
        s = (prefix + out).decode('utf-8')
    except UnicodeDecodeError:
        s = '<Invalid UTF-8>'
    print()
    print(s)


    plt.xscale('log')
    plt.yscale('log')
    plt.plot(list(range(10, steps, 10)), losses)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    parser.add_argument('-s', '--steps', type=int, help='number of training steps')
    parser.add_argument('-l', '--learning-rate', type=float, help='learning rate', default=0.1)
    parser.add_argument('-r', '--regularization', type=float, help='L2 regularization coefficient',
                        default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir, args.steps, args.learning_rate, args.regularization)
