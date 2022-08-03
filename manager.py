import jax
import jax.numpy as jnp
import json
import math
import optax
import os

import models
from model import NCHAR


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
        if step <= initial_steps:
            return initial_lr

        if step >= total_steps:
            return final_lr

        t = (step - initial_steps) / (total_steps - initial_steps)
        return initial_lr * math.exp(t * log_scale)

    return sch


def deserialize_params(spec):
    params = {}
    for key, value in spec.items():
        if type(value) is dict:
            params[key] = deserialize_params(value)
        else:
            params[key] = jnp.array(value)
    return params


class Manager(object):
    @staticmethod
    def from_spec(spec):
        model = models.build(spec['model'])
        params = deserialize_params(spec['params'])
        return Manager(model, spec['learning_rate'], spec['regularization'], spec.get('total_steps', 0), spec['step'], params)

    def __init__(self, model, learning_rate, regularization, total_steps, step=0, params=None):
        print('Creating Model')
        self._key = jax.random.PRNGKey(42)
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.total_steps = total_steps
        self.step = step
        if params is not None:
            self.params = params
        else:
            self.params = self.model.init_params(self.key())
        self.loss = None

    def init(self):
        print('Total parameters:', self.model.total_params(self.params))
        print('Creating batch loss')
        self._loss_batch = jax.vmap(self.model.loss, in_axes=(None, 0), out_axes=0)
        print('Creating loss_avg')
        self._loss_avg = lambda params, xs: jnp.average(self._loss_batch(params, xs)) * LOG2
        print('Creating loss_grad')
        self._loss_grad = jax.jit(jax.value_and_grad(self._loss_avg))
        print('Creating optimizer')
        self.optimizer = optax.chain(
            clip_by_global_norm(1),
            optax.scale_by_adam(),
            additive_weight_decay(self.regularization),
            optax.scale(-self.learning_rate),
            optax.scale_by_schedule(
                exp_schedule(
                    self.total_steps//10, self.total_steps,
                    self.learning_rate, self.learning_rate/10))
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

        print('Sample loss:', jnp.average(loss))

    def batch_loss(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg(self.params, xs)

    def evaluate(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        self.loss = self._loss_avg(self.params, xs).item()
        return self.loss

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

        self.step += 1

        return loss

    def serialize_params(self, params):
        serialized = {}
        for key, value in params.items():
            if type(value) is dict:
                serialized[key] = self.serialize_params(value)
            else:
                serialized[key] = value.tolist()
        return serialized

    def save(self, dir):
        model = self.model.serialize()
        model_name = self.name()

        path = os.path.join(dir, f'{model_name}.json')

        params = self.serialize_params(self.params)

        data = {
            'model': model,
            'total_steps': self.total_steps,
            'step': self.step,
            'params': params,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'loss': self.loss
        }

        print(f'Saving model to {path}')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def name(self):
        model_name = self.model.name
        return f'{model_name}-{self.step}'
