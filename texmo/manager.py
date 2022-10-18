import jax
import jax.numpy as jnp
import json
import math
import optax
import os
import random

from .layered import LayeredModel2
from .model import NCHAR


LOG2 = 1 / math.log(2)


def global_norm(updates):
    pre_sqrt = sum(
        [jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(updates)]
    )
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

    def update_fn(updates, state, weights=None):
        del weights
        g_norm = global_norm(updates)
        trigger = g_norm < max_norm
        updates = jax.tree_util.tree_map(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_norm), updates
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def additive_weight_decay(
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Add a delta emulating L2 regularization to all parameters except biases."""

    def init_fn(_):
        return optax.AdditiveWeightDecayState()

    def update_fn(updates, state, weights):
        updates = jax.tree_util.tree_map(
            lambda g, p: g + weight_decay * p * (len(g.shape) > 1),
            updates,
            weights,
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def gpt3_schedule(warmup_steps, anneal_steps, peak_lr, end_lr):
    def sch(step):
        warmup_pct = jnp.clip(step, 0, warmup_steps) / warmup_steps
        anneal_pct = (
            jnp.clip(step - warmup_steps, 0, anneal_steps) / anneal_steps
        )

        return (
            warmup_pct * peak_lr
            - (peak_lr - end_lr) * (1 - jnp.cos(jnp.pi * anneal_pct)) / 2
        )

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


def deserialize_weights(spec):
    weights = {}
    for key, value in spec.items():
        if type(value) is dict:
            weights[key] = deserialize_weights(value)
        else:
            weights[key] = jnp.array(value)
    return weights


class Manager(object):
    @staticmethod
    def from_spec(spec):
        model_spec = spec["model"]
        assert model_spec["name"] == "layered"
        model = LayeredModel2.from_spec(model_spec)
        weights = deserialize_weights(spec["weights"])
        return Manager(
            model,
            spec["learning_rate"],
            spec["regularization"],
            spec["step"],
            weights,
            step_loss=spec.get("step_loss", None),
        )

    def __init__(
        self,
        model,
        learning_rate,
        regularization,
        step=0,
        weights=None,
        step_loss=None,
        init_scale=1,
    ):
        self._key = jax.random.PRNGKey(random.randrange(2**32))
        self.model = model
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.total_steps = 100000
        self.step = step
        self.init_scale = init_scale
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.model.init_weights(self.key(), self.init_scale)
        self.loss = None
        if step_loss is None:
            self.step_loss = []
        else:
            self.step_loss = step_loss

    def init(self, quiet=False, training=True):
        if not quiet:
            print("Total parameters:", self.model.total_weights(self.weights))
            print("Creating batch loss")
        self._loss_batch = jax.vmap(
            self.model.loss, in_axes=(None, 0), out_axes=0
        )
        if not quiet:
            print("Creating loss_avg")
        self._loss_avg = (
            lambda weights, xs: jnp.average(self._loss_batch(weights, xs))
            * LOG2
        )
        if training:
            if not quiet:
                print("Creating loss_grad")
            self._loss_grad = jax.jit(jax.value_and_grad(self._loss_avg))
            if not quiet:
                print("Creating optimizer")
            self.optimizer = optax.chain(
                clip_by_global_norm(1),
                optax.scale_by_adam(),
                additive_weight_decay(self.regularization),
                optax.scale(-self.learning_rate),
                optax.scale_by_schedule(
                    exp_schedule(
                        10000,  # initial_steps
                        100000,  # steps to lr/10
                        self.learning_rate,
                        self.learning_rate / 10,
                    )
                ),
            )

            self.opt_state = self.optimizer.init(self.weights)
        else:
            self._loss_grad = None
            self.optimizer = None
            self.opt_state = None

    def key(self):
        self._key, key = jax.random.split(self._key)
        return key

    def sample_loss(self, s):
        """Loss of the model for a given string."""

        sarray = jnp.array(list(s))
        soh = jax.nn.one_hot(sarray, NCHAR)

        state = self.model.init_state(self.weights)
        _, r = self.model.step(self.weights, state, soh[0])

        loss = self.model.loss(self.weights, soh)

        print("Sample loss:", jnp.average(loss))

    def batch_loss(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        return self._loss_avg(self.weights, xs)

    def evaluate(self, xs):
        loss = 0
        for i in range(xs.shape[0] // 256):
            shard = xs[i * 256 : (i + 1) * 256]
            shard = jax.nn.one_hot(shard, NCHAR)
            loss += self._loss_avg(self.weights, shard).item()

        self.loss = loss / (xs.shape[0] // 256)

        return self.loss

    def sample(self, prefix, l, temperature=0.05):
        prefix = jnp.array(list(prefix))
        c_selected = prefix[-1]
        prefix = jax.nn.one_hot(prefix, 256)
        c = prefix[-1, :]

        state = self.model.init_state(self.weights)
        state, _ = jax.lax.scan(
            lambda s, c: self.model.step(self.weights, s, c), state, prefix[:-1]
        )

        out = []
        while len(out) < l and c_selected != 0:
            state, c = self.model.step_prob(self.weights, state, c)
            c_selected = jax.random.choice(self.key(), NCHAR, p=c)
            c = jax.nn.one_hot(c_selected, NCHAR)
            out.append(c_selected)
        return bytes(out)

    def train(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        loss, grads = self._loss_grad(self.weights, xs)

        # if loss > 100:
        #     self.weights = self.prev_weights
        #     print('Revert step')
        # else:
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.weights
        )
        self.prev_weights = self.weights
        self.weights = optax.apply_updates(self.weights, updates)

        self.step += 1
        self.step_loss.append(float(loss))

        return loss

    def serialize_weights(self, weights):
        if weights is None:
            return None
        elif type(weights) is dict:
            serialized = {}
            for key, value in weights.items():
                serialized[key] = self.serialize_weights(value)
            return serialized
        else:
            return weights.tolist()

    def save(self, dir):
        model_spec = self.model.serialize()
        model_name = self.name()

        path = os.path.join(dir, f"{model_name}.json")

        weights = self.serialize_weights(self.weights)

        data = {
            "model": model_spec,
            "total_steps": self.total_steps,
            "step": self.step,
            "weights": weights,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "loss": self.loss,
            "step_loss": self.step_loss,
        }

        print(f"Saving model to {path}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def name(self):
        model_name = self.model.full_name
        return f"{model_name}-{self.step}"
