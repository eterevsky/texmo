import csv
import json
import logging
import math
import os
import time
from copy import copy
from datetime import datetime
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from jaxlib.xla_extension import XlaRuntimeError

from . import latency
from .common import INF, NCHAR
from .configuration import (Configuration, conf_from_dict, conf_to_dict,
                            conf_to_string)
from .model2 import Model2, Weights
from .prng import Rng
from .record import TrainingRecord
from .steploss import StepLossPredictor

LOG2 = 1 / math.log(2)


def deserialize_weights(saved_weights):
    if isinstance(saved_weights, list):
        weights = []
        for value in saved_weights:
            if isinstance(value, dict):
                value = deserialize_weights(value)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], float):
                    value = jnp.array(value)
                else:
                    value = deserialize_weights(value)
            elif value is None:
                pass
            else:
                print(repr(value))
                assert False
            weights.append(value)
    else:
        weights = {}
        for key, value in saved_weights.items():
            if type(value) is dict:
                weights[key] = deserialize_weights(value)
            else:
                weights[key] = jnp.array(value)
    return weights


def release_device_buffers():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


class TrainingDiverged(Exception):
    """Thrown if the loss turs inf or NaN."""

    pass


class Manager(object):
    def __init__(
        self,
        conf: Configuration,
        step: int = 0,
        weights: Optional[Weights] = None,
        step_loss: Optional[list[float]] = None,
        test_sample_len: int = 1024,
        test_batch: int = 1024,
    ):
        self._rng: Rng = Rng()
        self.conf: Configuration = conf
        self.model: Model2 = self.conf.model
        self.step: int = step
        if weights is None:
            weights = self.model.init_weights(self._rng, 1.0)
        self.weights: Weights = weights
        # Only layers starting from this one will be trained
        self.train_from: int = 0

        # Record of the latest and the past losses of the model.
        self.loss: Optional[float] = None
        self.step_loss: list[float] = [] if step_loss is None else step_loss

        self.test_sample_len: int = test_sample_len
        self.test_batch: int = test_batch

    def save(self, dir):
        model_name = self.name()
        path = os.path.join(dir, f"{model_name}.json")

        weights = self.serialize_weights(self.weights)

        data = {
            "conf": conf_to_dict(self.conf),
            "training": {
                "step": self.step,
                "step_loss": self.step_loss,
            },
            "weights": weights,
        }

        print(f"Saving model to {path}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path, training=True):
        with open(path) as f:
            spec = json.load(f)

        conf = conf_from_dict(spec["conf"])
        training_history = spec["training"]
        weights = deserialize_weights(spec["weights"])

        manager = Manager(conf, weights=weights)

        return manager

    def update_conf(self, lr, sample_len, batch, t):
        """Updates the configuration with new parameters.

        Useful after loading a model.
        """
        if lr is not None:
            self.conf = self.conf._replace(lr=lr)
        if sample_len is not None:
            self.conf = self.conf._replace(sample_len=sample_len)
        if batch is not None:
            self.conf = self.conf._replace(batch=batch)
        if t is not None:
            self.conf = self.conf._replace(t=t)

    def add_layers(self, layers_spec):
        """Fix pretrained weights and add more layers."""
        # Drop the output layer
        model = copy(self.conf.model)
        model.add_layers(layers_spec)
        self.conf = self.conf._replace(model=model)

        self.train_from = len(self.weights) - 1
        weights = self.model.init_weights(self._rng, 1.0)
        weights[:self.train_from] = self.weights[:-1]
        self.weights = weights

    def init(self, quiet=False, training=True):
        logging.info("Conf: " + conf_to_string(self.conf))

        if self.train_from == 0:
            self._loss_avg = self.model.loss_batch
        else:
            self._loss_avg = lambda w, batch: self.model.loss_batch(
                self.weights[:self.train_from] + w, batch
            )

        if training:
            if not quiet:
                logging.info("Creating loss_grad")
            self._loss_grad = jax.jit(jax.value_and_grad(self._loss_avg))
            if not quiet:
                logging.info("Creating optimizer")
            mask_bias = lambda tree: jax.tree_util.tree_map(
                lambda g: len(g.shape) > 1, tree
            )
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(self.conf.lr, mask=mask_bias, weight_decay=0.01),
            )

            self.opt_state = self.optimizer.init(self.weights[self.train_from:])
        else:
            self._loss_grad = None
            self.optimizer = None
            self.opt_state = None

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
        return self.model.loss_batch(self.weights, xs)

    def evaluate(self, xs):
        shards = 4
        while shards <= xs.shape[0]:
            logging.info(f"Evaluating with {shards} batches")
            shard_size = xs.shape[0] // shards

            loss = 0
            evaluation_failed = False
            for i in range(shards):
                shard = xs[i * shard_size : (i + 1) * shard_size]
                try:
                    shard = jax.nn.one_hot(shard, NCHAR)
                    loss += self.model.loss_batch(self.weights, shard).item()
                except (XlaRuntimeError, ValueError):
                    evaluation_failed = True
                    break

            if not evaluation_failed:
                self.loss = loss / shards
                return self.loss

            # Convert weights to numpy arrays and then release all the GPU buffers.
            self.weights = jax.device_get(self.weights)
            release_device_buffers()
            shards *= 2

        logging.info(
            "Can't evaluate the model even with shards 1. Assuming INF loss."
        )
        self.loss = INF
        return self.loss

    def sample(self, prefix, l, temperature=0.05):
        self._rng = Rng()
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
            c_selected = jax.random.choice(self._rng.gen(), NCHAR, p=c)
            c = jax.nn.one_hot(c_selected, NCHAR)
            out.append(c_selected)
        return bytes(out)

    def train_step(self, xs):
        xs = jax.nn.one_hot(xs, NCHAR)
        trainable_weights = self.weights[self.train_from:]
        loss, grads = self._loss_grad(trainable_weights, xs)

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, trainable_weights
        )
        trainable_weights = optax.apply_updates(trainable_weights, updates)
        self.weights[self.train_from:] = trainable_weights

        loss = float(loss)

        self.step += 1
        self.step_loss.append(loss)

        return loss

    def train(
        self,
        steps,
        time_limit,
        train_set,
        temp_steps=None,
        temp_dir=None,
        quiet=False,
    ):
        last_report = 0

        start = time.time()
        finish_time = start + time_limit if time_limit else INF

        if steps is None:
            steps = INF

        t = "" if time_limit is None else f" {time_limit} s"
        s = "" if steps > 1e10 else f" {steps} steps"
        logging.info(f"Training for{t}{s}")

        while time.time() < finish_time and self.step < steps:
            batch = train_set.sample(
                length=self.conf.sample_len, batch_size=self.conf.batch
            )
            loss = self.train_step(batch)
            # try:
            #     loss = self.train_step(batch)
            # except (XlaRuntimeError, ValueError):
            #     logging.warn(
            #         "Internal XLA error, probably OOM. Returning +inf loss."
            #     )
            #     raise TrainingDiverged

            if math.isnan(loss) or math.isinf(loss):
                raise TrainingDiverged
            if not quiet and (
                self.step < 10
                or (self.step % 10 == 0 and time.time() - last_report > 3)
                or time.time() - last_report > 10
            ):
                last_report = time.time()
                recent_losses = self.step_loss[-10:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"{self.step} {avg_loss:.4f} {self.step_loss[-1]:.4f}")

            if (
                temp_steps is not None
                and temp_steps > 0
                and self.step % temp_steps == 0
                and temp_dir is not None
            ):
                self.save(temp_dir)

        return time.time() - start

    def eval(self, dataset) -> float:
        """Evaluate a model on a random sample from the training data."""
        with latency.timer("Manager.eval"):
            batch = dataset.sample(self.test_sample_len, self.test_batch)
            return self.evaluate(batch)

    def train_and_eval(
        self,
        steps,
        time_limit,
        train_set,
        temp_steps,
        temp_dir,
        output_dir,
        log,
        quiet=False,
    ) -> TrainingRecord:
        try:
            train_time = self.train(
                steps,
                time_limit,
                train_set,
                temp_steps,
                temp_dir,
                quiet=quiet,
            )
            if output_dir is not None:
                self.save(output_dir)

            batch_loss = self.eval(train_set)
            if math.isnan(batch_loss):
                batch_loss = INF
        except TrainingDiverged:
            print("Training stopped early.")
            batch_loss = INF
            train_time = time_limit  # This is a hack, but we need to record
            # the loss with correct time.

        loss_model = StepLossPredictor()
        loss_model.fit(self.step_loss)

        report = TrainingRecord(
            timestamp=datetime.now(),
            model_spec=str(self.conf.model),
            weights=self.conf.model.weights,
            steps=self.step,
            train_time_s=train_time,
            learning_rate=self.conf.lr,
            regularization=1e-4,
            train_sample_len=self.conf.sample_len,
            train_batch=self.conf.batch,
            total_data=train_set.total_size,
            loss=batch_loss,
            test_sample_len=self.test_sample_len,
            test_batch=self.test_batch,
            test_poisoned=True,
            init_scale=1.0,
            planned_time_s=time_limit,
            final_time_s=time_limit,
            loss_model_v=loss_model.version,
            loss_model_params=loss_model.params(),
        )

        print(report)
        if log is not None:
            with open(log, "a", newline="") as logfile:
                writer = csv.writer(logfile)
                writer.writerow(report.csv_tuple())

        if quiet:
            # Clear all GPU memory
            backend = jax.lib.xla_bridge.get_backend()
            for buf in backend.live_buffers():
                buf.delete()

        return report

    def continue_prefix(self, prefix: str, length: int) -> str | bytes:
        prefix_bytes: bytes = prefix.encode()  # convert str to bytes (?)
        out = self.sample(prefix_bytes, length)

        try:
            s = (prefix_bytes + out).decode("utf-8")
        except UnicodeDecodeError:
            s = prefix_bytes + out
        return s

    def serialize_weights(self, weights):
        if weights is None:
            return None
        elif isinstance(weights, list):
            return [self.serialize_weights(w) for w in weights]
        elif type(weights) is dict:
            serialized = {}
            for key, value in weights.items():
                serialized[key] = self.serialize_weights(value)
            return serialized
        else:
            return weights.tolist()

    def name(self):
        model_name = str(self.model)
        return f"{model_name}-{self.step}"
