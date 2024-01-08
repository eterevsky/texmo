import csv
import json
import logging
import math
import os
import time
from copy import copy
from datetime import datetime
from typing import Optional
from time import perf_counter
from statistics import mean

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxlib.xla_extension import XlaRuntimeError

from . import latency
from .common import INF, ttoa3
from .configuration2 import Configuration2
from .dataset import DataSet

# from .model2 import Model2, Weights
from .model3 import Model3, Weights
from .prng import Rng
from .record import TrainingRecord
from .run import Run
from .predict import SampleTiming, LossTrend, TrainTiming
from .tokens import Tokenizer, TokenSet, get_tokenizer

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
        conf: Configuration2,
        system: str,
        weights: Optional[Weights] = None,
        test_sample_len: int = 1024,
        test_batch: int = 1024,
        pre_training: Optional[list] = None,
    ):
        self._rng: Rng = Rng()
        assert isinstance(system, str), f"`system` must be a string, got {system}"
        self._system: str = system
        assert isinstance(conf, Configuration2)
        self.conf: Configuration2 = conf
        if weights is None:
            weights = self.model.init_weights(self._rng, 1.0)
        self.weights: Weights = weights
        # Only layers starting from this one will be trained
        self.train_from: int = 0

        self.test_sample_len: int = test_sample_len
        self.test_batch: int = test_batch
        if pre_training is not None and not isinstance(pre_training, list):
            pre_training = [pre_training]
        self.pre_training: Optional[list] = pre_training
        self.run: Optional[Run] = None

    # def __del__(self):
    #     # Clear all GPU memory
    #     release_device_buffers()

    @property
    def step(self):
        return self.run.steps

    @property
    def loss(self) -> float:
        return self.run.loss

    @property
    def ntokens(self) -> int:
        return self.conf.model.input.ntokens

    @property
    def model(self) -> Model3:
        return self.conf.model

    @property
    def tokenizer(self) -> Tokenizer:
        return self.conf.model.input.tokenizer

    def save(self, dir):
        model_name = self.name()
        path = os.path.join(dir, f"{model_name}.json")

        weights = self.serialize_weights(self.weights)

        if self.pre_training:
            training = self.pre_training
            if not isinstance(training, list):
                training = [training]
        else:
            training = []
        training.append(
            {
                "conf": self.conf.to_dict(),
                "run": self.run.to_dict(),
            }
        )

        data = {
            "conf": self.conf.to_dict(),
            "training": training,
            "weights": weights,
        }

        print(f"Saving model to {path}")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path, token_set=None, training=True):
        with open(path) as f:
            spec = json.load(f)

        conf = Configuration2.from_dict(spec["conf"])
        weights = deserialize_weights(spec["weights"])

        manager = Manager(
            conf,
            token_set=token_set,
            weights=weights,
            pre_training=spec["training"],
        )

        return manager

    # def update_conf(self, lr, sample_len, batch, t):
    #     """Updates the configuration with new parameters.

    #     Useful after loading a model.
    #     """
    #     if lr is not None:
    #         self.conf = self.conf._replace(lr=lr)
    #     if sample_len is not None:
    #         self.conf = self.conf._replace(sample_len=sample_len)
    #     if batch is not None:
    #         self.conf = self.conf._replace(batch=batch)
    #     if t is not None:
    #         self.conf = self.conf._replace(t=t)

    def add_layers(self, layers_spec):
        """Fix pretrained weights and add more layers."""
        # Drop the output layer
        model = copy(self.conf.model)
        model.add_layers(layers_spec)
        self.conf = self.conf.replace(model=model)

        self.train_from = len(self.weights) - 1
        weights = self.model.init_weights(self._rng, 1.0)
        weights[: self.train_from] = self.weights[:-1]
        self.weights = weights

    def init(self, quiet=False, training=True):
        logging.info(f"Conf: {self.conf}")

        if self.train_from == 0:
            self._loss_avg = self.model.loss_batch
        else:
            self._loss_avg = lambda w, batch: self.model.loss_batch(
                self.weights[: self.train_from] + w, batch
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

            self.opt_state = self.optimizer.init(self.weights[self.train_from :])
            self.run = Run(loss_trend=LossTrend(), system=self._system)
        else:
            self._loss_grad = None
            self.optimizer = None
            self.opt_state = None

    def _eval(self, xs, lengths):
        lengths = np.array(lengths)

        assert self.tokenizer is not None
        if self.tokenizer.token_set.entropy0 > 0:
            zeros = 0
            for sample, l in zip(xs, lengths):
                zeros += l - np.count_nonzero(sample[:l])
            total_entropy0 = zeros * self.tokenizer.token_set.entropy0
        else:
            total_entropy0 = 0

        shards = 16
        while shards <= xs.shape[0]:
            logging.info(f"Evaluating with {shards} batches")
            shard_size = xs.shape[0] // shards

            loss = 0
            evaluation_failed = False
            for i in range(shards):
                start = i * shard_size
                end = (i + 1) * shard_size
                shard = xs[start:end]
                shard_len = lengths[start:end]
                try:
                    # shard = jax.nn.one_hot(shard, self.ntokens)
                    loss += self.model.loss_batch_masked(
                        self.weights, shard, shard_len
                    ).item()
                except (XlaRuntimeError, ValueError):
                    evaluation_failed = True
                    break

            if not evaluation_failed:
                return (loss + total_entropy0) / (
                    self.test_sample_len * self.test_batch
                )

            # Convert weights to numpy arrays and then release all the GPU buffers.
            self.weights = jax.device_get(self.weights)
            release_device_buffers()
            shards *= 2

        # TODO: When we can't run eval with forward, we could just run it
        # step-by-step.

        logging.info("Can't evaluate the model even with shards 1. Assuming INF loss.")
        return INF

    def eval(self, dataset: DataSet) -> float:
        """Evaluate a model on a random sample from the training data."""
        with latency.timer("Manager.eval"):
            batch, lengths = dataset.sample_bytes(
                length=self.test_sample_len,
                batch_size=self.test_batch,
                token_set_name=self.tokenizer.token_set.name,
            )
            return self._eval(batch, lengths)

    def sample(self, prefix, l, temperature=0.05):
        """Sample from the distribution to continue the given prefix."""
        self._rng = Rng()
        prefix = jnp.array(list(prefix))
        # prefix = jax.nn.one_hot(prefix, self.ntokens)
        c = prefix[-1]

        state = self.model.init_state(self.weights)
        state, _ = jax.lax.scan(
            lambda s, c: self.model.step(self.weights, s, c), state, prefix[:-1]
        )

        out = []
        while len(out) < l:
            state, c = self.model.step_sample(self.weights, state, c, self._rng, temperature)
            out.append(c)
        return out

    def train_step(self, xs):
        trainable_weights = self.weights[self.train_from :]
        loss, grads = self._loss_grad(trainable_weights, xs)

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, trainable_weights
        )
        trainable_weights = optax.apply_updates(trainable_weights, updates)
        self.weights[self.train_from :] = trainable_weights

        loss = float(loss)
        byte_loss = self.tokenizer.token_set.byte_loss(loss)
        self.run.add_step(byte_loss)

        return loss

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        dataset: DataSet,
        temp_steps=None,
        temp_dir=None,
        quiet=False,
    ):
        last_report = 0  # Timestamp of the last printed report

        if steps is None and time_limit is None:
            steps = self.conf.steps

        if steps is None:
            steps = INF

        t = "" if time_limit is None else f" {time_limit} s"
        s = "" if steps > 1e10 else f" {steps} steps"

        sample_times = []
        step_times = []

        logging.info(f"Training for{t}{s}")
        finish_time = INF
        start = None

        step_start = perf_counter()

        while perf_counter() < finish_time and self.step < steps:
            batch, sample_time = dataset.sample_tokens(
                ntokens=self.conf.length,
                batch_size=self.conf.batch,
                token_set_name=self.conf.tokens_name,
            )

            sample_times.append(sample_time)

            # try:
            loss = self.train_step(batch)
            # except (XlaRuntimeError, ValueError) as e:
            #     logging.warn("Internal XLA error, probably OOM. Returning +inf loss:\n" + str(e))
            #     step_times = []
            #     break

            step_end = perf_counter()
            step_time = step_end - step_start
            step_times.append(step_time)
            step_start = step_end  # Counting output in the next step time

            if math.isnan(loss) or math.isinf(loss):
                logging.warning(f"Loss is {loss}, stopping training")
                break

            if not quiet and (
                self.step < 10
                or (self.step % 10 == 0 and step_end - last_report > 3)
                or step_end - last_report > 10
            ):
                last_report = step_end
                logging.info(self.run.report_recent_loss(self.tokenizer.token_set))

            if (
                temp_steps is not None
                and temp_steps > 0
                and self.step % temp_steps == 0
                and temp_dir is not None
            ):
                self.save(temp_dir)

            if len(step_times) == 1:
                logging.info(
                    f"First training step took {ttoa3(step_times[0])}. "
                    + "Disregarded for time limit."
                )
                start = step_start
                finish_time = start + time_limit if time_limit else INF

        total_time = 0 if start is None else perf_counter() - start
        return total_time, sample_times, step_times

    def train_and_eval(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        train_set,
        temp_steps,
        temp_dir,
        output_dir,
        log,
        quiet=False,
    ) -> tuple[TrainingRecord, Run]:
        try:
            train_time, sample_times, step_times = self.train(
                steps,
                time_limit,
                train_set,
                temp_steps,
                temp_dir,
                quiet=quiet,
            )
            if output_dir is not None:
                self.save(output_dir)

            eval_loss = self.eval(train_set)
            if math.isnan(eval_loss):
                eval_loss = INF
        except TrainingDiverged:
            logging.warning("Training stopped early.")
            eval_loss = INF

        self.run.finalize(eval_loss, train_time)

        # step_time = train_time / (len(step_times) - 1)

        logging.info(f"{self.conf}:  loss {eval_loss:.4f} b/byte  T = {ttoa3(train_time)}")

        return (self.run, self.weights)

    def continue_prefix(self, prefix: str, length: int, temperature: float) -> str | bytes:
        prefix_bytes: bytes = prefix.encode()  # convert str to bytes
        prefix_tokens = [t.id for t in self.tokenizer.tokenize(prefix_bytes)]

        out = self.sample(prefix_tokens, length, temperature)

        full_text = prefix_tokens + out

        out = self.tokenizer.untokenize(full_text) if self.tokenizer else bytes(out)

        return out

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

        total_steps = self.step

        if self.pre_training:
            for stage in self.pre_training:
                total_steps += stage["step"]
        return f"{model_name}-{self.step}"
