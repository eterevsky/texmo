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
from .common import INF, ttoa3, itoa3
from .configuration2 import Configuration2
from .dataset import DataSet, DataSetWrapper

from .model3 import Model3, Weights
from .prng import Rng
from .record import TrainingRecord
from .run import Run
from .predict import LossTrend
from .tokens import Tokenizer

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
        dataset: DataSet,
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

        assert isinstance(dataset, DataSet) or isinstance(dataset, DataSetWrapper)
        self.dataset: DataSet = dataset

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

        # Full training data, will be initialized in init() if self.sync_tokens
        # is True.
        self._training_data: Optional[jax.Array] = None

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
        model_name = model_name.replace("|", "!")
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

    def _init_training_data(self):
        pass
        # with latency.timer("Manager._init_training_data") as timer:
        #     self._training_data = self.dataset.sample_tokens(
        #             ntokens=self.conf.length,
        #             batch=self.conf.batch * self.conf.steps,
        #             tokenset_name=self.conf.tokens_name,
        #         )
        #     self._training_data = jnp.array(self._training_data)
        #     total_tokens = self.conf.length * self.conf.batch * self.conf.steps
        #     total_bytes = round(
        #         total_tokens * self.tokenizer.tokenset.avg_bytes_per_token
        #     )
        #     logging.info(
        #         f"Prepared {itoa3(total_tokens)}T / ~{itoa3(total_bytes)}B"
        #         + f" of training data in {ttoa3(timer.elapsed)}"
        #     )

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
            self._init_training_data()
        else:
            self._loss_grad = None
            self.optimizer = None
            self.opt_state = None

    def _eval(self, xs, lengths):
        lengths = np.array(lengths)

        assert self.tokenizer is not None

        shards = 4
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
                return loss / (self.test_sample_len * self.test_batch)

            # Convert weights to numpy arrays and then release all the GPU buffers.
            self.weights = jax.device_get(self.weights)
            release_device_buffers()
            shards *= 2

        # TODO: When we can't run eval with forward, we could just run it
        # step-by-step.

        logging.info("Can't evaluate the model even with shards 1. Assuming INF loss.")
        return INF

    def eval(self) -> float:
        """Evaluate a model on a random sample from the training data."""
        with latency.timer("Manager.eval"):
            batch, lengths = self.dataset.sample_bytes(
                nbytes=self.test_sample_len,
                batch=self.test_batch,
                tokenset_name=self.tokenizer.tokenset.name,
            )
            return self._eval(batch, lengths)

    def sample_(self, prefix, l, temperature=0.05):
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
            state, c = self.model.step_sample(
                self.weights, state, c, self._rng, temperature
            )
            out.append(c)
        return out

    def sample(self, prefix, l, temperature=0.05):
        """Sample from the distribution to continue the given prefix."""
        self._rng = Rng()
        state = self.model.init_state(self.weights)
        for c in prefix[:-1]:
            state, _ = self.model.step(self.weights, state, c)

        c = prefix[-1]
        out = []
        while len(out) < l:
            state, c = self.model.step_sample(
                self.weights, state, c, self._rng, temperature
            )
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
        self.run.add_step(loss)

        return loss / (xs.shape[0] * xs.shape[1])

    def _get_batch(self) -> jax.Array:
        with latency.timer("Manager._get_batch"):
            return self.dataset.sample_tokens(
                ntokens=self.conf.length,
                batch=self.conf.batch,
                tokenset_name=self.conf.tokens_name,
            )
        # assert self._training_data is not None
        # start = self.step * self.conf.batch
        # finish = start + self.conf.batch
        # assert finish <= self._training_data.shape[0]
        # return self._training_data[start:finish, :]

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        temp_steps=None,
        temp_dir=None,
        quiet=False,
        soft_tl: Optional[float] = None,
    ) -> tuple[float, Configuration2]:
        last_report = 0  # Timestamp of the last printed report

        if steps is None and time_limit is None:
            steps = self.conf.steps

        if steps is None:
            steps = INF

        t = "" if time_limit is None else f" {time_limit} s"
        s = "" if steps > 1e10 else f" {steps} steps"

        logging.info(f"Training for{t}{s}")
        deadline = INF
        soft_deadline = INF
        start_time = None

        # Checking deadline, step limit and soft deadline when the number of
        # steps is a power of 2.

        while self.step < steps:
            if perf_counter() > deadline:
                logging.info(
                    f"Stopped at step {self.step} due to hard time limit {ttoa3(time_limit)}"
                )
                break
            if self.step & (self.step - 1) != 0 and perf_counter() > soft_deadline:
                logging.info(
                    f"Stopped at step {self.step} / {steps} due to soft time limit {ttoa3(soft_tl)}"
                )
                break

            batch = self._get_batch()

            # try:
            loss = self.train_step(batch)
            # except (XlaRuntimeError, ValueError) as e:
            #     logging.warn("Internal XLA error, probably OOM:\n" + str(e))
            #     break

            step_end = perf_counter()

            if math.isnan(loss) or math.isinf(loss):
                logging.warning(f"Loss is {loss}, stopping training")
                # Don't register the training time if it diverged.
                start_time = None
                break

            if not quiet and (
                self.step < 10
                or (self.step % 10 == 0 and step_end - last_report > 3)
                or step_end - last_report > 10
            ):
                last_report = step_end
                logging.info(self.run.report_recent_loss(self.tokenizer.tokenset))

            if (
                temp_steps is not None
                and temp_steps > 0
                and self.step % temp_steps == 0
                and temp_dir is not None
            ):
                self.save(temp_dir)

            if start_time is None:
                start_time = step_end
                deadline = start_time + time_limit if time_limit else INF
                soft_deadline = start_time + soft_tl if soft_tl else INF

        total_time = None if start_time is None else perf_counter() - start_time

        return total_time, self.conf.replace(steps=self.step)

    def train_and_eval(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
        temp_steps,
        temp_dir,
        output_dir,
        log,
        quiet: bool = False,
        soft_tl: Optional[float] = None,
    ) -> tuple[Run, Weights, Configuration2]:
        """Train a model and evaluate it.

        Args:
            soft_tl: if not None, the training will stop as soon as the total
                spent time is above soft_tl AND the total number of steps
                is a power of 2
        """
        try:
            train_time, final_conf = self.train(
                steps,
                time_limit,
                temp_steps,
                temp_dir,
                quiet=quiet,
                soft_tl=soft_tl,
            )

            if train_time is None:
                # Training diverged
                eval_loss = INF

            if output_dir is not None:
                self.save(output_dir)

            eval_loss = self.eval()
            if math.isnan(eval_loss):
                eval_loss = INF
        except TrainingDiverged:
            logging.warning("Training stopped early.")
            eval_loss = INF

        self.run.finalize(eval_loss, train_time)
        logging.info(
            f"{self.conf}:  loss {eval_loss:.4f} b/byte  T = {ttoa3(train_time)}"
        )

        return (self.run, self.weights, final_conf)

    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> str | bytes:
        prefix_bytes: bytes = prefix.encode()  # convert str to bytes
        prefix_tokens = self.tokenizer.tokenize(prefix_bytes)

        out = self.sample(prefix_tokens, length, temperature)
        full_text = list(prefix_tokens) + out

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
