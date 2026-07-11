import logging
import random
from time import perf_counter
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .common import ttoa3
from .configuration import Configuration
from .manager import Manager
from .model import build_model_def
from .model2 import Model2Def
from .model_jax import ModelJax, Weights
from .precision import Precision
from .predict import LossTrend
from .run import Run
from .spec_parser import parse_model2
from .tokens import get_tokenizer

# Number of training steps batched into a single JIT'd `lax.scan`.
# Removes the per-step Python overhead and the host-sync caused by
# `float(loss)` in the loop -- both of which dominate for tiny models
# on accelerators. Caps the data tensor size so it stays comfortable
# even for the widest configs (256 steps * 64 batch * 8192 length is
# ~128 MB of int8).
_CHUNK_SIZE = 256

# Minimum wall-clock seconds between progress logs in quiet mode.
# Verbose mode still prints every chunk.
_QUIET_PROGRESS_INTERVAL = 30.0


class ManagerJax(Manager):
    """JAX training backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dtype = self.conf.precision.jax_dtype

        logging.info(f'{self.conf}')
        self.model: ModelJax = self.model_def.build_jax()
        rng = jax.random.PRNGKey(random.randrange(2**32))
        self.weights: Weights = self.model.init_weights(rng)

        if self.verbose:
            logging.info('Creating optimizer')
        self._build_optimizer()
        self.run = Run(loss_trend=LossTrend(), system=self.system)

        # JIT-compile combined loss+gradient computation (single fwd+bwd pass).
        self._loss_grad = jax.jit(jax.value_and_grad(self.model.loss_batch))
        # JIT-compile the per-token inference step. Without this,
        # continue_prefix and byte_distribution pay JAX's dispatch
        # overhead (~10 ms) on every model.step call, which dominates
        # for any reasonable continuation length.
        self._step_jit = jax.jit(self.model.step)
        # JIT-compile a chunk of training steps (one fwd+bwd+update
        # per scan iteration). One Python dispatch + one host sync per
        # `_CHUNK_SIZE` steps, which is the main speedup vs the per-
        # step loop in `Manager.train`.
        self._train_chunk = jax.jit(self._build_train_chunk_fn())

    def _build_train_chunk_fn(self):
        loss_fn = self.model.loss_batch
        optimizer = self.optimizer

        def chunk(weights, opt_state, batches):
            def step(carry, batch):
                w, opt = carry
                loss, grads = jax.value_and_grad(loss_fn)(w, batch)
                updates, opt = optimizer.update(grads, opt, w)
                w = optax.apply_updates(w, updates)
                return (w, opt), loss
            (w_out, opt_out), losses = jax.lax.scan(
                step, (weights, opt_state), batches)
            return w_out, opt_out, losses

        return chunk

    def _build_optimizer(self):
        lr = self.conf.lr
        if self.conf.cosine:
            lr = optax.cosine_decay_schedule(
                init_value=self.conf.lr,
                decay_steps=self.conf.steps,
                alpha=0.0,
            )
        elif self.conf.decay != 1:
            initial_lr = self.conf.lr
            decay = self.conf.decay
            steps = self.conf.steps
            lr = lambda count: initial_lr * decay ** (count / steps)

        eps = 1e-4 if self.conf.precision == Precision.FP16 else 1e-8
        # Apply weight decay only to weight matrices, not biases.
        mask_bias = lambda tree: jax.tree.map(
            lambda x: x.ndim >= 2, tree)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, weight_decay=0.01, eps=eps, mask=mask_bias),
        )
        self._opt_state = self.optimizer.init(self.weights)

    def _get_batch(self):
        data = self.dataset.sample_tokens(
            ntokens=self.conf.length,
            batch=self.conf.batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        return jnp.asarray(data)

    def train_step(self, batch) -> float:
        loss, grads = self._loss_grad(self.weights, batch)

        updates, self._opt_state = self.optimizer.update(
            grads, self._opt_state, self.weights)
        self.weights = optax.apply_updates(self.weights, updates)

        loss_val = float(loss) / self.bytes_per_token
        self.run.add_step(loss_val)
        return loss_val

    # Toggleable from the train CLI for A/B benchmarking. When False,
    # falls back to the per-step `Manager.train` loop (which still
    # uses our `train_step` and so supports time_limit, divergence
    # early-exit, etc.).
    scan_train: bool = True

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
    ) -> tuple[Optional[float], Configuration]:
        """JAX-specific training: batches `_CHUNK_SIZE` steps into a
        JIT'd `lax.scan`. Drops the per-step loop's host sync, which
        is the main bottleneck for tiny models on accelerators.

        time_limit is unsupported on this path -- the search doesn't
        use it, and a chunk runs uninterruptibly. Wall-clock time
        includes the first chunk (which carries JIT-compile cost on
        the first occurrence of each unique tensor shape); the DB's
        median across runs filters the resulting outliers.

        NaN losses propagate -- once the model diverges, subsequent
        steps run with NaN updates and the final eval will catch it.
        We don't try to early-exit.
        """
        if not self.scan_train:
            return super().train(steps, time_limit)
        if steps is None:
            steps = self.conf.steps
        if time_limit is not None:
            logging.warning(
                'time_limit is not supported with the chunked JAX '
                'trainer; ignoring. Pass --no-scan to use the per-step '
                'loop instead.')
        logging.info(f'Training for {steps} steps')

        batch = self.conf.batch
        length = self.conf.length
        tokens_name = self.model_def.input.tokens_name

        start_time = perf_counter()
        last_progress_log = start_time
        while self.step < steps:
            n = min(_CHUNK_SIZE, steps - self.step)
            # One big sample of (n*batch, length), reshaped to
            # (n, batch, length). Avoids n round-trips through the
            # prefetch queue and n separate host->device transfers.
            data = self.dataset.sample_tokens(
                ntokens=length,
                batch=n * batch,
                tokenset_name=tokens_name,
            )
            batches = jnp.asarray(data).reshape(n, batch, length)
            self.weights, self._opt_state, losses = self._train_chunk(
                self.weights, self._opt_state, batches)
            losses_host = np.asarray(losses)
            for loss_val in losses_host:
                self.run.add_step(float(loss_val) / self.bytes_per_token)
            now = perf_counter()
            if (
                self.verbose
                or now - last_progress_log >= _QUIET_PROGRESS_INTERVAL
            ):
                last = float(losses_host[-1]) / self.bytes_per_token
                logging.info(f'{self.step}  {last:.4f} b/B')
                last_progress_log = now

        total_time = perf_counter() - start_time
        logging.info(f'Trained for {self.step} steps in {ttoa3(total_time)}')
        return total_time, self.conf.replace(steps=self.step)

    def eval(self) -> float:
        """Evaluate the model on a random test batch in fp32.

        Samples by bytes (not tokens) so the eval metric is consistent
        across tokenizations with different bytes-per-token ratios.
        """
        weights32 = jax.tree.map(
            lambda x: x.astype(jnp.float32), self.weights)
        # Rebuild via the same factory the original used so the
        # weights pytree matches (Model2Jax and ModelJax weight
        # layouts differ slightly).
        if isinstance(self.model_def, Model2Def):
            model32 = parse_model2(
                self.model_def.spec, Precision.FP32,
                cap=self.model_def.codec.cap).build_jax()
        else:
            model32 = build_model_def(
                self.model_def.spec, Precision.FP32).build_jax()

        batch, lengths = self.dataset.sample_bytes(
            nbytes=self.test_sample_len,
            batch=self.test_batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        # Recurrent form avoids the O(B*H*L^2) scores tensor that the
        # parallel forward materializes — required to fit 1024x1024 eval
        # in 16 GB on systems with msr layers.
        total_loss = model32.loss_batch_masked_recurrent(
            weights32, batch, lengths)
        return float(total_loss) / (self.test_sample_len * self.test_batch)

    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> bytes:
        """Sample text continuation from the model."""
        tokenizer = get_tokenizer(self.model_def.input.tokens_name)
        prefix_tokens = tokenizer.tokenize(prefix.encode())

        states, _ = self.model.initial_step(self.weights)
        for c in prefix_tokens[:-1]:
            states, _ = self._step_jit(self.weights, states, int(c))

        c = int(prefix_tokens[-1])
        rng = jax.random.PRNGKey(random.randrange(2**32))
        out = []
        for _ in range(length):
            states, logits = self._step_jit(self.weights, states, c)
            probs = jax.nn.softmax(logits / temperature)
            rng, sub = jax.random.split(rng)
            c = int(jax.random.choice(sub, self.model.ntokens, p=probs))
            out.append(c)

        return tokenizer.untokenize(list(prefix_tokens) + out)
