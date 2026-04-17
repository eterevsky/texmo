import logging
import random

import jax
import jax.numpy as jnp
import optax

from .manager import Manager
from .model import build_model_def
from .model_jax import ModelJax, Weights
from .predict import LossTrend
from .precision import Precision
from .run import Run
from .tokens import get_tokenizer


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

    def _build_optimizer(self):
        lr = self.conf.lr
        if self.conf.decay != 1:
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

    def eval(self) -> float:
        """Evaluate the model on a random test batch in fp32.

        Samples by bytes (not tokens) so the eval metric is consistent
        across tokenizations with different bytes-per-token ratios.
        """
        weights32 = jax.tree.map(
            lambda x: x.astype(jnp.float32), self.weights)
        model32 = build_model_def(
            self.model_def.spec, Precision.FP32).build_jax()

        batch, lengths = self.dataset.sample_bytes(
            nbytes=self.test_sample_len,
            batch=self.test_batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        total_loss = model32.loss_batch_masked(weights32, batch, lengths)
        return float(total_loss) / (self.test_sample_len * self.test_batch)

    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> bytes:
        """Sample text continuation from the model."""
        tokenizer = get_tokenizer(self.model_def.input.tokens_name)
        prefix_tokens = tokenizer.tokenize(prefix.encode())

        states, _ = self.model.initial_step(self.weights)
        for c in prefix_tokens[:-1]:
            states, _ = self.model.step(self.weights, states, c)

        c = prefix_tokens[-1]
        rng = jax.random.PRNGKey(random.randrange(2**32))
        out = []
        for _ in range(length):
            states, logits = self.model.step(self.weights, states, c)
            probs = jax.nn.softmax(logits / temperature)
            rng, sub = jax.random.split(rng)
            c = int(jax.random.choice(sub, self.model.ntokens, p=probs))
            out.append(c)

        return tokenizer.untokenize(list(prefix_tokens) + out)
