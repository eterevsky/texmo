"""JAX runtime for Model2.

Mirrors `ModelJax`'s public surface (init_weights, step,
initial_step, forward, loss_batch, loss_batch_masked,
forward_recurrent, loss_batch_masked_recurrent) but routes the
layer chain through a single `LayerSeqJax` instead of a flat
`list[LayerJax]` plus skip-target bookkeeping. Skip translation
is the parser's job -- by the time anything reaches Model2Jax, the
spec is split-form (or skip-free).

Weights layout: `[input_weights, layer_seq_weights, output_weights]`.
All input layers share the weighted signature (weights first, like
regular layers): the byte/bit encodings are parameter-free and return
None from init_weights / ignore the argument, while tokens.*.emb puts
its embedding table in slot 0. `layer_seq_weights` is itself a list
(one entry per child layer in the LayerSeqJax), so the full pytree is
`[input_or_None, [w_layer_0, w_layer_1, ...], w_output]` -- JAX treats
nested lists/dicts uniformly.

States layout: `[input_state, layer_seq_state]` where
`layer_seq_state` is a list of per-layer states.
"""

import math

import jax
import jax.numpy as jnp
import optax

from .layers.dense import DenseJax
from .layers.input_bits import InputBitsJax
from .layers.seq import LayerSeqJax

_1_BY_LOG2 = 1.0 / math.log(2.0)


class Model2Jax:
    def __init__(
        self,
        input_layer: InputBitsJax,
        layer_seq: LayerSeqJax,
        output: DenseJax,
        ntokens: int,
        total_padding: int = 1,
    ):
        self.input = input_layer
        self.layer_seq = layer_seq
        self.output = output
        self.ntokens = ntokens
        self._pad_output = ntokens <= 2
        self._total_padding = total_padding

    def init_weights(self, rng: jax.Array):
        k_in, k_seq, k_out = jax.random.split(rng, 3)
        return [
            self.input.init_weights(k_in),
            self.layer_seq.init_weights(k_seq),
            self.output.init_weights(k_out),
        ]

    def initial_step(self, weights) -> tuple[list, jax.Array]:
        """Predict the first token before any input has been seen.

        Mirrors ModelJax.initial_step: feeds `total_padding` initial
        vectors through the chain to warm up stateful sub-layers,
        then projects the last activation through the output dense.
        """
        input_state = self.input.init_state()
        layer_seq_state = self.layer_seq.init_state()

        p = self._total_padding
        v = None
        for i in range(p):
            v = self.input._initial_vector(weights[0], position=-p + i)
            layer_seq_state, v = self.layer_seq.step(
                weights[1], layer_seq_state, v)

        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return [input_state, layer_seq_state], logits

    def step(
        self, weights, states, token,
    ) -> tuple[list, jax.Array]:
        input_state, v = self.input.step(weights[0], states[0], token)
        layer_seq_state, v = self.layer_seq.step(
            weights[1], states[1], v)
        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return [input_state, layer_seq_state], logits

    def forward(self, weights, batch: jax.Array) -> jax.Array:
        v = self.input.forward(
            weights[0], batch[:, :-1], padding=self._total_padding)
        v = self.layer_seq.forward(weights[1], v)
        logits = self.output.forward(weights[-1], v)
        if self._pad_output:
            logits = jnp.pad(logits, ((0, 0), (0, 0), (0, 1)))
        return logits

    def loss_batch(self, weights, batch: jax.Array) -> jax.Array:
        logits = self.forward(weights, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        return _1_BY_LOG2 * jnp.mean(loss)

    def loss_batch_masked(
        self, weights, batch: jax.Array, lengths: jax.Array,
    ) -> jax.Array:
        logits = self.forward(weights, batch)
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        mask = jnp.arange(per_token.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(per_token * mask)

    def forward_recurrent(
        self, weights, batch: jax.Array,
    ) -> jax.Array:
        """Same logits as forward(), computed via scanned step().

        Same trick as ModelJax: vmap one step across the batch axis
        and lax.scan along time, so msr-style layers can use their
        per-step matrix-state path instead of the parallel-form
        O(T^2) scores tensor.
        """
        batch_size = batch.shape[0]
        init_state, init_logits = self.initial_step(weights)

        def add_batch(x):
            arr = jnp.asarray(x)
            return jnp.broadcast_to(arr, (batch_size,) + arr.shape)
        batched_state = jax.tree.map(add_batch, init_state)

        batched_step = jax.vmap(
            lambda s, t: self.step(weights, s, t),
            in_axes=(0, 0),
        )
        inputs_t = jnp.transpose(batch[:, :-1], (1, 0))

        def scan_fn(state, token_t):
            return batched_step(state, token_t)

        _, logits_t = jax.lax.scan(scan_fn, batched_state, inputs_t)

        init_logits_batched = jnp.broadcast_to(
            init_logits, (batch_size,) + init_logits.shape)
        all_logits = jnp.concatenate(
            [init_logits_batched[None], logits_t], axis=0)
        return jnp.transpose(all_logits, (1, 0, 2))

    def loss_batch_masked_recurrent(
        self, weights, batch: jax.Array, lengths: jax.Array,
    ) -> jax.Array:
        logits = self.forward_recurrent(weights, batch)
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        mask = jnp.arange(per_token.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(per_token * mask)
