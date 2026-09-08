"""JAX runtime for Model2.

Mirrors `ModelJax`'s public surface (init_weights, step,
initial_step, forward, loss_batch, loss_batch_masked,
forward_recurrent, loss_batch_masked_recurrent) but routes the
layer chain through a single `LayerSeqJax` instead of a flat
`list[LayerJax]` plus skip-target bookkeeping. Skip translation
is the parser's job -- by the time anything reaches Model2Jax, the
spec is split-form (or skip-free).

Weights layout: `[input_weights, layer_seq_weights, output_weights]`.
Slots 0 and 2 belong to the codec (see layers/codec.py): OneHotCodec's
and PairCodec's fixed codebooks are parameter-free and keep None in
slot 0, EmbeddingCodec puts its table there, and slot 2 holds the head.
`layer_seq_weights` is itself a list (one entry per child layer in the
LayerSeqJax), so the full pytree is
`[input_or_None, [w_layer_0, w_layer_1, ...], w_head]` -- JAX treats
nested lists/dicts uniformly. Keeping the codec's two ends in separate
slots preserves the layout of existing pickled checkpoints.

States layout: `[input_state, layer_seq_state]` where
`layer_seq_state` is a list of per-layer states.
"""

import math

import jax
import jax.numpy as jnp
import optax

from .layers.embedding_codec import EmbeddingCodecJax
from .layers.one_hot_codec import OneHotCodecJax
from .layers.pair_codec import PairCodecJax
from .layers.seq import LayerSeqJax

_1_BY_LOG2 = 1.0 / math.log(2.0)


class Model2Jax:
    def __init__(
        self,
        codec: OneHotCodecJax | EmbeddingCodecJax | PairCodecJax,
        layer_seq: LayerSeqJax,
        total_padding: int = 1,
        layer_widths: frozenset[int] = frozenset(),
    ):
        self.codec = codec
        self.layer_seq = layer_seq
        self._total_padding = total_padding
        # Every activation width in the model (Model2Def.layer_widths).
        # forward_recurrent keeps its batch size off this set; see
        # `_recurrent_batch_size`.
        self._layer_widths = layer_widths

    @property
    def ntokens(self) -> int:
        return self.codec.ntokens

    def init_weights(self, rng: jax.Array):
        k_in, k_seq, k_out = jax.random.split(rng, 3)
        return [
            self.codec.init_input_weights(k_in),
            self.layer_seq.init_weights(k_seq),
            self.codec.init_head_weights(k_out),
        ]

    def initial_step(self, weights) -> tuple[list, jax.Array]:
        """Predict the first token before any input has been seen.

        Prefills the `total_padding` prefix in forward semantics (see
        LayerJax.prefill): the prefix is built exactly as
        `codec.encode(..., padding=p)` builds it, then walked through
        the chain layer by layer, each layer seeing the trimmed output
        of the one before it. The resulting state is therefore the
        state `forward` implicitly holds at position 0, and `logits0`
        equals forward's position-0 logits.

        This replaces a synchronous warm-up that ticked every layer p
        times. That walk fed a consuming layer's transient outputs --
        positions `forward` never emits -- into whatever stateful layer
        followed it, permanently polluting that state (and double-
        padding the consuming layer, which already zero-inits its own
        buffer). The prefix walk drops those transients the way
        training does.
        """
        input_state = self.codec.init_state()

        p = self._total_padding
        prefix = jnp.stack([
            self.codec.initial_vector(weights[0], position=-p + i)
            for i in range(p)
        ])
        layer_seq_state, v = self.layer_seq.prefill(weights[1], prefix)

        # The chain consumes p-1 of the p prefix positions, leaving the
        # single activation that predicts the first token.
        logits = self.codec.logits_step(weights[0], weights[-1], v[-1])
        return [input_state, layer_seq_state], logits

    def step(
        self, weights, states, token,
    ) -> tuple[list, jax.Array]:
        input_state, v = self.codec.encode_step(weights[0], states[0], token)
        layer_seq_state, v = self.layer_seq.step(
            weights[1], states[1], v)
        logits = self.codec.logits_step(weights[0], weights[-1], v)
        return [input_state, layer_seq_state], logits

    def forward(self, weights, batch: jax.Array) -> jax.Array:
        v = self.codec.encode(
            weights[0], batch[:, :-1], padding=self._total_padding)
        v = self.layer_seq.forward(weights[1], v)
        return self.codec.logits(weights[0], weights[-1], v)

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

    def _recurrent_batch_size(self, batch_size: int) -> int:
        """Batch size to actually run `forward_recurrent` at.

        Under `vmap(step)` every layer's dot is `[batch, width]`, so
        `batch == width` makes it square -- and a square dot whose
        output layout is `{0,1}` (which a `split.cat` merge downstream
        produces) is miscompiled by XLA:GPU into a cuBLASLt
        `epilogue:"BIAS"` that broadcasts the bias along the wrong
        axis. The corruption is silent and large: 10.8 b/B instead of
        1.36 for `models/hb32-8k-s3.json` at batch 32. See
        docs/findings.md, "XLA:GPU adds a fused bias on the wrong axis
        (2026-08-29)".

        So round the batch up to the first size that is not any layer's
        width; the extra rows are dummies, dropped before the caller
        (and before any loss reduction) sees the logits. Widths are
        powers of two in practice, so this costs at most one row.
        """
        while batch_size in self._layer_widths:
            batch_size += 1
        return batch_size

    def forward_recurrent(
        self, weights, batch: jax.Array,
    ) -> jax.Array:
        """Same logits as forward(), computed via scanned step().

        Same trick as ModelJax: vmap one step across the batch axis
        and lax.scan along time, so msr-style layers can use their
        per-step matrix-state path instead of the parallel-form
        O(T^2) scores tensor.

        The batch may be padded with dummy rows before the scan (see
        `_recurrent_batch_size`); they are sliced off again here, so
        the returned logits always match `batch`'s first axis and no
        caller has to know about it.
        """
        n_real = batch.shape[0]
        padded = self._recurrent_batch_size(n_real)
        if padded != n_real:
            # Token id 0 is valid for every codec, and these rows are
            # discarded below -- they exist only to change the dot
            # shapes.
            pad = jnp.zeros(
                (padded - n_real,) + batch.shape[1:], dtype=batch.dtype)
            batch = jnp.concatenate([batch, pad], axis=0)
            return self._forward_recurrent(weights, batch)[:n_real]
        return self._forward_recurrent(weights, batch)

    def _forward_recurrent(
        self, weights, batch: jax.Array,
    ) -> jax.Array:
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
