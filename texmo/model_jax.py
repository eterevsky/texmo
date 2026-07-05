import math

import jax
import jax.numpy as jnp
import optax

from .layer_jax import LayerJax, LayerState, LayerWeights
from .layers.dense import DenseJax
from .layers.input_bits import InputBitsJax
from .layers.skip import SkipJax

_1_BY_LOG2 = 1.0 / math.log(2.0)

Weights = list[LayerWeights]
State = list[LayerState]


def _merge_add(v: jax.Array, source: jax.Array) -> jax.Array:
    # Last-axis merge: max size, sum the overlap, append the tail from
    # whichever input is longer.
    d_v = v.shape[-1]
    d_s = source.shape[-1]
    if d_v == d_s:
        return v + source
    if d_v < d_s:
        head = v + source[..., :d_v]
        tail = source[..., d_v:]
    else:
        head = v[..., :d_s] + source
        tail = v[..., d_s:]
    return jnp.concatenate([head, tail], axis=-1)


def _merge_cat(v: jax.Array, source: jax.Array) -> jax.Array:
    return jnp.concatenate([v, source], axis=-1)


def _apply_merges(v, pending, skip_list):
    """Apply all merges in skip_list to v, popping sources from pending.

    Used in step mode — ignores shrinkage (the saved source is already
    a single-timestep vector).
    """
    for start_pos, op, _ in skip_list:
        source = pending.pop(start_pos)
        v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)
    return v


class ModelJax:
    """JAX model.

    Weights are a list: [input_weights, *layer_weights, output_weights].
    Input layers have no learned weights (input_weights is always None).
    """

    def __init__(
        self,
        input_layer: InputBitsJax,
        layers: list[LayerJax],
        output: DenseJax,
        ntokens: int,
        total_padding: int = 1,
        skip_targets: dict[int, list[tuple[int, str, int]]] | None = None,
    ):
        self.input = input_layer
        self.layers = layers
        self.output = output
        self.ntokens = ntokens
        self._pad_output = ntokens <= 2
        self._total_padding = total_padding
        # target position -> [(start_pos, op), ...] sorted by start_pos.
        self._skip_targets = skip_targets or {}

    def init_weights(self, rng: jax.Array) -> Weights:
        keys = jax.random.split(rng, len(self.layers) + 1)
        return (
            [None]  # input layer has no learned weights
            + [layer.init_weights(keys[i]) for i, layer in enumerate(self.layers)]
            + [self.output.init_weights(keys[-1])]
        )

    def _run_pipeline_step(self, v, weights, layer_states):
        """Run v through all layers (save/merge skips). Returns (v, new_states)."""
        pending: dict[int, jax.Array] = {}
        new_states: list[LayerState] = []
        for i, (layer, lw, ls) in enumerate(
            zip(self.layers, weights[1:-1], layer_states)):
            if i in self._skip_targets:
                v = _apply_merges(v, pending, self._skip_targets[i])
            if isinstance(layer, SkipJax):
                pending[i] = v
            ls, v = layer.step(lw, ls, v)
            new_states.append(ls)
        n = len(self.layers)
        if n in self._skip_targets:
            v = _apply_merges(v, pending, self._skip_targets[n])
        return v, new_states

    def initial_step(self, weights: Weights) -> tuple[State, jax.Array]:
        """Predict the first token (before any input)."""
        input_state = self.input.init_state()
        layer_states = [layer.init_state() for layer in self.layers]

        p = self._total_padding
        v = None
        for i in range(p):
            v = self.input._initial_vector(None, position=-p + i)
            v, layer_states = self._run_pipeline_step(v, weights, layer_states)

        states = [input_state] + layer_states
        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return states, logits

    def step(
        self, weights: Weights, states: State, token: int
    ) -> tuple[State, jax.Array]:
        """Run one step of inference."""
        input_state, v = self.input.step(None, states[0], token)
        v, new_layer_states = self._run_pipeline_step(v, weights, states[1:])
        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return [input_state] + new_layer_states, logits

    def forward(self, weights: Weights, batch: jax.Array) -> jax.Array:
        """Forward pass on a batch for training."""
        v = self.input.forward(None, batch[:, :-1], padding=self._total_padding)

        pending: dict[int, jax.Array] = {}
        for i, (layer, lw) in enumerate(zip(self.layers, weights[1:-1])):
            if i in self._skip_targets:
                for start_pos, op, shrinkage in self._skip_targets[i]:
                    source = pending.pop(start_pos)
                    if shrinkage > 0:
                        source = source[:, shrinkage:]
                    v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)
            if isinstance(layer, SkipJax):
                pending[i] = v
            v = layer.forward(lw, v)

        n = len(self.layers)
        if n in self._skip_targets:
            for start_pos, op, shrinkage in self._skip_targets[n]:
                source = pending.pop(start_pos)
                if shrinkage > 0:
                    source = source[:, shrinkage:]
                v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)

        logits = self.output.forward(weights[-1], v)
        if self._pad_output:
            logits = jnp.pad(logits, ((0, 0), (0, 0), (0, 1)))
        return logits

    def loss_batch(self, weights: Weights, batch: jax.Array) -> jax.Array:
        """Average cross-entropy loss in bits per token."""
        logits = self.forward(weights, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        return _1_BY_LOG2 * jnp.mean(loss)

    def loss_batch_masked(
        self, weights: Weights, batch: jax.Array, lengths: jax.Array
    ) -> jax.Array:
        """Total cross-entropy in bits, masked to actual sample lengths."""
        logits = self.forward(weights, batch)
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        mask = jnp.arange(per_token.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(per_token * mask)

    def forward_recurrent(
        self, weights: Weights, batch: jax.Array
    ) -> jax.Array:
        """Same logits as forward(), computed via scanned per-token step().

        For msr layers this trades the O(batch*heads*length^2) scores
        tensor for an O(batch*heads*dim^2) state matrix — a large memory
        and (on CPU at tiny dim) compute win. For non-msr layers it's
        the same step path used in inference, just batched and scanned,
        so no asymptotic change.
        """
        batch_size = batch.shape[0]

        init_state, init_logits = self.initial_step(weights)

        # Broadcast each leaf of the un-batched initial state to the
        # batch dim. tree.map skips None leaves automatically.
        # jnp.asarray converts Python-int leaves (e.g. the input
        # layer's position counter) into 0-d jnp scalars so they can
        # be broadcast and follow the vmap.
        def add_batch(x):
            arr = jnp.asarray(x)
            return jnp.broadcast_to(arr, (batch_size,) + arr.shape)
        batched_state = jax.tree.map(add_batch, init_state)

        # vmap one step over the batch axis (state and input token both
        # have axis 0). weights stay un-batched (shared across samples).
        batched_step = jax.vmap(
            lambda s, t: self.step(weights, s, t),
            in_axes=(0, 0),
        )

        # Consume tokens 0..length-2; their outputs are logits 1..length-1.
        inputs_t = jnp.transpose(batch[:, :-1], (1, 0))  # (length-1, batch)

        def scan_fn(state, token_t):
            return batched_step(state, token_t)

        _, logits_t = jax.lax.scan(scan_fn, batched_state, inputs_t)
        # logits_t: (length-1, batch, vocab)

        init_logits_batched = jnp.broadcast_to(
            init_logits, (batch_size,) + init_logits.shape)
        all_logits = jnp.concatenate(
            [init_logits_batched[None], logits_t], axis=0
        )  # (length, batch, vocab)
        return jnp.transpose(all_logits, (1, 0, 2))  # (batch, length, vocab)

    def loss_batch_masked_recurrent(
        self, weights: Weights, batch: jax.Array, lengths: jax.Array
    ) -> jax.Array:
        """Same as loss_batch_masked but via forward_recurrent.

        Used for eval where the parallel form's O(L^2) scores tensor
        OOMs at full eval shapes (e.g. 1024x1024) on the 16 GB systems.
        """
        logits = self.forward_recurrent(weights, batch)
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        mask = jnp.arange(per_token.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(per_token * mask)
