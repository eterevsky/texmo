import math

import jax
import jax.numpy as jnp
import optax

from .layer_jax import LayerJax, LayerState, LayerWeights
from .layers.dense import DenseJax
from .layers.input_bits import InputBitsJax

_1_BY_LOG2 = 1.0 / math.log(2.0)

Weights = list[LayerWeights]
State = list[LayerState]


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
    ):
        self.input = input_layer
        self.layers = layers
        self.output = output
        self.ntokens = ntokens
        self._pad_output = ntokens <= 2
        self._total_padding = total_padding

    def init_weights(self, rng: jax.Array) -> Weights:
        keys = jax.random.split(rng, len(self.layers) + 1)
        return (
            [None]  # input layer has no learned weights
            + [layer.init_weights(keys[i]) for i, layer in enumerate(self.layers)]
            + [self.output.init_weights(keys[-1])]
        )

    def initial_step(self, weights: Weights) -> tuple[State, jax.Array]:
        """Predict the first token (before any input).

        Feeds total_padding initial vectors through the layers to warm
        up stateful layers (e.g. suffix), mirroring the padding in
        forward().

        Returns:
            (states, logits) where states[0] is input state, states[1:]
            are layer states, and logits is (ntokens,).
        """
        input_state = self.input.init_state()
        layer_states = [layer.init_state() for layer in self.layers]

        p = self._total_padding
        for i in range(p):
            v = self.input._initial_vector(position=-p + i)
            for j, layer in enumerate(self.layers):
                layer_states[j], v = layer.step(
                    weights[j + 1], layer_states[j], v)

        states = [input_state] + layer_states
        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return states, logits

    def step(
        self, weights: Weights, states: State, token: int
    ) -> tuple[State, jax.Array]:
        """Run one step of inference.

        Returns:
            (new_states, logits) where logits is (ntokens,).
        """
        new_states = []
        input_state, v = self.input.step(states[0], token)
        new_states.append(input_state)

        for layer, lw, ls in zip(self.layers, weights[1:-1], states[1:]):
            ls, v = layer.step(lw, ls, v)
            new_states.append(ls)

        _, logits = self.output.step(weights[-1], None, v)
        if self._pad_output:
            logits = jnp.pad(logits, (0, 1))
        return new_states, logits

    def forward(self, weights: Weights, batch: jax.Array) -> jax.Array:
        """Forward pass on a batch for training.

        Args:
            weights: from init_weights.
            batch: (batch_size, seq_len) int token indices.

        Returns:
            logits: (batch_size, seq_len, ntokens).
        """
        v = self.input.forward(batch[:, :-1], padding=self._total_padding)

        for layer, lw in zip(self.layers, weights[1:-1]):
            v = layer.forward(lw, v)

        logits = self.output.forward(weights[-1], v)
        if self._pad_output:
            logits = jnp.pad(logits, ((0, 0), (0, 0), (0, 1)))
        return logits

    def loss_batch(self, weights: Weights, batch: jax.Array) -> jax.Array:
        """Average cross-entropy loss in bits per token.

        Returns:
            scalar.
        """
        logits = self.forward(weights, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        return _1_BY_LOG2 * jnp.mean(loss)

    def loss_batch_masked(
        self, weights: Weights, batch: jax.Array, lengths: jax.Array
    ) -> jax.Array:
        """Total cross-entropy in bits, masked to actual sample lengths.

        Used for eval where samples have fixed byte length but variable
        token length. Only the first `lengths[i]` tokens of each sample
        contribute to the loss.

        Returns:
            scalar total loss (not averaged).
        """
        logits = self.forward(weights, batch)
        per_token = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch)
        mask = jnp.arange(per_token.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(per_token * mask)
