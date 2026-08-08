from typing import Any

import jax
import jax.numpy as jnp

LayerWeights = Any  # pytree of jax.Array, or None
LayerState = Any    # pytree of jax.Array, or None


def xavier_uniform(rng: jax.Array, shape: tuple[int, int], dtype=jnp.float32) -> jax.Array:
    """Xavier uniform init for a (fan_out, fan_in) weight matrix."""
    fan_out, fan_in = shape
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    return jax.random.uniform(rng, shape, dtype=dtype, minval=-bound, maxval=bound)


class LayerJax:
    """Lightweight base for JAX layer implementations.

    Weights are not owned by the layer — they're passed as explicit
    arguments. This follows JAX's functional paradigm: weights are
    inputs to jax.grad / jax.jit / lax.scan.

    All batch-dimension methods expect batch as the first axis.
    """

    def __init__(self, input_size: int, size: int, dtype=jnp.float32):
        self.input_size = input_size
        self.size = size
        self.dtype = dtype

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        """Initialize layer weights.

        Returns:
            A pytree of weight arrays, or None for weightless layers.
        """
        return None

    def init_state(self) -> LayerState:
        """Initialize recurrent state for a single sample.

        Returns None for stateless layers.
        """
        return None

    def step(
        self, weights: LayerWeights, state: LayerState, x: jax.Array
    ) -> tuple[LayerState, jax.Array]:
        """Single-timestep forward on a single sample (no batch dim).

        Args:
            weights: from init_weights.
            state: recurrent state, or None for stateless layers.
            x: (input_size,) input at one timestep.

        Returns:
            (new_state, output) where output is (size,).
        """
        raise NotImplementedError

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        """Full-sequence forward on a batch.

        Args:
            weights: from init_weights.
            inputs: (batch, seq_len, input_size).

        Returns:
            (batch, seq_len, size).
        """
        raise NotImplementedError
