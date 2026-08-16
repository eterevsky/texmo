import jax
import jax.nn as jax_nn
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform

_JAX_ACTIVATIONS = {
    "relu": jax_nn.relu,
    "tanh": jnp.tanh,
    "gelu": jax_nn.gelu,
    "silu": jax_nn.silu,
}


class RnnJax(LayerJax):
    """Elman RNN with separate input and hidden projections.

    Splits the weight matrix so forward() can hoist the input
    projection out of the scan loop (one batched matmul over the full
    sequence instead of one per timestep). Each matrix is Xavier-init
    based on its own fan_in/fan_out. Uses a single bias.

    Weights:
        w_ih: (size, input_size)
        w_hh: (size, size)
        b: (size,)

    Uses a single bias vector, not the two-bias `bias_ih + bias_hh`
    convention some RNN libraries use; see docs/layers.md,
    "Parameter-count convention".
    """

    def __init__(self, input_size: int, size: int, activation_name: str, dtype):
        super().__init__(input_size, size, dtype)
        self.activation = _JAX_ACTIVATIONS[activation_name]

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ih, k_hh = jax.random.split(rng)
        return {
            'w_ih': xavier_uniform(
                k_ih, (self.size, self.input_size), dtype=self.dtype),
            'w_hh': xavier_uniform(
                k_hh, (self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        new_state = self.activation(
            weights['w_ih'] @ x + weights['w_hh'] @ state + weights['b'])
        return new_state, new_state

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        w_hh = weights['w_hh']
        # Hoist input projection out of the scan: one batched matmul
        # over the whole sequence.
        Wx = inputs @ weights['w_ih'].T + weights['b']
        # (batch, seq_len, size) → (seq_len, batch, size) for scan
        Wx_t = jnp.transpose(Wx, (1, 0, 2))

        def scan_step(state, wx_batch):
            # state: (batch, size), wx_batch: (batch, size)
            new_state = self.activation(wx_batch + state @ w_hh.T)
            return new_state, new_state

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, init, Wx_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class RnnDef(LayerDef):
    name = "rnn"

    def __init__(self, size: int, input_size: int, activation: str | None = None):
        super().__init__(input_size=input_size)
        self.size = size
        self._activation = activation

    def __str__(self) -> str:
        if self._activation:
            return f"rnn.{self.size}.{self._activation}"
        return f"rnn.{self.size}"

    def is_valid(self) -> bool:
        # relu is retired (see DenseDef.is_valid): parses and runs,
        # but no longer search-valid. silu joined 2026-08.
        return (is_power2_int(self.size)
                and self._activation in ("tanh", "gelu", "silu"))

    @property
    def num_weights(self) -> int:
        # Single-bias count (matches RnnJax).
        s = self.size
        return s * self.input_size + s * s + s

    def build_jax(self, dtype) -> RnnJax:
        return RnnJax(self.input_size, self.size, self._activation, dtype)
