import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


class LstmJax(LayerJax):
    """LSTM with stacked gate projections and hoisted input projection.

    Uses the standard LSTM equations:
        f = sigmoid(W_if x + W_hf h + b_f)   # forget
        i = sigmoid(W_ii x + W_hi h + b_i)   # input
        o = sigmoid(W_io x + W_ho h + b_o)   # output
        g = tanh(W_ig x + W_hg h + b_g)      # candidate
        c_new = f * c + i * g
        h_new = o * tanh(c_new)

    All four gates are stacked into a single pair of weight matrices
    (shape (4*size, input_size) and (4*size, size)) and a single bias
    (shape (4*size,)). The input projection is computed once per
    sequence; only the hidden projection runs inside the scan.

    State is (h, c) — both needed to continue inference.

    Uses one bias per gate (not the two-bias convention some LSTM
    libraries use); see docs/layers.md, "Parameter-count convention".
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ih, k_hh = jax.random.split(rng)
        return {
            'w_ih': xavier_uniform(
                k_ih, (4 * self.size, self.input_size), dtype=self.dtype),
            'w_hh': xavier_uniform(
                k_hh, (4 * self.size, self.size), dtype=self.dtype),
            'b': jnp.zeros(4 * self.size, dtype=self.dtype),
        }

    def init_state(self):
        h = jnp.zeros(self.size, dtype=self.dtype)
        c = jnp.zeros(self.size, dtype=self.dtype)
        return (h, c)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        # x: (input_size,), state: ((size,), (size,))
        h, c = state
        gates = weights['w_ih'] @ x + weights['w_hh'] @ h + weights['b']
        s = self.size
        f = jax.nn.sigmoid(gates[:s])
        i = jax.nn.sigmoid(gates[s:2 * s])
        o = jax.nn.sigmoid(gates[2 * s:3 * s])
        g = jnp.tanh(gates[3 * s:])
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # Hoist input projections for all 4 gates as one batched matmul.
        Wx = inputs @ weights['w_ih'].T + weights['b']  # (batch, seq_len, 4*size)
        Wx_t = jnp.transpose(Wx, (1, 0, 2))  # (seq_len, batch, 4*size)

        w_hh = weights['w_hh']
        s = self.size

        def scan_step(state, wx):
            h, c = state
            # One matmul for all 4 hidden projections.
            gates = wx + h @ w_hh.T  # (batch, 4*size)
            fio = jax.nn.sigmoid(gates[:, :3 * s])
            g = jnp.tanh(gates[:, 3 * s:])
            f = fio[:, :s]
            i = fio[:, s:2 * s]
            o = fio[:, 2 * s:]
            c_new = f * c + i * g
            h_new = o * jnp.tanh(c_new)
            return (h_new, c_new), h_new

        batch = inputs.shape[0]
        init_h = jnp.zeros((batch, self.size), dtype=self.dtype)
        init_c = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(scan_step, (init_h, init_c), Wx_t)
        return jnp.transpose(outputs_t, (1, 0, 2))


class LstmDef(LayerDef):
    name = "lstm"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"lstm.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # Single-bias per gate (matches LstmJax).
        return 4 * self.size * (self.input_size + self.size) + 4 * self.size

    def build_jax(self, dtype) -> LstmJax:
        return LstmJax(self.input_size, self.size, dtype)
