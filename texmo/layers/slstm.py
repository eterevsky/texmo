import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


class SLstmJax(LayerJax):
    """Scalar-state sLSTM from xLSTM (Beck et al. 2024).

    Vector cells with full memory mixing (single head), exponential
    input gate with max-tracking stabilisation, sigmoid forget/output
    gates, and an attention-like normaliser carry.

    Per step (matching paper equations 8-17):
        gates    = W_ih x + W_hh h + b                    # (4·size,)
        z        = tanh(gates[:s])                         # cell input
        o        = sigmoid(gates[3s:4s])                   # output gate
        log_f    = log_sigmoid(gates[2s:3s])               # log(sigmoid f)
        m_new    = max(log_f + m, gates[s:2s])             # stabiliser
        i_st     = exp(gates[s:2s] - m_new)                # <= 1
        f_st     = exp(log_f + m - m_new)                  # <= 1
        c_new    = f_st * c + i_st * z
        n_new    = f_st * n + i_st                          # normaliser
        h_new    = o * (c_new / n_new)                     # paper eq (10)

    Same parameter count as LSTM (4·size·(input_size + size) + 4·size).
    The n and m carries add no weights. State per sample is four
    (size,) vectors: (h, c, n, m).

    Single head (R is one full size×size matrix mixing all cells).
    Multi-head would split cells into blocks and block-diagonalise R;
    deferred until single-head data tells us whether it's worth it.
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
        s = self.size
        return (
            jnp.zeros(s, dtype=self.dtype),  # h
            jnp.zeros(s, dtype=self.dtype),  # c
            jnp.zeros(s, dtype=self.dtype),  # n
            jnp.zeros(s, dtype=self.dtype),  # m
        )

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array,
    ]:
        h, c, n, m = state
        s = self.size

        gates = weights['w_ih'] @ x + weights['w_hh'] @ h + weights['b']
        z = jnp.tanh(gates[:s])
        i_pre = gates[s:2 * s]
        f_pre = gates[2 * s:3 * s]
        o = jax.nn.sigmoid(gates[3 * s:])

        log_f = jax.nn.log_sigmoid(f_pre)
        m_new = jnp.maximum(log_f + m, i_pre)
        i_st = jnp.exp(i_pre - m_new)
        f_st = jnp.exp(log_f + m - m_new)

        c_new = f_st * c + i_st * z
        n_new = f_st * n + i_st
        h_new = o * c_new / n_new
        return (h_new, c_new, n_new, m_new), h_new

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        s = self.size

        # Hoist the input projection -- one big matmul over the
        # sequence instead of T small ones. Everything else (gate
        # activations, the m-stabilised carry updates, the output
        # division) stays in the scan body because it depends on the
        # h carry through w_hh @ h.
        Wx = inputs @ weights['w_ih'].T + weights['b']  # (B, T, 4·size)
        Wx_t = jnp.transpose(Wx, (1, 0, 2))  # (T, B, 4·size)

        w_hh = weights['w_hh']

        def scan_step(state, wx):
            h, c, n, m = state
            gates = wx + h @ w_hh.T  # (B, 4·size)
            z = jnp.tanh(gates[:, :s])
            i_pre = gates[:, s:2 * s]
            f_pre = gates[:, 2 * s:3 * s]
            o = jax.nn.sigmoid(gates[:, 3 * s:])

            log_f = jax.nn.log_sigmoid(f_pre)
            m_new = jnp.maximum(log_f + m, i_pre)
            i_st = jnp.exp(i_pre - m_new)
            f_st = jnp.exp(log_f + m - m_new)

            c_new = f_st * c + i_st * z
            n_new = f_st * n + i_st
            h_new = o * c_new / n_new
            return (h_new, c_new, n_new, m_new), h_new

        batch = inputs.shape[0]
        init_h = jnp.zeros((batch, s), dtype=self.dtype)
        init_c = jnp.zeros((batch, s), dtype=self.dtype)
        init_n = jnp.zeros((batch, s), dtype=self.dtype)
        init_m = jnp.zeros((batch, s), dtype=self.dtype)
        _, h_t = jax.lax.scan(
            scan_step, (init_h, init_c, init_n, init_m), Wx_t)
        return jnp.transpose(h_t, (1, 0, 2))


class SLstmDef(LayerDef):
    name = "slstm"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"slstm.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # Identical to LstmDef -- n and m carries add no parameters.
        return 4 * self.size * (self.input_size + self.size) + 4 * self.size

    def build_jax(self, dtype) -> SLstmJax:
        return SLstmJax(self.input_size, self.size, dtype)
