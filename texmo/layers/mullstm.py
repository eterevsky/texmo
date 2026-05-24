import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


class MulLstmJax(LayerJax):
    """Multiplicative LSTM (Krause, Murray, Renals & Lu 2016 --
    "Multiplicative LSTM for Sequence Modelling").

    Renamed here to `mullstm` to disambiguate from the matrix-state
    `matlstm` (Beck-2024 xLSTM), both of which the literature has
    referred to as "mLSTM".

    The key idea: the previous hidden state h_{t-1} is element-wise
    modulated by an input-dependent projection before reaching the
    standard LSTM gates. This gives the recurrent transition an
    input-dependent flavour without changing the cell-state machinery.

    Per step (gates ordered [f, i, o, z] to match LstmJax so the
    three sigmoids occupy a contiguous range):
        m       = (W_mx x) ⊙ (W_mh h_{t-1})            # (size,)
        gates   = W_ih x + W_hh m + b                    # (4·size,)
        f,i,o   = sigmoid(gates[:3s]) split into thirds
        z       = tanh(gates[3s:])
        c_new   = f * c + i * z
        h_new   = o * tanh(c_new)

    State per sample: (h, c) -- identical to vanilla LSTM.

    Weights:
        w_mx: (size, input_size) -- multiplicative input projection
        w_mh: (size, size)       -- multiplicative hidden projection
        w_ih: (4·size, input_size)
        w_hh: (4·size, size)
        b:    (4·size,)
    """

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_mx, k_mh, k_ih, k_hh = jax.random.split(rng, 4)
        s = self.size
        return {
            'w_mx': xavier_uniform(
                k_mx, (s, self.input_size), dtype=self.dtype),
            'w_mh': xavier_uniform(
                k_mh, (s, s), dtype=self.dtype),
            'w_ih': xavier_uniform(
                k_ih, (4 * s, self.input_size), dtype=self.dtype),
            'w_hh': xavier_uniform(
                k_hh, (4 * s, s), dtype=self.dtype),
            'b': jnp.zeros(4 * s, dtype=self.dtype),
        }

    def init_state(self):
        s = self.size
        return (
            jnp.zeros(s, dtype=self.dtype),  # h
            jnp.zeros(s, dtype=self.dtype),  # c
        )

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        h, c = state
        s = self.size

        mx = weights['w_mx'] @ x
        mh = weights['w_mh'] @ h
        m = mx * mh

        gates = weights['w_ih'] @ x + weights['w_hh'] @ m + weights['b']
        f = jax.nn.sigmoid(gates[:s])
        i = jax.nn.sigmoid(gates[s:2 * s])
        o = jax.nn.sigmoid(gates[2 * s:3 * s])
        z = jnp.tanh(gates[3 * s:])

        c_new = f * c + i * z
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        s = self.size

        # Hoist both input-side projections (W_ih x and W_mx x) -- both
        # are input-only. W_hh m and W_mh h depend on the carry, stay
        # in the scan body.
        Wx_ih = inputs @ weights['w_ih'].T + weights['b']  # (B, T, 4s)
        Wx_mx = inputs @ weights['w_mx'].T                 # (B, T, s)

        Wx_ih_t = jnp.transpose(Wx_ih, (1, 0, 2))  # (T, B, 4s)
        Wx_mx_t = jnp.transpose(Wx_mx, (1, 0, 2))  # (T, B, s)

        w_hh = weights['w_hh']
        w_mh = weights['w_mh']

        def scan_step(state, inp):
            h, c = state
            wx_ih, wx_mx = inp
            m = wx_mx * (h @ w_mh.T)
            gates = wx_ih + m @ w_hh.T  # (B, 4s)
            # Single sigmoid call for the f/i/o block.
            fio = jax.nn.sigmoid(gates[:, :3 * s])
            z = jnp.tanh(gates[:, 3 * s:])
            f = fio[:, :s]
            i = fio[:, s:2 * s]
            o = fio[:, 2 * s:]
            c_new = f * c + i * z
            h_new = o * jnp.tanh(c_new)
            return (h_new, c_new), h_new

        batch = inputs.shape[0]
        init_h = jnp.zeros((batch, s), dtype=self.dtype)
        init_c = jnp.zeros((batch, s), dtype=self.dtype)
        _, h_t = jax.lax.scan(
            scan_step, (init_h, init_c), (Wx_ih_t, Wx_mx_t))
        return jnp.transpose(h_t, (1, 0, 2))


class MulLstmDef(LayerDef):
    name = "mullstm"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"mullstm.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # LSTM weights + multiplicative projections (no biases):
        #   5·size·(input_size + size) + 4·size
        return (
            5 * self.size * (self.input_size + self.size)
            + 4 * self.size
        )

    def build_jax(self, dtype) -> MulLstmJax:
        return MulLstmJax(self.input_size, self.size, dtype)
