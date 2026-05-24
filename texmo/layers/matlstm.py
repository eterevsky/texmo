import math

import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


class MatLstmJax(LayerJax):
    """Matrix-state LSTM (xLSTM family, Beck et al. 2024).

    Per-cell dim D = `size`. State per sample:
        C: (D, D)   — matrix cell
        n: (D,)     — normaliser vector
        m: scalar   — running log-magnitude tracker (numerical
                      stabiliser for the exponential input gate)

    Per step (single head; scalar i/f/o gates):
        q  = W_q x
        k  = (W_k x) / sqrt(D)
        v  = W_v x
        i_pre, f_pre, o_pre = W_ifo x + b_ifo
        log_f = log_sigmoid(f_pre)
        m_new = max(log_f + m, i_pre)
        i_st  = exp(i_pre - m_new)
        f_st  = exp(log_f + m - m_new)            # = sigmoid(f_pre)·exp(m-m_new)
        o     = sigmoid(o_pre)
        C_new = f_st · C + i_st · outer(v, k)
        n_new = f_st · n + i_st · k
        h     = o · (C_new @ q) / max(|n_new · q|, 1)

    Weights (no bias on Q/K/V, transformer convention):
        w_qkv:  (3·D, input_size)
        w_ifo:  (3, input_size)
        b_ifo:  (3,)
    """

    def __init__(self, input_size: int, size: int, dtype=jnp.float32):
        super().__init__(input_size, size, dtype)
        self._inv_sqrt_d = 1.0 / math.sqrt(size)

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_qkv, k_ifo = jax.random.split(rng)
        return {
            'w_qkv': xavier_uniform(
                k_qkv, (3 * self.size, self.input_size), dtype=self.dtype),
            'w_ifo': xavier_uniform(
                k_ifo, (3, self.input_size), dtype=self.dtype),
            'b_ifo': jnp.zeros(3, dtype=self.dtype),
        }

    def init_state(self):
        d = self.size
        return (
            jnp.zeros((d, d), dtype=self.dtype),
            jnp.zeros(d, dtype=self.dtype),
            jnp.zeros((), dtype=self.dtype),
        )

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
        C, n, m = state
        d = self.size

        qkv = weights['w_qkv'] @ x
        q = qkv[:d]
        k = qkv[d:2 * d] * self._inv_sqrt_d
        v = qkv[2 * d:]

        gates = weights['w_ifo'] @ x + weights['b_ifo']
        i_pre, f_pre, o_pre = gates[0], gates[1], gates[2]

        log_f = jax.nn.log_sigmoid(f_pre)
        m_new = jnp.maximum(log_f + m, i_pre)
        i_st = jnp.exp(i_pre - m_new)
        f_st = jnp.exp(log_f + m - m_new)
        o = jax.nn.sigmoid(o_pre)

        C_new = f_st * C + i_st * jnp.outer(v, k)
        n_new = f_st * n + i_st * k

        h_pre = C_new @ q
        denom = jnp.maximum(jnp.abs(jnp.dot(n_new, q)), 1.0)
        h = o * h_pre / denom
        return (C_new, n_new, m_new), h

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        batch_size, length, _ = inputs.shape
        d = self.size

        # Hoist everything that's input-only out of the scan body:
        # input projections (one big matmul instead of T small ones),
        # the log_sigmoid / sigmoid gate activations, and the per-token
        # outer product k·v^T. The scan body keeps only the truly
        # recurrent computation -- the m_t stabiliser arithmetic and
        # the C/n updates. (B, T, D, D) for kv is small at our scales
        # -- 16 MB at B=64, T=1024, D=8.
        qkv = inputs @ weights['w_qkv'].T  # (B, T, 3D)
        gates = inputs @ weights['w_ifo'].T + weights['b_ifo']  # (B, T, 3)

        q_all = qkv[..., :d]
        k_all = qkv[..., d:2 * d] * self._inv_sqrt_d
        v_all = qkv[..., 2 * d:]
        i_pre_all = gates[..., 0]
        log_f_all = jax.nn.log_sigmoid(gates[..., 1])
        o_all = jax.nn.sigmoid(gates[..., 2])
        kv_all = v_all[..., :, None] * k_all[..., None, :]  # (B, T, D, D)

        # Scan over the time axis -> (T, B, ...).
        q_t = jnp.transpose(q_all, (1, 0, 2))
        k_t = jnp.transpose(k_all, (1, 0, 2))
        kv_t = jnp.transpose(kv_all, (1, 0, 2, 3))
        i_pre_t = jnp.transpose(i_pre_all, (1, 0))
        log_f_t = jnp.transpose(log_f_all, (1, 0))
        o_t = jnp.transpose(o_all, (1, 0))

        def scan_step(state, inp):
            C, n, m = state  # (B, D, D), (B, D), (B,)
            q, k, kv, i_pre, log_f, o = inp

            m_new = jnp.maximum(log_f + m, i_pre)
            i_st = jnp.exp(i_pre - m_new)
            f_st = jnp.exp(log_f + m - m_new)

            C_new = f_st[:, None, None] * C + i_st[:, None, None] * kv
            n_new = f_st[:, None] * n + i_st[:, None] * k

            h_pre = jnp.einsum('bij,bj->bi', C_new, q)
            denom = jnp.maximum(
                jnp.abs(jnp.sum(n_new * q, axis=-1)), 1.0)
            h = o[:, None] * h_pre / denom[:, None]
            return (C_new, n_new, m_new), h

        init_C = jnp.zeros((batch_size, d, d), dtype=self.dtype)
        init_n = jnp.zeros((batch_size, d), dtype=self.dtype)
        init_m = jnp.zeros((batch_size,), dtype=self.dtype)
        _, h_t = jax.lax.scan(
            scan_step, (init_C, init_n, init_m),
            (q_t, k_t, kv_t, i_pre_t, log_f_t, o_t),
        )
        return jnp.transpose(h_t, (1, 0, 2))


class MatLstmDef(LayerDef):
    name = "matlstm"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"matlstm.{self.size}"

    def is_valid(self) -> bool:
        # size >= 2 since the matrix state needs more than one element
        # to be meaningfully a "matrix" vs a scalar (D=1 -> C is 1x1).
        return is_power2_int(self.size) and self.size >= 2

    @property
    def num_weights(self) -> int:
        # 3*D*I (Q/K/V, no bias) + 3*I (i/f/o input proj) + 3 (i/f/o bias).
        return 3 * self.size * self.input_size + 3 * self.input_size + 3

    @property
    def num_mults(self) -> int:
        # Per-token: projections (= num_weights) + per-step matrix-state
        # ops (outer product k v^T plus C@q -> 2·D²) + the small n
        # updates. Rotation-free, no L^2 term.
        return self.num_weights + 2 * self.size * self.size

    def build_jax(self, dtype) -> MatLstmJax:
        return MatLstmJax(self.input_size, self.size, dtype)
