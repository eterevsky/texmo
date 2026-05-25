import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform
from .latent import _normalize_jax


class LmguJax(LayerJax):
    """Latent-recurrent mGRU.

    Combines mGRU's gating with lrnn's depth-recurrent iteration of
    the gate and candidate. Per token, the gate `f` and candidate
    `hc` are jointly refined for `reps` iterations, then the final
    pair updates the time-recurrent hidden state via the standard
    mGRU blend.

    Per step:
        hc[0] = 0
        f[0]  = 0
        for i in range(reps):
            hc[i+1] = norm(tanh(
                W_h_i x + W_h_h (f[i] * h) + W_h_s hc[i] + b_h))
            f[i+1]  = sigmoid(
                W_f_i x + W_f_h h + W_f_s hc[i+1] + b_f)
        h_new = (1 - f[reps]) * h + f[reps] * hc[reps]

    Only `hc` is L2-normalized -- this is the architecturally
    significant constraint (direction-only candidate); raw `x` and
    `h` are fed to the W matmuls without normalization, matching
    standard mGRU.

    The unit-norm constraint on hc gives a clean factorisation:
    hc represents direction, f represents magnitude. Both f and
    hc are computed against unit-norm hc, so the final blend is
    consistent (f gates hc at the magnitude it was trained for).

    State per sample: h (size,) -- same as mgru.

    Weights (mirroring lrnn's input / h / state split for clarity):
        w_h_i: (size, input_size)  -- input projection for hc
        w_h_h: (size, size)        -- gated h projection for hc
        w_h_s: (size, size)        -- recurrent hc projection
        b_h:   (size,)
        w_f_i: (size, input_size)  -- input projection for f
        w_f_h: (size, size)        -- h projection for f
        w_f_s: (size, size)        -- recurrent hc projection for f
        b_f:   (size,)
    """

    def __init__(self, input_size: int, size: int, reps: int, dtype):
        super().__init__(input_size, size, dtype)
        self.reps = reps

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_hi, k_hh, k_hs, k_fi, k_fh, k_fs = jax.random.split(rng, 6)
        s = self.size
        i = self.input_size
        d = self.dtype
        return {
            'w_h_i': xavier_uniform(k_hi, (s, i), dtype=d),
            'w_h_h': xavier_uniform(k_hh, (s, s), dtype=d),
            'w_h_s': xavier_uniform(k_hs, (s, s), dtype=d),
            'b_h':   jnp.zeros(s, dtype=d),
            'w_f_i': xavier_uniform(k_fi, (s, i), dtype=d),
            'w_f_h': xavier_uniform(k_fh, (s, s), dtype=d),
            'w_f_s': xavier_uniform(k_fs, (s, s), dtype=d),
            'b_f':   jnp.zeros(s, dtype=d),
        }

    def init_state(self):
        return jnp.zeros(self.size, dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (input_size,), state: (size,)
        h = state

        # Per-token constants (computed once for all reps).
        eh_input = weights['w_h_i'] @ x + weights['b_h']
        ef_input = (
            weights['w_f_i'] @ x
            + weights['w_f_h'] @ h
            + weights['b_f']
        )

        w_h_h = weights['w_h_h']
        w_h_s = weights['w_h_s']
        w_f_s = weights['w_f_s']

        def body(carry, _):
            hc, f = carry
            e_h = eh_input + w_h_h @ (f * h) + w_h_s @ hc
            hc_new = _normalize_jax(jnp.tanh(e_h))
            e_f = ef_input + w_f_s @ hc_new
            f_new = jax.nn.sigmoid(e_f)
            return (hc_new, f_new), None

        init = (
            jnp.zeros(self.size, dtype=self.dtype),  # hc[0]
            jnp.zeros(self.size, dtype=self.dtype),  # f[0]
        )
        (hc, f), _ = jax.lax.scan(body, init, None, length=self.reps)

        h_new = (1 - f) * h + f * hc
        return h_new, h_new

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # Hoist input-only projections (with bias baked in) outside
        # the time scan -- they're constant per token.
        eh_xb_all = inputs @ weights['w_h_i'].T + weights['b_h']
        ef_xb_all = inputs @ weights['w_f_i'].T + weights['b_f']

        eh_xb_t = jnp.transpose(eh_xb_all, (1, 0, 2))  # (T, B, size)
        ef_xb_t = jnp.transpose(ef_xb_all, (1, 0, 2))

        w_h_h = weights['w_h_h']
        w_h_s = weights['w_h_s']
        w_f_h = weights['w_f_h']
        w_f_s = weights['w_f_s']
        reps = self.reps

        def time_step(h, inp):
            eh_xb, ef_xb = inp
            # h-dependent part for f is constant across reps.
            ef_input = ef_xb + h @ w_f_h.T

            def reps_body(carry, _):
                hc, f = carry
                e_h = (
                    eh_xb
                    + (f * h) @ w_h_h.T
                    + hc @ w_h_s.T
                )
                hc_new = _normalize_jax(jnp.tanh(e_h))
                e_f = ef_input + hc_new @ w_f_s.T
                f_new = jax.nn.sigmoid(e_f)
                return (hc_new, f_new), None

            batch_zeros = jnp.zeros_like(h)
            init = (batch_zeros, batch_zeros)
            (hc, f), _ = jax.lax.scan(
                reps_body, init, None, length=reps)
            h_new = (1 - f) * h + f * hc
            return h_new, h_new

        batch = inputs.shape[0]
        init_h = jnp.zeros((batch, self.size), dtype=self.dtype)
        _, outputs_t = jax.lax.scan(
            time_step, init_h, (eh_xb_t, ef_xb_t))
        return jnp.transpose(outputs_t, (1, 0, 2))


class LmguDef(LayerDef):
    name = "lmgu"

    def __init__(self, size: int, reps: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size
        self.reps = reps

    def __str__(self) -> str:
        return f"lmgu.{self.size}.{self.reps}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.size) and self.size > 1
            and is_power2_int(self.reps) and self.reps >= 2
        )

    @property
    def num_weights(self) -> int:
        # 6 weight matrices + 2 biases:
        #   w_h_i, w_f_i: (size, input_size)
        #   w_h_h, w_h_s, w_f_h, w_f_s: (size, size)
        #   b_h, b_f: (size,)
        s = self.size
        return 2 * s * self.input_size + 4 * s * s + 2 * s

    @property
    def num_mults(self) -> int:
        # The w_*_h, w_*_s matmuls fire `reps` times per token; the
        # input projections fire once. Match lrnn's coarse
        # approximation: scale the whole weight count by reps.
        return self.num_weights * self.reps

    def build_jax(self, dtype) -> LmguJax:
        return LmguJax(self.input_size, self.size, self.reps, dtype)
