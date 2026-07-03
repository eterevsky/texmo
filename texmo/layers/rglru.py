"""RG-LRU: the Real-Gated Linear Recurrent Unit from Griffin / Hawk
(De et al. 2024, "Griffin"), as shipped in RecurrentGemma.

A diagonal *real* linear recurrence with input-dependent gating:

    r_t = sigmoid(W_a x_t + b_a)          # recurrence gate
    i_t = sigmoid(W_x x_t + b_x)          # input gate
    a_t = exp(-c * r_t * softplus(Lambda)), c = 8   (per-channel)
    h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)

The recurrence is element-wise (diagonal `a`), and neither gate depends
on h_{t-1}, so the whole thing is a linear scan. `Lambda` is a learned
per-channel parameter; the `sqrt(1 - a^2)` factor keeps the state
variance stable as `a` varies.

Dimension-preserving (output size == input size, like `conv`). The two
gate projections are **block-diagonal** with `blocks` blocks (matching
RecurrentGemma, where `blocks == num_attention_heads` and the block
width is `lru_width / blocks`) -- a parameter-efficiency choice we
replicate so pretrained weights load directly.

JAX only (no PyTorch module), consistent with the other recent cells.
Matches `transformers` `RecurrentGemmaRglru` numerically: gates use the
stored `Lambda` directly inside `softplus`, the recurrence accumulates
in float32, and the first position of a sequence is a reset (input
multiplier 1, no carry).
"""
import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights

_C = 8.0  # recurrence-gate temperature constant (Griffin eq. 3)


class RglruJax(LayerJax):
    """Real-Gated Linear Recurrent Unit (block-diagonal gates)."""

    def __init__(self, input_size: int, blocks: int, dtype):
        # Dimension-preserving: the recurrence is diagonal.
        super().__init__(input_size, input_size, dtype)
        assert input_size % blocks == 0
        self.blocks = blocks
        self.block_width = input_size // blocks

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_ig, k_rg = jax.random.split(rng, 2)
        bw = self.block_width
        shape = (self.blocks, bw, bw)
        # Xavier-uniform per block (each block is a bw x bw linear;
        # fan_in = fan_out = bw -> limit = sqrt(3/bw)).
        limit = (3.0 / bw) ** 0.5

        def gate(key):
            return jax.random.uniform(
                key, shape, dtype=self.dtype, minval=-limit, maxval=limit)

        # Initialise Lambda so that, at gate midpoint (r ~ 0.5), the
        # per-channel decay a^c lands around 0.9-0.99 (Griffin init):
        # a = exp(-c * 0.5 * softplus(Lambda)). softplus(-4) ~ 0.018 ->
        # a ~ 0.93, a stable starting memory horizon.
        return {
            'lam': jnp.full((self.size,), -4.0, dtype=self.dtype),
            'w_ig': gate(k_ig),
            'b_ig': jnp.zeros((self.blocks, bw), dtype=self.dtype),
            'w_rg': gate(k_rg),
            'b_rg': jnp.zeros((self.blocks, bw), dtype=self.dtype),
        }

    def init_state(self):
        # State accumulates in float32 (matches the reference).
        return jnp.zeros(self.size, dtype=jnp.float32)

    def _gates(self, weights: LayerWeights, x):
        """Block-diagonal input + recurrence gates for x of shape
        (..., size). Returns (input_gate, recurrence_gate), both
        (..., size), pre-sigmoid folded in."""
        lead = x.shape[:-1]
        xb = x.reshape(*lead, self.blocks, self.block_width)
        # Per-block: g[k] = x[k] @ W[k] + b[k].
        ig = jnp.einsum('...kc,kcd->...kd', xb, weights['w_ig']) + weights['b_ig']
        rg = jnp.einsum('...kc,kcd->...kd', xb, weights['w_rg']) + weights['b_rg']
        i = jax.nn.sigmoid(ig).reshape(*lead, self.size)
        r = jax.nn.sigmoid(rg).reshape(*lead, self.size)
        return i, r

    def _decay(self, weights: LayerWeights, r):
        """a_t and sqrt(1 - a_t^2) from the recurrence gate r."""
        log_a = (-_C) * r * jax.nn.softplus(weights['lam'])
        a = jnp.exp(log_a)
        a_sq = jnp.exp(2.0 * log_a)
        mult = jnp.sqrt(jnp.clip(1.0 - a_sq, 1e-12, 1.0))
        return a, mult

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # x: (size,), state: (size,) float32.
        i, r = self._gates(weights, x)
        a, mult = self._decay(weights, r)
        normalized_x = (x * i) * mult
        new_state = (a * state + normalized_x).astype(jnp.float32)
        return new_state, new_state.astype(self.dtype)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, size).
        i, r = self._gates(weights, inputs)
        a, mult = self._decay(weights, r)
        gated = inputs * i

        # Reset at the first position: input multiplier 1, no carry
        # (h_{-1} = 0 anyway, so the a-reset is a no-op, but the
        # multiplier-reset is what matches the reference at t=0).
        seq_len = inputs.shape[1]
        reset = (jnp.arange(seq_len) == 0)[None, :, None]  # (1, T, 1)
        mult = jnp.where(reset, 1.0, mult)
        a = jnp.where(reset, 0.0, a)

        normalized_x = (gated * mult).astype(jnp.float32)
        a = a.astype(jnp.float32)

        a_t = jnp.transpose(a, (1, 0, 2))               # (T, B, size)
        nx_t = jnp.transpose(normalized_x, (1, 0, 2))

        def scan_step(h, an):
            a_step, nx_step = an
            h = a_step * h + nx_step
            return h, h

        batch = inputs.shape[0]
        init = jnp.zeros((batch, self.size), dtype=jnp.float32)
        _, out_t = jax.lax.scan(scan_step, init, (a_t, nx_t))
        out = jnp.transpose(out_t, (1, 0, 2))
        return out.astype(self.dtype)


class RglruDef(LayerDef):
    name = "rglru"

    def __init__(self, blocks: int, input_size: int):
        super().__init__(input_size=input_size)
        self.blocks = blocks
        self.size = input_size  # dimension-preserving
        self.block_width = (
            input_size // blocks if blocks and input_size % blocks == 0
            else 0)

    def __str__(self) -> str:
        return f"rglru.{self.blocks}"

    def is_valid(self) -> bool:
        # `blocks` is the only metaparameter; require it to be a power of
        # 2. For the power-of-2 channel widths the search uses, that makes
        # the block width a power of 2 too. The divisibility guard keeps
        # block_width a valid integer (e.g. blocks > width is rejected).
        # RecurrentGemma's blocks=10 is intentionally not search-valid --
        # that model is loaded outside the search.
        return (
            is_power2_int(self.blocks)
            and self.input_size % self.blocks == 0
        )

    @property
    def num_weights(self) -> int:
        # Learnable Lambda (per-channel decay, `size` params) + two
        # block-diagonal gate projections, each blocks*block_width^2
        # weights + blocks*block_width biases.
        lam = self.size
        gate = self.blocks * self.block_width * (self.block_width + 1)
        return lam + 2 * gate

    @property
    def projects_input(self) -> bool:
        # The gates are (block-diagonal) projections of x, but x itself
        # also enters elementwise via i * x -- a preceding bare dense
        # survives (Griffin's linear front-end feeds conv + rglru).
        return False

    def build_jax(self, dtype) -> RglruJax:
        return RglruJax(self.input_size, self.blocks, dtype)
