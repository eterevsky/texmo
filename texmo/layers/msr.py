import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform


ROPE_BASE = 10000.0


def _gammas(heads: int) -> list[float]:
    return [1.0 - 2.0 ** (-5 - h) for h in range(heads)]


def _thetas(dim: int) -> list[float]:
    return [ROPE_BASE ** (-2.0 * j / dim) for j in range(dim // 2)]


def _rope_jax_step(x, pos, thetas):
    angles = pos.astype(thetas.dtype) * thetas
    c = jnp.cos(angles).astype(x.dtype)
    s = jnp.sin(angles).astype(x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * c - x_odd * s
    rot_odd = x_even * s + x_odd * c
    return jnp.stack([rot_even, rot_odd], axis=-1).reshape(x.shape)


def _rope_jax_seq(x, positions, thetas):
    angles = positions[:, None].astype(thetas.dtype) * thetas[None, :]
    c = jnp.cos(angles).astype(x.dtype)
    s = jnp.sin(angles).astype(x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * c - x_odd * s
    rot_odd = x_even * s + x_odd * c
    return jnp.stack([rot_even, rot_odd], axis=-1).reshape(x.shape)


class MsrJax(LayerJax):
    """Multi-Scale Retention (RetNet, Sun et al. 2023).

    Per-head linear-attention recurrence with fixed scalar decay
    γ_h = 1 - 2^(-5-h) and RoPE rotation θ_j = ROPE_BASE^(-2j/dim)
    shared across heads. No biases on Q/K/V, no output projection,
    no GroupNorm — stack `norm` or `dense.<size>` after as needed.

        S_h_t = γ_h · S_h_{t-1} + RoPE(k_h_t)^T v_h_t
        o_h_t = RoPE(q_h_t) · S_h_t

    Weights:
        w_qkv: (3 * heads * dim, input_size), the stacked [Q; K; V]
        projection — one matmul per token.
    State:
        (S, pos) where S has shape (heads, dim, dim) — a per-head
        outer-product accumulator — and pos is an int32 scalar
        tracking the absolute position for RoPE.

    Output shape: (..., heads * dim).

    The decay and rotation tables are kept in fp32: γ^t for high h and
    θ for low h both fall below bf16 precision around t ≈ 100.
    """

    def __init__(self, input_size: int, dim: int, heads: int, dtype=jnp.float32):
        super().__init__(input_size, heads * dim, dtype)
        self.dim = dim
        self.heads = heads
        self.gammas = jnp.array(_gammas(heads), dtype=jnp.float32)
        self.thetas = jnp.array(_thetas(dim), dtype=jnp.float32)
        # Lazy fp32 decay-matrix cache keyed by sequence length T.
        self._decay_cache: dict[int, jax.Array] = {}

    def _decay_matrix(self, T: int) -> jax.Array:
        cached = self._decay_cache.get(T)
        if cached is not None:
            return cached
        positions = jnp.arange(T, dtype=jnp.float32)
        diff = positions[:, None] - positions[None, :]
        mask = (diff >= 0).astype(jnp.float32)
        decay = (
            self.gammas.reshape(self.heads, 1, 1) ** diff[None, :, :]
        ) * mask[None, :, :]
        self._decay_cache[T] = decay
        return decay

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        return {
            'w_qkv': xavier_uniform(
                rng, (3 * self.size, self.input_size), dtype=self.dtype),
        }

    def init_state(self):
        S = jnp.zeros(
            (self.heads, self.dim, self.dim), dtype=self.dtype)
        pos = jnp.zeros((), dtype=jnp.int32)
        return (S, pos)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        S, pos = state
        H, D = self.heads, self.dim

        qkv = (weights['w_qkv'] @ x).reshape(3, H, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_rot = _rope_jax_step(q, pos, self.thetas)
        k_rot = _rope_jax_step(k, pos, self.thetas)

        gamma = self.gammas.reshape(H, 1, 1).astype(S.dtype)
        kv = k_rot[:, :, None] * v[:, None, :]
        S_new = gamma * S + kv

        o = jnp.einsum('hd,hde->he', q_rot, S_new)
        return (S_new, pos + 1), o.reshape(self.size)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        B, T, _ = inputs.shape
        H, D = self.heads, self.dim

        qkv = (inputs @ weights['w_qkv'].T).reshape(B, T, 3, H, D)
        # (B, T, H, D) → (B, H, T, D)
        Q = jnp.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        K = jnp.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        V = jnp.transpose(qkv[:, :, 2], (0, 2, 1, 3))

        positions = jnp.arange(T, dtype=jnp.float32)
        Q_rot = _rope_jax_seq(Q, positions, self.thetas)
        K_rot = _rope_jax_seq(K, positions, self.thetas)

        scores = jnp.einsum('bhtd,bhsd->bhts', Q_rot, K_rot)
        decayed = scores * self._decay_matrix(T).astype(inputs.dtype)

        out = jnp.einsum('bhts,bhsd->bhtd', decayed, V)
        return jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T, H * D)


class MsrDef(LayerDef):
    name = "msr"

    def __init__(self, dim: int, heads: int, input_size: int):
        super().__init__(input_size=input_size)
        self.dim = dim
        self.heads = heads
        self.size = heads * dim

    def __str__(self) -> str:
        return f"msr.{self.dim}.{self.heads}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.dim) and self.dim >= 2
            and is_power2_int(self.heads)
        )

    @property
    def num_weights(self) -> int:
        # 3 bias-free projections: Q, K, V, each (heads*dim, input_size).
        return 3 * self.heads * self.dim * self.input_size

    @property
    def num_mults(self) -> int:
        # Projections (= num_weights) plus the per-head state update
        # (outer product, D^2 mults) and read (q · S, D^2 mults). The
        # rotation adds 2·H·D, negligible at these sizes.
        return self.num_weights + 2 * self.heads * self.dim * self.dim

    def build_jax(self, dtype) -> MsrJax:
        return MsrJax(self.input_size, self.dim, self.heads, dtype)
