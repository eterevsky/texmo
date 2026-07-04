"""Local (sliding-window) multi-query attention with rotary position
embeddings -- the attention block of Griffin / RecurrentGemma.

Spec: `attn.{size}.{heads}.{window}` with `head_dim = size / heads`.

    q = W_q x                      # (heads * head_dim,) -- h distinct heads
    k = W_k x                      # (head_dim,)         -- ONE shared head (MQA)
    v = W_v x                      # (head_dim,)         -- ONE shared head
    rope(q_i), rope(k)             # split-half pairs, first half of head_dim
    s_i = softmax(q_i . k_past / sqrt(head_dim))   over the last `window`
                                                   positions (incl. current)
    out = W_o concat_i(s_i @ v_past) + b_o         # (size,)

Matches `transformers` RecurrentGemmaAttention numerically so pretrained
weights load directly:
  - MQA: one shared K/V head; per-position KV state is 2*head_dim.
  - RoPE in the split-half convention -- pairs are (x_j, x_{j+rd/2}),
    NOT the interleaved (x_{2j}, x_{2j+1}) pairs msr uses -- applied to
    the first `head_dim * _ROTARY_FRACTION` dims of q and k only; the
    remainder passes through unrotated. Angles computed in float32.
  - Banded causal mask: position t attends to (t-window, t] -- `window`
    positions including itself. Position 0 attends to itself only (the
    shift-pad initial vector plays the BOS role); no leading padding is
    consumed, so `length = 1` -- history lives in the step-mode state (a
    rolling K/V buffer), like an RNN's hidden state, not in padding.
  - Scores scaled by head_dim**-0.5; softmax in float32.
  - Bias on the output projection only (q/k/v are bias-free).

JAX only, like the other Griffin layers.
"""
import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform

# Fraction of head_dim that gets rotated (RecurrentGemma: 0.5). A module
# constant for now; promote to a spec parameter if the search wants it.
_ROTARY_FRACTION = 0.5
# RoPE base frequency (RecurrentGemma rope_theta).
_ROPE_BASE = 10000.0
# Additive mask value for disallowed positions (fp32 softmax).
_NEG_INF = -1e30


def _rotary_dim(head_dim: int) -> int:
    return int(head_dim * _ROTARY_FRACTION)


def _rope_cos_sin(positions: jax.Array, rd: int) -> tuple[jax.Array, jax.Array]:
    """cos/sin tables for split-half RoPE, shape (..., rd), float32.

    `positions` is any-shaped int/float array of absolute positions;
    frequencies ladder over the rotary dim: inv_freq[j] = base^(-2j/rd).
    """
    inv_freq = 1.0 / (
        _ROPE_BASE ** (jnp.arange(0, rd, 2, dtype=jnp.float32) / rd)
    )
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq  # (..., rd/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)               # (..., rd)
    return jnp.cos(emb), jnp.sin(emb)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate the first `rd` dims of x (split-half pairing); the rest
    pass through. cos/sin broadcast against x's leading dims."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    half = rd // 2
    rotated = jnp.concatenate(
        [-x_rot[..., half:], x_rot[..., :half]], axis=-1)
    x_rot = x_rot * cos + rotated * sin
    return jnp.concatenate([x_rot, x_pass], axis=-1)


class AttnJax(LayerJax):
    """Sliding-window MQA. Weights: w_q (size, D), w_k/w_v (head_dim, D),
    w_o (size, size), b_o (size,). Step state: rolling K/V buffer of the
    last window-1 positions (K stored post-rotation) + position counter.
    """

    def __init__(
        self, input_size: int, size: int, heads: int, window: int, dtype,
    ):
        super().__init__(input_size, size, dtype)
        self.heads = heads
        self.head_dim = size // heads
        self.window = window

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        k_q, k_k, k_v, k_o = jax.random.split(rng, 4)
        hd = self.head_dim
        return {
            'w_q': xavier_uniform(
                k_q, (self.size, self.input_size), dtype=self.dtype),
            'w_k': xavier_uniform(
                k_k, (hd, self.input_size), dtype=self.dtype),
            'w_v': xavier_uniform(
                k_v, (hd, self.input_size), dtype=self.dtype),
            'w_o': xavier_uniform(
                k_o, (self.size, self.size), dtype=self.dtype),
            'b_o': jnp.zeros(self.size, dtype=self.dtype),
        }

    def init_state(self):
        hd = self.head_dim
        w = self.window - 1
        return (
            jnp.zeros((w, hd), dtype=self.dtype),   # roped keys, oldest first
            jnp.zeros((w, hd), dtype=self.dtype),   # values
            jnp.zeros((), dtype=jnp.int32),         # tokens seen so far
        )

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (B, T, D)
        B, T, _ = inputs.shape
        h, hd = self.heads, self.head_dim

        q = (inputs @ weights['w_q'].T).reshape(B, T, h, hd)
        k = inputs @ weights['w_k'].T                     # (B, T, hd)
        v = inputs @ weights['w_v'].T                     # (B, T, hd)

        positions = jnp.arange(T)
        cos, sin = _rope_cos_sin(positions, _rotary_dim(hd))  # (T, rd)
        q = _apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])
        k = _apply_rope(k, cos[None, :, :], sin[None, :, :])

        # Scores vs the shared K head: (B, h, T, S), fp32 softmax.
        scale = float(hd) ** -0.5
        scores = jnp.einsum('bthd,bsd->bhts', q, k) * scale
        i = positions[:, None]
        j = positions[None, :]
        allowed = (j <= i) & (j > i - self.window)
        scores = jnp.where(
            allowed[None, None], scores.astype(jnp.float32), _NEG_INF)
        attn = jax.nn.softmax(scores, axis=-1).astype(self.dtype)

        out = jnp.einsum('bhts,bsd->bthd', attn, v)       # (B, T, h, hd)
        out = out.reshape(B, T, self.size)
        return out @ weights['w_o'].T + weights['b_o']

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[tuple, jax.Array]:
        # x: (D,). Buffers hold the previous window-1 positions.
        k_buf, v_buf, pos = state
        h, hd, w = self.heads, self.head_dim, self.window

        q = (weights['w_q'] @ x).reshape(h, hd)
        k = weights['w_k'] @ x                            # (hd,)
        v = weights['w_v'] @ x

        cos, sin = _rope_cos_sin(pos, _rotary_dim(hd))    # (rd,)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Attend over [buffer, current] = `window` slots; only the last
        # min(pos, window-1) buffer entries hold real history.
        k_all = jnp.concatenate([k_buf, k[None]], axis=0)   # (w, hd)
        v_all = jnp.concatenate([v_buf, v[None]], axis=0)
        valid = jnp.arange(w) >= (w - 1) - jnp.minimum(pos, w - 1)

        scale = float(hd) ** -0.5
        scores = (q @ k_all.T) * scale                      # (h, w)
        scores = jnp.where(
            valid[None], scores.astype(jnp.float32), _NEG_INF)
        attn = jax.nn.softmax(scores, axis=-1).astype(self.dtype)

        out = (attn @ v_all).reshape(self.size)
        out = weights['w_o'] @ out + weights['b_o']

        new_state = (
            jnp.concatenate([k_buf[1:], k[None]], axis=0),
            jnp.concatenate([v_buf[1:], v[None]], axis=0),
            pos + 1,
        )
        return new_state, out


class AttnDef(LayerDef):
    name = "attn"

    def __init__(
        self, size: int, heads: int, window: int, input_size: int,
    ):
        super().__init__(input_size=input_size)
        self.size = size
        self.heads = heads
        self.window = window
        self.head_dim = size // heads if heads and size % heads == 0 else 0

    def __str__(self) -> str:
        return f"attn.{self.size}.{self.heads}.{self.window}"

    def is_valid(self) -> bool:
        # head_dim >= 4 so the rotary half still has at least one
        # rotatable pair; window >= 2 so there's some actual context.
        # (RecurrentGemma's attn.2560.10.2048 is not search-valid --
        # non-power-of-2 -- and is loaded outside the search, like
        # rglru.10.)
        return (
            is_power2_int(self.size)
            and is_power2_int(self.heads)
            and self.size % max(self.heads, 1) == 0
            and self.head_dim >= 4
            and is_power2_int(self.window)
            and self.window >= 2
        )

    @property
    def num_weights(self) -> int:
        # q (size x D) + shared k, v (head_dim x D each) + o (size^2)
        # + o bias (q/k/v are bias-free, matching the reference).
        return (
            self.size * self.input_size
            + 2 * self.head_dim * self.input_size
            + self.size * self.size
            + self.size
        )

    @property
    def num_mults(self) -> int:
        # Projections (~num_weights) plus the per-token attention work:
        # scores q.k and the value mix each cost ~window * size.
        return self.num_weights + 2 * self.window * self.size

    def build_jax(self, dtype) -> AttnJax:
        return AttnJax(
            self.input_size, self.size, self.heads, self.window, dtype)
