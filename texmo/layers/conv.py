import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights


class ConvJax(LayerJax):
    """Depthwise causal 1D convolution along the time axis.

    Output for channel c, valid (un-padded) frame position t:

        y[c, t] = sum_{k=0..L-1} w[k, c] * x[c, t + L-1 - k] + b[c]

    Per-channel kernels (no cross-channel mixing) and per-channel
    biases. The L-1 leading "transient" positions (whose receptive
    field would include positions before the first input) are NOT
    emitted by `forward` -- this matches the suffix-like convention
    the framework expects: every layer with `LayerDef.length == L`
    consumes L-1 positions, and `Model.total_padding` accumulates
    them. For step-based inference the framework instead warms up
    each stateful layer with L-1 padding iterations before any real
    token, so `step` itself emits one output per call.

    Input shape (B, T, D) -> output (B, T - L + 1, D).
    """

    def __init__(self, input_size: int, kernel: int, dtype):
        # Depthwise conv preserves channel count.
        super().__init__(input_size, input_size, dtype)
        self.kernel = kernel

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        # Small normal init scaled by 1/sqrt(L) so the per-channel sum
        # of L weighted inputs starts at unit variance.
        scale = 1.0 / jnp.sqrt(jnp.array(self.kernel, dtype=self.dtype))
        w = jax.random.normal(
            rng, (self.kernel, self.input_size), dtype=self.dtype) * scale
        return {
            'w': w,
            'b': jnp.zeros(self.input_size, dtype=self.dtype),
        }

    def init_state(self):
        # Buffer of the last (L-1) inputs, oldest first. Zero-padded
        # at the start of the sequence -- matches the forward()
        # left-pad convention.
        return jnp.zeros(
            (self.kernel - 1, self.input_size), dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # state: (L-1, D) oldest-first buffer of past inputs.
        # x:     (D,)     current input.
        w = weights['w']  # (L, D)
        b = weights['b']  # (D,)
        # Combined: x[t-L+1], ..., x[t-1], x[t] -- length L, newest last.
        combined = jnp.concatenate([state, x[jnp.newaxis, :]], axis=0)
        # y[c] = sum_k w[k, c] * x[c, t-k] = sum_k w[k, c] * combined[L-1-k, c].
        # Reversing w along axis 0 aligns it with `combined`.
        y = jnp.sum(w[::-1] * combined, axis=0) + b
        # Drop oldest, append current -- ready for the next step.
        new_state = combined[1:]
        return new_state, y

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (B, T, D). Output: (B, T - L + 1, D).
        # "Valid" convolution -- the L-1 leading transient positions
        # (those whose receptive field would reach before the start
        # of `inputs`) are not emitted. The framework allocates
        # `total_padding` extra positions at the model input to
        # cover that consumption.
        w = weights['w']  # (L, D)
        b = weights['b']  # (D,)
        L = self.kernel
        out_len = inputs.shape[1] - L + 1
        # y[t] = sum_k w[k] * inputs[t + L-1 - k], t in [0, out_len).
        # For each kernel position k, slice inputs[L-1-k : L-1-k+out_len].
        out = w[0] * inputs[:, L - 1: L - 1 + out_len, :]
        for k in range(1, L):
            out = out + w[k] * inputs[:, L - 1 - k: L - 1 - k + out_len, :]
        return out + b


class ConvDef(LayerDef):
    name = "conv"

    def __init__(self, kernel: int, input_size: int):
        super().__init__(input_size=input_size)
        self.kernel = kernel
        self.size = input_size  # depthwise preserves channels
        # length = kernel: conv reads `kernel` past positions, consumes
        # `kernel - 1` of them via valid-conv output trimming, and
        # qualifies as "suffix-like" for the framework's adjacency
        # rule. That rule forbids stacking conv with conv, conv with
        # suffix, or suffix with conv -- mathematically redundant
        # (conv-conv collapses to a wider conv) or architecturally
        # awkward (suffix-conv channel anchors get smeared).
        self.length = kernel

    def __str__(self) -> str:
        return f"conv.{self.kernel}"

    def is_valid(self) -> bool:
        # L >= 2 (a length-1 kernel collapses to a per-channel scale
        # + bias -- equivalent to a no-op for the search). Power of 2
        # to match the suffix mutation pattern.
        return is_power2_int(self.kernel) and self.kernel >= 2

    @property
    def num_weights(self) -> int:
        # Per-channel kernel (L * D) plus per-channel bias (D).
        return self.input_size * (self.kernel + 1)

    @property
    def projects_input(self) -> bool:
        # Depthwise: per-channel temporal kernels, no cross-channel
        # matmul -- a preceding bare dense doesn't collapse into it.
        return False

    def build_jax(self, dtype) -> ConvJax:
        return ConvJax(self.input_size, self.kernel, dtype)
