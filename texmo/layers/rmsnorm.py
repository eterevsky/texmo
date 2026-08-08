"""RMSNorm with a learned per-channel scale, as used in Gemma /
RecurrentGemma:

    out = x * rsqrt(mean(x^2, axis=-1) + eps) * (1 + gamma)

`gamma` is a per-channel vector initialised to zero, so the layer starts
as pure RMS normalisation (scale = 1) and weight decay regularises
toward identity. Distinct from texmo's `norm`, which is parameter-free
L2 normalisation; the only substantive difference is this learned scale.

The reduction and the (1 + gamma) multiply run in float32 and the result
is cast back to the layer dtype -- matching transformers'
RecurrentGemmaRMSNorm so pretrained weights load and run faithfully.

Dimension-preserving and stateless.
"""
import jax
import jax.numpy as jnp

from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights

# Gemma / RecurrentGemma RMSNorm epsilon.
_EPS = 1e-6


class RmsNormJax(LayerJax):
    """Gemma-style RMSNorm: out = x * rsqrt(mean(x^2)+eps) * (1+gamma)."""

    def __init__(self, input_size: int, dtype):
        super().__init__(input_size, input_size, dtype)

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        # gamma init 0 -> (1 + gamma) = 1 -> identity scale at start.
        return {'gamma': jnp.zeros(self.size, dtype=self.dtype)}

    def init_state(self):
        return None

    def _rmsnorm(self, weights: LayerWeights, x: jax.Array) -> jax.Array:
        # Reduce over the (last) feature axis. Works for step input
        # (size,) and forward input (batch, seq_len, size) alike.
        x32 = x.astype(jnp.float32)
        ms = jnp.mean(x32 * x32, axis=-1, keepdims=True)
        normed = x32 * jax.lax.rsqrt(ms + _EPS)
        scaled = normed * (1.0 + weights['gamma'].astype(jnp.float32))
        return scaled.astype(self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        return None, self._rmsnorm(weights, x)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        return self._rmsnorm(weights, inputs)


class RmsNormDef(LayerDef):
    name = "rmsnorm"

    def __init__(self, input_size: int):
        super().__init__(input_size=input_size)
        self.size = input_size  # dimension-preserving

    def __str__(self) -> str:
        return "rmsnorm"

    def is_valid(self) -> bool:
        # A 1-channel RMS norm is degenerate (mean over one element).
        return self.input_size > 1

    @property
    def num_weights(self) -> int:
        return self.size  # the learned per-channel gamma

    @property
    def projects_input(self) -> bool:
        return False  # nonlinear elementwise rescale, no matmul

    def build_jax(self, dtype) -> RmsNormJax:
        return RmsNormJax(self.input_size, dtype)
