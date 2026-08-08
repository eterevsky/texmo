import jax
import jax.numpy as jnp

from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights


# Tikhonov-smoothed L2 normalization (matches _normalize_jax in
# latent.py). Using `sqrt(||x||^2 + eps)` rather than
# `max(||x||, eps)` makes the gradient well-defined at x=0; see the
# latent.py docstring for why we add `eps` (not `eps*eps`) inside
# sqrt -- it keeps the smoothing meaningful in fp16/bf16.
_EPS = 1e-5


class NormJax(LayerJax):
    """L2 normalization along the feature dimension.

    out = x / sqrt(||x||_2^2 + eps)
    """

    def __init__(self, input_size: int, dtype):
        super().__init__(input_size, input_size, dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        return None, x / jnp.sqrt(jnp.sum(x * x) + _EPS)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, size)
        norm_sq = jnp.sum(inputs * inputs, axis=-1, keepdims=True)
        return inputs / jnp.sqrt(norm_sq + _EPS)


class NormDef(LayerDef):
    name = "norm"

    def __init__(self, input_size: int):
        super().__init__(input_size=input_size)
        self.size = input_size

    def __str__(self) -> str:
        return "norm"

    def is_valid(self) -> bool:
        return self.input_size > 1

    @property
    def num_weights(self) -> int:
        return 0

    @property
    def projects_input(self) -> bool:
        return False  # nonlinear elementwise rescale, no matmul

    def build_jax(self, dtype) -> NormJax:
        return NormJax(self.input_size, dtype)
