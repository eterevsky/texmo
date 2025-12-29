import jax
import jax.numpy as jnp

from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Norm(Layer):
    name = 'norm'

    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.output_shape = self.input_shape

    def __str__(self) -> str:
        return 'norm'

    def is_valid(self) -> bool:
        return self.input_size > 1

    @property
    def weights(self) -> int:
        return 0

    def neighbors(self):
        return []

    def init_weights(self, _rng: Rng, _init_scale: float, dtype) -> LayerWeights:
        return None

    def init_state(self, _weights, dtype) -> LayerState:
        return None

    def step(
        self, _weights: jnp.ndarray, _state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        input = input.flatten()
        norm = jnp.linalg.norm(input)
        output = input / (norm + 1E-5)
        return _state, output

    def forward_batch(
        self, _weights: LayerWeights, input: jax.Array, dtype
    ) -> jax.Array:
        batch_size, sample_len = input.shape[0:2]
        input = jnp.reshape(input, (batch_size, sample_len, -1))

        norms = jnp.linalg.norm(input, axis=-1)
        norms = jnp.where(norms > 0, norms, 1)
        norms = jnp.expand_dims(norms, 2)

        return input / norms
