import jax
import jax.numpy as jnp

from ..common import is_power2_int, power2_neighbors
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Suffix(Layer):
    name = "suffix"

    def __init__(self, length, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.length = length
        self.output_shape = (self.length, self.input_size)
        self._state_shape = (self.length - 1, self.input_size)

    def __str__(self) -> str:
        return f"suffix.{self.length}"

    def is_valid(self) -> bool:
        return is_power2_int(self.length) and self.length > 1

    @property
    def weights(self) -> int:
        return 0

    def neighbors(self):
        if self.length > 2:
            l = self.length // 2
            yield f"suffix.{l}"
        l = self.length * 2
        yield f"suffix.{l}"
        input_size = max(8, self.input_size)
        yield f"attn.{self.length}.4.{input_size}"

    def init_weights(self, _rng: Rng, _init_scale: float = 1.0) -> LayerWeights:
        return None

    def init_state(self, _weights) -> LayerState:
        return jnp.ones(self._state_shape) / self.input_size

    def step(
        self, _weights: jnp.ndarray, state: LayerState, input: jnp.ndarray
    ) -> tuple[LayerState, jnp.ndarray]:
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix

    def forward(self, _weights, input: jnp.ndarray) -> jnp.ndarray:
        slices = []
        input_len = input.shape[0]
        for offset in range(self.length):
            slices.append(input[offset : input_len - self.length + offset + 1])
        return jnp.stack(slices, axis=1)

    def forward_batch(self, _weights, input: jnp.ndarray) -> jnp.ndarray:
        slices = []
        input_len = input.shape[1]
        for offset in range(self.length):
            slices.append(
                input[:, offset : input_len - self.length + offset + 1]
            )
        return jnp.stack(slices, axis=2)
