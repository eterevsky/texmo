import jax
import jax.numpy as jnp

from ..common import is_power2_int
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
        if self.input_size >= 4 and is_power2_int(self.input_size):
            yield f"att.{self.length}.4.{self.input_size}"

    def init_weights(self, _rng: Rng, _init_scale: float, dtype) -> LayerWeights:
        return None

    def init_state(self, _weights, dtype) -> LayerState:
        return jnp.zeros(self._state_shape, dtype=dtype)

    def step(
        self, _weights: jnp.ndarray, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix

    def forward(self, _weights, input: jnp.ndarray, dtype) -> jnp.ndarray:
        padded_input = jnp.concatenate(
            [jnp.zeros((self.length - 1, self.input_size), dtype=dtype), input]
        )
        input_len = padded_input.shape[0]
        slices = []
        for offset in range(self.length):
            slices.append(padded_input[offset : input_len - self.length + offset + 1])
        return jnp.stack(slices, axis=1)

    def forward_batch(self, _weights, input: jnp.ndarray, dtype) -> jnp.ndarray:
        batch = input.shape[0]
        padded_input = jnp.concatenate(
            [jnp.zeros((batch, self.length - 1, self.input_size), dtype=dtype), input], axis=1
        )
        input_len = padded_input.shape[1]

        slices = []
        for offset in range(self.length):
            slices.append(padded_input[:, offset : input_len - self.length + offset + 1])

        return jnp.stack(slices, axis=2)
