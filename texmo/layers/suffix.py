import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..common import is_power2_int, power2_neighbors
from ..layer import Layer, LayerState
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
        yield f"attn.{self.length}.4.{self.input_size}"

    def init_weights(self, _rng, _init_scale) -> None:
        return None

    def init_state(self, _weights) -> LayerState:
        return jnp.ones(self._state_shape) / self.input_size

    def step(self, _weights, state: LayerState, input: DeviceArray) -> tuple[LayerState, DeviceArray]:
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix

    def forward(self, _weights, input: DeviceArray) -> DeviceArray:
        slices = []
        input_len = input.shape[0]
        for offset in range(self.length):
            slices.append(input[offset:input_len - self.length + offset + 1])
        return jnp.stack(slices, axis=1)

    def forward_batch(self, _weights, input: DeviceArray) -> DeviceArray:
        slices = []
        input_len = input.shape[1]
        for offset in range(self.length):
            slices.append(input[:,offset:input_len - self.length + offset + 1])
        return jnp.stack(slices, axis=2)
