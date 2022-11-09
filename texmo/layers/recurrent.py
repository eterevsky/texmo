import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..common import is_power2_int
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Rec(Layer):
    name = "rec"

    def __init__(self, size, relu=False, tanh=False, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)
        if relu:
            assert not tanh
            self.activation = jax.nn.relu
            self._activation_suffix = ".relu"
        elif tanh:
            self.activation = jnp.tanh
            self._activation_suffix = ".tanh"
        else:
            self.activation = None
            self._activation_suffix = ""

    def __str__(self) -> str:
        return f"rec.{self.size}{self._activation_suffix}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return self.size * self.input_size + self.size * self.size + self.size

    def init_weights(self, rng: Rng, init_scale: float) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "w": rng.he((self.size, total_size), input_size=total_size) * init_scale,
            "b": rng.normal((self.size,)) * init_scale,
        }
        return weights

    def init_state(self, _weights) -> LayerState:
        return jnp.zeros((self.size,))

    def step(
        self, weights: LayerWeights, state: LayerState, input: DeviceArray
    ) -> tuple[LayerState, DeviceArray]:
        input = input.flatten()
        input_state = jnp.concatenate((input, state))
        new_state = jnp.dot(weights["w"], input_state) + weights["b"]
        if self.activation is not None:
            new_state = self.activation(new_state)
        return new_state, new_state

    # use_step_batch = True
    # def step_batch(
    #     self, weights: LayerWeights, state: LayerState, input: DeviceArray
    # ) -> tuple[LayerState, DeviceArray]:
    #     input = input.reshape((input.shape[0], -1))

    #     print("input", input.shape)
    #     print("state", state.shape)
    #     print("winput", weights["winput"].shape)
    #     print("wstate", weights["wstate"].shape)
    #     print("b", jnp.expand_dims(weights["b"], 0).shape)
    #     new_state = (
    #         jnp.einsum("oi,bi->bo", weights["winput"], input) +
    #         jnp.einsum("oi,bi->bo", weights["wstate"], state) +
    #         jnp.expand_dims(weights["b"], 0)
    #     )
    #     if self.activation is not None:
    #         new_state = self.activation(new_state)
    #     return new_state, new_state
