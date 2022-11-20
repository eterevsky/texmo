import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..common import is_power2_int
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Gru(Layer):
    name = "gru"

    def __init__(self, size, input_shape=None, tanh=False, relu=False):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)
        self._no_activation = not tanh and not relu

    def __str__(self) -> str:
        return f"gru.{self.size}"

    def is_valid(self) -> bool:
        return self._no_activation and is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return 3 * self.size * (self.input_size + self.size + 1)

    def init_weights(self, rng: Rng, init_scale: float) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "w": rng.he((2 * self.size, total_size), input_size=total_size) * init_scale,
            "b": rng.normal((2 * self.size,)) * init_scale,
            "wh": rng.he((self.size, total_size), input_size=total_size) * init_scale,
            "bh": rng.normal((self.size,)) * init_scale,
        }
        return weights

    def init_state(self, _weights) -> LayerState:
        return jnp.zeros((self.size,))

    def step(
        self, weights: LayerWeights, state: LayerState, input: DeviceArray
    ) -> tuple[LayerState, DeviceArray]:
        input = input.flatten()
        input_state = jnp.concatenate((input, state))
        zr = jnp.dot(weights["w"], input_state) + weights["b"]
        zr = jax.nn.sigmoid(zr)
        z = zr[:self.size]
        r = zr[self.size:]

        input_state2 = jnp.concatenate((input, r * state))
        hc = jnp.dot(weights["wh"], input_state2) + weights["bh"]
        hc = jnp.tanh(hc)

        state = (1 - z) * state + z * hc
        return state, state


@layer_cls
class Mgru(Layer):
    name = "mgru"

    def __init__(self, size, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)

    def __str__(self) -> str:
        return f"mgru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return 2 * self.size * (self.input_size + self.size + 1)

    def init_weights(self, rng: Rng, init_scale: float) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "wf": rng.he((self.size, total_size), input_size=total_size) * init_scale,
            "bf": rng.normal((self.size,)) * init_scale,
            "wh": rng.he((self.size, total_size), input_size=total_size) * init_scale,
            "bh": rng.normal((self.size,)) * init_scale,
        }
        return weights

    def init_state(self, _weights) -> LayerState:
        return jnp.zeros((self.size,))

    def step(
        self, weights: LayerWeights, state: LayerState, input: DeviceArray
    ) -> tuple[LayerState, DeviceArray]:
        input = input.flatten()
        input_state = jnp.concatenate((input, state))
        f = jnp.dot(weights["wf"], input_state) + weights["bf"]
        f = jax.nn.sigmoid(f)

        input_state2 = jnp.concatenate((input, f * state))
        hc = jnp.dot(weights["wh"], input_state2) + weights["bh"]
        hc = jnp.tanh(hc)

        state = (1 - f) * state + f * hc
        return state, state
