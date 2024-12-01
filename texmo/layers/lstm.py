import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Lstm(Layer):
    name = "lstm"

    def __init__(self, size, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)

    def __str__(self) -> str:
        return f"lstm.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return 4 * self.size * (self.input_size + self.size + 1)

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "w": rng.he((4 * self.size, total_size), input_size=total_size, dtype=dtype) * init_scale,
            "b": rng.normal((4 * self.size,), dtype=dtype) * init_scale,
        }
        return weights

    def init_state(self, _weights, dtype) -> LayerState:
        return {
            "h": jnp.zeros((self.size,), dtype=dtype),
            "c": jnp.zeros((self.size,), dtype=dtype),
        }

    def step(
        self, weights: LayerWeights, state: LayerState, input: jax.Array, dtype
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()

        h = state["h"]
        c = state["c"]

        input_h = jnp.concatenate((input, h))

        everything = jnp.dot(weights["w"], input_h) + weights["b"]
        fio = everything[:3 * self.size]
        fio = jax.nn.sigmoid(fio)
        cn = everything[3 * self.size:]
        cn = jax.nn.tanh(cn)

        f = fio[:self.size]
        i = fio[self.size:2*self.size]
        o = fio[2*self.size:]

        c = f * c + i * cn
        h = o * c

        return {"h": h, "c": c}, h
