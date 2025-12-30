import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class S4(Layer):
    """SpaceTime-like S4 layer.

    A layer similar to suffix (stacking the last several outputs) of the previous
    """

    name = "s4"

    def __init__(self, length, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.length = length
        self.output_shape = (self.length, self.input_size)
        self._state_shape = (self.length, self.input_size)

    def __str__(self) -> str:
        return f's4.{self.length}'

    def is_valid(self) -> bool:
        return is_power2_int(self.length) and self.length >= 1

    @property
    def weights(self) -> int:
        return self.length * self.input_size

    def neighbors(self):
        if self.length >= 2:
            yield f'suffix.{self.length}'
            yield f's4.{self.length // 2}'
        yield f's4.{self.length * 2}'

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        return {
            'mix': rng.he((self.length, self.input_size), input_size=self.length*self.input_size, dtype=dtype) * 0.1
        }

    def init_state(self, _weights, dtype) -> LayerState:
        return jnp.zeros(self._state_shape, dtype=dtype)

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        outgoing = state[-1,:]
        new_state = jnp.vstack((input.reshape((1, -1)), state[:-1,:]))
        new_state += jnp.einsum('pi,i->pi', weights['mix'], outgoing)
        return new_state, new_state

    def step_batch(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        batch = input.shape[0]
        outgoing = state[:,-1,:]
        new_state = jnp.concatenate((input.reshape((batch, 1, -1)), state[:,:-1,:]), axis=1)
        new_state += jnp.einsum('pi,bi->bpi', weights['mix'], outgoing)
        return new_state, new_state

