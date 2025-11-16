import jax
import jax.numpy as jnp

from ..common import is_power2_int, total_size
from ..layer import Layer, LayerState, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Gru(Layer):
    name = "gru"

    def __init__(self, size, input_shape=None, skip=False):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)
        # self._no_activation = not tanh and not relu
        if skip:
            assert total_size(input_shape) == total_size(self.output_shape)
        self._skip = skip

    def __str__(self) -> str:
        skip = ".skip" if self._skip else ""
        return f"gru.{self.size}{skip}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return 3 * self.size * (self.input_size + self.size + 1)

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "w": rng.he((2 * self.size, total_size), input_size=total_size, dtype=dtype) * init_scale,
            "b": rng.normal((2 * self.size,), dtype=dtype) * init_scale,
            "wh": rng.he((self.size, total_size), input_size=total_size, dtype=dtype) * init_scale,
            "bh": rng.normal((self.size,), dtype=dtype) * init_scale,
        }
        return weights

    def init_state(self, _weights, dtype) -> LayerState:
        return jnp.zeros((self.size,), dtype=dtype)

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jax.Array]:
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
        if self._skip:
            out = state + input
        else:
            out = state
        return state, out


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

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        total_size = self.input_size + self.size
        weights = {
            "wf": rng.he((self.size, total_size), input_size=total_size, dtype=dtype) * init_scale,
            "bf": rng.normal((self.size,), dtype=dtype) * init_scale,
            "wh": rng.he((self.size, total_size), input_size=total_size, dtype=dtype) * init_scale,
            "bh": rng.normal((self.size,), dtype=dtype) * init_scale,
        }
        return weights

    def init_state(self, _weights, dtype) -> LayerState:
        return jnp.zeros((self.size,), dtype=dtype)

    def step(
        self, weights: LayerWeights, state: LayerState, input: jax.Array, dtype
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()
        input_state = jnp.concatenate((input, state))
        f = jnp.dot(weights["wf"], input_state) + weights["bf"]
        f = jax.nn.sigmoid(f)

        input_state2 = jnp.concatenate((input, f * state))
        hc = jnp.dot(weights["wh"], input_state2) + weights["bh"]
        hc = jnp.tanh(hc)

        state = (1 - f) * state + f * hc
        return state, state


@layer_cls
class MinGru(Layer):
    """From https://arxiv.org/abs/2305.17473"""
    name = "mingru"

    def __init__(self, size, input_shape=None, skip=False):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)
        # self._no_activation = not tanh and not relu
        if skip:
            assert total_size(input_shape) == total_size(self.output_shape)
        self._skip = skip

    def __str__(self) -> str:
        return f"mingru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def weights(self) -> int:
        return 2 * self.size * (self.input_size + 1)

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        weights = {
            "w": rng.he((2 * self.size, self.input_size), input_size=self.input_size, dtype=dtype) * init_scale,
            "b": rng.normal((2 * self.size,), dtype=dtype) * init_scale,
        }
        return weights

    def init_state(self, _weights, dtype) -> LayerState:
        return jnp.zeros((self.size,), dtype=dtype)

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()
        zh = jnp.dot(weights["w"], input) + weights["b"]
        z = jax.nn.sigmoid(zh[:self.size])
        h = zh[self.size:]

        state = (1 - z) * state + z * h
        return state, state

    def forward_batch(
            self, weights: LayerWeights, inputs: jax.Array, dtype
    ) -> jax.Array:
        batch, length = inputs.shape[0:2]
        inputs = jnp.reshape(inputs, (batch, length, -1))

        # (batch, position, ...) -> (position, batch, ...)
        inputs = jnp.swapaxes(inputs, 0, 1)

        b = jnp.reshape(weights["b"], (1, 1, -1))
        zhs = jnp.einsum("oi,bni->bno", weights["w"], inputs) + b
        zs = jax.nn.sigmoid(zhs[:,:,:self.size])
        hs = zhs[:,:,self.size:]

        def step_batch(state, v):
            # Both should have dimentions (batch, size)
            z, h = v
            state = (1.0 - z) * state + z * h
            return state, state

        init_state = jnp.zeros((batch, self.size), dtype=dtype)

        _, out = jax.lax.scan(
            step_batch,
            init_state,
            (zs, hs)
        )
        return jnp.swapaxes(out, 0, 1)
