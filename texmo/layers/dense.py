import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..common import total_size
from ..layer import Layer, LayerWeights
from ..prng import Rng
from .registry import layer_cls


@layer_cls
class Dense(Layer):
    name = "dense"

    def __init__(self, size, relu=False, tanh=False, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.output_shape = (size,)
        if relu:
            assert not tanh
            self.activation = jax.nn.relu
        elif tanh:
            self.activation = jnp.tanh
        else:
            self.activation = None

    @property
    def weights(self) -> int:
        return self.size * self.input_size + self.size

    def init_weights(self, rng: Rng, init_scale: float) -> LayerWeights:
        return {
            "w": rng.he((self.size, self.input_size)) * init_scale,
            "b": rng.normal((self.size,)) * init_scale,
        }

    def step(
        self, weights: LayerWeights, _state: None, input: DeviceArray
    ) -> tuple[None, DeviceArray]:
        input = input.flatten()
        out = jnp.dot(weights["w"], input) + weights["b"]
        if self.activation is not None:
            out = self.activation(out)
        return None, out

    def step_batch(
        self, weights: LayerWeights, _state: None, input: DeviceArray
    ) -> tuple[None, DeviceArray]:
        batch_size = input.shape[0]
        input = jnp.reshape(input, (batch_size, -1))
        # out = jnp.matmul(input, weights["w"].transpose()) + jnp.expand_dims(weights["b"], 0)

        out = jnp.einsum("oi,bi->bo", weights["w"], input) + jnp.expand_dims(weights["b"], 0)
        print("out:", out.shape)
        if self.activation is not None:
            out = self.activation(out)
        return None, out

    def forward(self, weights: LayerWeights, input: DeviceArray) -> DeviceArray:
        sample_len = input.shape[0]
        input = jnp.reshape(input, (sample_len, -1))
        # out = jnp.matmul(input, weights["w"].transpose()) + jnp.expand_dims(weights["b"], 0)

        out = jnp.einsum("oi,ni->no", weights["w"], input) + jnp.expand_dims(weights["b"], 0)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def forward_batch(self, weights: LayerWeights, input: DeviceArray) -> DeviceArray:
        batch_size, sample_len = input.shape[0:2]
        input = jnp.reshape(input, (batch_size, sample_len, -1))
        # out = jnp.matmul(input, weights["w"].transpose()) + jnp.expand_dims(weights["b"], 0)

        out = jnp.einsum("oi,bni->bno", weights["w"], input) + jnp.reshape(weights["b"], (1, 1, -1))
        if self.activation is not None:
            out = self.activation(out)
        return out
