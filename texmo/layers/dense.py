import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..layer import Layer, LayerWeights
from ..prng import Rng
from .registry import layer_cls


# dense.128.relu batch=128 16 seconds:
#
# auto use_step_batch=False -> 246
# auto use_step_batch=True -> 252
# custom step_batch -> 262
# custom forward -> 333
# custom forward_batch -> 342
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
            self._activation_suffix = ".relu"
        elif tanh:
            self.activation = jnp.tanh
            self._activation_suffix = ".tanh"
        else:
            self.activation = None
            self._activation_suffix = ""

    def __str__(self) -> str:
        return f"dense.{self.size}{self._activation_suffix}"

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

        out = jnp.einsum("oi,bi->bo", weights["w"], input) + jnp.expand_dims(
            weights["b"], 0
        )
        print("out:", out.shape)
        if self.activation is not None:
            out = self.activation(out)
        return None, out

    def forward(self, weights: LayerWeights, input: DeviceArray) -> DeviceArray:
        sample_len = input.shape[0]
        input = jnp.reshape(input, (sample_len, -1))
        out = jnp.einsum("oi,ni->no", weights["w"], input) + jnp.expand_dims(
            weights["b"], 0
        )
        if self.activation is not None:
            out = self.activation(out)
        return out

    def forward_batch(
        self, weights: LayerWeights, input: DeviceArray
    ) -> DeviceArray:
        batch_size, sample_len = input.shape[0:2]
        input = jnp.reshape(input, (batch_size, sample_len, -1))
        b = jnp.reshape(weights["b"], (1, 1, -1))
        out = jnp.einsum("oi,bni->bno", weights["w"], input) + b
        if self.activation is not None:
            out = self.activation(out)
        return out
