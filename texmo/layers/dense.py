import jax
import jax.nn as jax_nn
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax, LayerModule, LayerState
from ..layer_jax import LayerWeights, xavier_uniform


class DenseModule(LayerModule):
    def __init__(self, input_size: int, size: int, activation):
        super().__init__()
        self.linear = nn.Linear(input_size, size)
        self.activation = activation

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        assert input.shape == (self.linear.in_features,)
        out = self.linear(input)
        if self.activation is not None:
            out = self.activation(out)
        return None, out

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.shape[-1] == self.linear.in_features
        out = self.linear(inputs)
        if self.activation is not None:
            out = self.activation(out)
        return out


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
}

_JAX_ACTIVATIONS = {
    "relu": jax_nn.relu,
    "tanh": jnp.tanh,
    "gelu": jax_nn.gelu,
}


class DenseJax(LayerJax):
    def __init__(self, input_size: int, size: int, activation_name: str | None, dtype):
        super().__init__(input_size, size, dtype)
        self.activation = _JAX_ACTIVATIONS.get(activation_name)

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        return {
            'w': xavier_uniform(rng, (self.size, self.input_size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def _apply(self, x, weights):
        out = x @ weights['w'].T + weights['b']
        return self.activation(out) if self.activation else out

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        return None, self._apply(x, weights)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        return self._apply(inputs, weights)


class DenseDef(LayerDef):
    name = "dense"

    def __init__(self, size: int, input_size: int, activation: str | None = None):
        super().__init__(input_size=input_size)
        self.size = size
        self._activation = activation
        self._activation_fn = _ACTIVATIONS[activation] if activation else None

    def __str__(self) -> str:
        if self._activation:
            return f"dense.{self.size}.{self._activation}"
        return f"dense.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size) and self._activation_fn is not None

    @property
    def num_weights(self) -> int:
        return self.size * self.input_size + self.size

    def build_module(self, state_dict: dict[str, Tensor] | None = None) -> DenseModule:
        module = DenseModule(self.input_size, self.size, self._activation_fn)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module

    def build_jax(self, dtype) -> DenseJax:
        return DenseJax(self.input_size, self.size, self._activation, dtype)
