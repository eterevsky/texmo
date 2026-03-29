import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer_torch import LayerDef, LayerModule, LayerState


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


class DenseDef(LayerDef):
    name = "dense"

    def __init__(self, size, relu=False, tanh=False, gelu=False, input_size=None):
        super().__init__(input_size=input_size)
        self.size = size
        self.output_size = size

        if relu:
            assert not tanh
            self._activation_fn = F.relu
            self._activation_suffix = ".relu"
        elif tanh:
            self._activation_fn = torch.tanh
            self._activation_suffix = ".tanh"
        elif gelu:
            self._activation_fn = F.gelu
            self._activation_suffix = ".gelu"
        else:
            self._activation_fn = None
            self._activation_suffix = ""

    def __str__(self) -> str:
        return f"dense.{self.size}{self._activation_suffix}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size) and self._activation_fn is not None

    def neighbors(self):
        for n in super().neighbors():
            yield n

        if self._activation_suffix == '.tanh' and self.size > 1:
            yield f'latent.{self.size}.2.tanh'

    @property
    def num_weights(self) -> int:
        return self.size * self.input_size + self.size

    def build_module(self, state_dict: dict[str, Tensor] | None = None) -> DenseModule:
        module = DenseModule(self.input_size, self.size, self._activation_fn)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
