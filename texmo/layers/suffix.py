import torch
import torch.nn as nn
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerModule, LayerState


class SuffixModule(LayerModule):
    """Sliding window layer: stacks the last `length` inputs into one vector."""

    def __init__(self, length: int, input_size: int):
        super().__init__()
        self.length = length
        self.input_size = input_size
        self.size = length * input_size

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(
            self.length - 1, self.input_size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        # state: (length-1, input_size), input: (input_size,)
        window = torch.cat([state, input.unsqueeze(0)], dim=0)
        new_state = window[1:]
        return new_state, window.reshape(-1)

    def forward(self, inputs: Tensor) -> Tensor:
        # Inputs: (batch, seq_len, input_size)
        # Output: (batch, seq_len - length + 1, length * input_size)
        seq_len = inputs.shape[1]
        out_len = seq_len - self.length + 1
        slices = []
        for offset in range(self.length):
            slices.append(inputs[:, offset:offset + out_len])
        return torch.cat(slices, dim=-1)


class SuffixDef(LayerDef):
    name = "suffix"

    def __init__(self, length: int, input_size: int):
        super().__init__(input_size=input_size)
        self.length = length
        self.size = length * input_size

    def __str__(self) -> str:
        return f"suffix.{self.length}"

    def is_valid(self) -> bool:
        return is_power2_int(self.length) and self.length > 1

    @property
    def num_weights(self) -> int:
        return 0

    def build_module(self, state_dict=None) -> SuffixModule:
        return SuffixModule(self.length, self.input_size)
