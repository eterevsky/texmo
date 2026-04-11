import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerModule, LayerState


class RnnModule(LayerModule):
    """Simple Elman RNN backed by nn.RNN (cuDNN-optimized)."""

    def __init__(self, input_size: int, size: int, nonlinearity: str):
        super().__init__()
        self.size = size
        self.rnn = nn.RNN(input_size, size, nonlinearity=nonlinearity, batch_first=True)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        # input: (input_size,), state: (size,)
        _, h = self.rnn(
            input.flatten().unsqueeze(0).unsqueeze(0),
            state.unsqueeze(0).unsqueeze(0),
        )
        new_state = h.squeeze(0).squeeze(0)
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        output, _ = self.rnn(inputs)
        return output


class RnnGELUModule(LayerModule):
    """Elman RNN with GELU activation (not supported by nn.RNN)."""

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(input_size + size, size)

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
        return torch.zeros(self.size, device=device, dtype=dtype)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        input_state = torch.cat([input.flatten(), state])
        new_state = F.gelu(self.linear(input_state))
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        batch, seq_len, _ = inputs.shape
        state = torch.zeros(batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            state = F.gelu(self.linear(torch.cat([inputs[:, t, :], state], dim=-1)))
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class RnnDef(LayerDef):
    name = "rnn"

    def __init__(self, size: int, input_size: int, activation: str | None = None):
        super().__init__(input_size=input_size)
        self.size = size
        self._activation = activation

    def __str__(self) -> str:
        if self._activation:
            return f"rnn.{self.size}.{self._activation}"
        return f"rnn.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size) and self._activation is not None

    @property
    def num_weights(self) -> int:
        s = self.size
        if self._activation == "gelu":
            return s * (self.input_size + s) + s
        return s * self.input_size + s * s + 2 * s

    def build_module(self, state_dict: dict[str, Tensor] | None = None) -> LayerModule:
        if self._activation == "gelu":
            module = RnnGELUModule(self.input_size, self.size)
        else:
            module = RnnModule(self.input_size, self.size, self._activation)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
