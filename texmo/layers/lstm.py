import torch
import torch.nn as nn
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerModule, LayerState


class LstmModule(LayerModule):
    """Standard LSTM backed by nn.LSTM (cuDNN-optimized)."""

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.lstm = nn.LSTM(input_size, size, batch_first=True)

    def init_state(self, device: torch.device | None = None) -> LayerState:
        h = torch.zeros(self.size, device=device)
        c = torch.zeros(self.size, device=device)
        return (h, c)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        h, c = state
        _, (h_new, c_new) = self.lstm(
            input.flatten().unsqueeze(0).unsqueeze(0),
            (h.unsqueeze(0).unsqueeze(0), c.unsqueeze(0).unsqueeze(0)),
        )
        h_new = h_new.squeeze(0).squeeze(0)
        c_new = c_new.squeeze(0).squeeze(0)
        return (h_new, c_new), h_new

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        output, _ = self.lstm(inputs)
        return output


class LstmDef(LayerDef):
    name = "lstm"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"lstm.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # nn.LSTM: 4 gates * (input_weights + hidden_weights + 2 biases)
        return 4 * self.size * (self.input_size + self.size) + 8 * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> LstmModule:
        module = LstmModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
