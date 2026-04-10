import torch
import torch.nn as nn
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerModule, LayerState


class GruModule(LayerModule):
    """Standard GRU backed by nn.GRU (cuDNN-optimized)."""

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.gru = nn.GRU(input_size, size, batch_first=True)

    def init_state(self, device: torch.device | None = None) -> LayerState:
        return torch.zeros(self.size, device=device)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        _, h = self.gru(
            input.flatten().unsqueeze(0).unsqueeze(0),
            state.unsqueeze(0).unsqueeze(0),
        )
        new_state = h.squeeze(0).squeeze(0)
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        output, _ = self.gru(inputs)
        return output


class MgruModule(LayerModule):
    """GRU variant with a single gate (update = 1 - forget).

        f = sigmoid(Wf @ [x, h] + bf)
        hc = tanh(Wh @ [x, f*h] + bh)
        h = (1-f) * h + f * hc
    """

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        # Split the input and hidden contributions so we can precompute the
        # input part for all timesteps at once in forward(). Biases live on
        # the input-side linears to avoid double-counting.
        self.wf_x = nn.Linear(input_size, size)
        self.wf_h = nn.Linear(size, size, bias=False)
        self.wh_x = nn.Linear(input_size, size)
        self.wh_h = nn.Linear(size, size, bias=False)

    def init_state(self, device: torch.device | None = None) -> LayerState:
        return torch.zeros(self.size, device=device)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        x = input.flatten()
        f = torch.sigmoid(self.wf_x(x) + self.wf_h(state))
        hc = torch.tanh(self.wh_x(x) + self.wh_h(f * state))
        new_state = (1 - f) * state + f * hc
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        # Precompute input contributions for all timesteps at once. Only the
        # hidden-state-dependent part stays inside the loop.
        f_x = self.wf_x(inputs)  # (batch, seq_len, size)
        h_x = self.wh_x(inputs)  # (batch, seq_len, size)

        batch, seq_len, _ = inputs.shape
        state = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            f = torch.sigmoid(f_x[:, t] + self.wf_h(state))
            hc = torch.tanh(h_x[:, t] + self.wh_h(f * state))
            state = (1 - f) * state + f * hc
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class MinGruModule(LayerModule):
    """Minimal GRU where the gate and candidate depend only on the input.

    From https://arxiv.org/abs/2305.17473

        zh = W @ x + b      (shape: 2*size)
        z = sigmoid(zh[:size])
        h_new = zh[size:]
        h = (1-z) * h + z * h_new
    """

    def __init__(self, input_size: int, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(input_size, 2 * size)

    def init_state(self, device: torch.device | None = None) -> LayerState:
        return torch.zeros(self.size, device=device)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        zh = self.linear(input.flatten())
        z = torch.sigmoid(zh[:self.size])
        h_new = zh[self.size:]
        new_state = (1 - z) * state + z * h_new
        return new_state, new_state

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        batch, seq_len, _ = inputs.shape
        # Compute z and h_new for all timesteps at once
        zh = self.linear(inputs)  # (batch, seq_len, 2*size)
        z = torch.sigmoid(zh[..., :self.size])
        h_new = zh[..., self.size:]

        state = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            state = (1 - z[:, t]) * state + z[:, t] * h_new[:, t]
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class GruDef(LayerDef):
    name = "gru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"gru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # nn.GRU: 3 gates * (input_weights + hidden_weights + 2 biases)
        return 3 * self.size * (self.input_size + self.size) + 6 * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> GruModule:
        module = GruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module


class MgruDef(LayerDef):
    name = "mgru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"mgru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # 2 linear layers, each with (input+size) inputs and size outputs
        return 2 * (self.size * (self.input_size + self.size) + self.size)

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> MgruModule:
        module = MgruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module


class MinGruDef(LayerDef):
    name = "mingru"

    def __init__(self, size: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size

    def __str__(self) -> str:
        return f"mingru.{self.size}"

    def is_valid(self) -> bool:
        return is_power2_int(self.size)

    @property
    def num_weights(self) -> int:
        # nn.Linear(input_size, 2*size): 2*size*input_size + 2*size
        return 2 * self.size * self.input_size + 2 * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> MinGruModule:
        module = MinGruModule(self.input_size, self.size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
