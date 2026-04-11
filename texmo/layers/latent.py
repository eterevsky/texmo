import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..common import is_power2_int
from ..layer import LayerDef, LayerModule, LayerState


def _normalize(x: Tensor) -> Tensor:
    return F.normalize(x, dim=-1, eps=1e-5)


class LatentModule(LayerModule):
    """Depth-recurrent dense layer (latent reasoning).

    Inspired by https://arxiv.org/abs/2502.05171.

    Each call iterates a shared block `reps` times on a fresh latent state:

        e = Wi @ normalize(x) + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e)   for i = 1..reps
        out = s_reps

    The input contribution `e` is re-injected at every iteration; this is
    what makes the recurrence stable and gives the layer its "iterative
    refinement" character.
    """

    def __init__(self, input_size: int, size: int, reps: int):
        super().__init__()
        self.size = size
        self.reps = reps
        self.wi = nn.Linear(input_size, size)
        self.wr = nn.Linear(size, size, bias=False)
        # Set to True in tests for deterministic output. The paper inits
        # the latent state randomly to encourage path independence; we
        # follow that in both train and eval modes.
        self._deterministic_init = False

    def _init_latent(self, e: Tensor) -> Tensor:
        if self._deterministic_init:
            return torch.zeros_like(e)
        return torch.randn_like(e)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        e = self.wi(_normalize(input))
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return None, s

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        e = self.wi(_normalize(inputs))  # (batch, seq_len, size)
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return s


class LrnnModule(LayerModule):
    """Depth-recurrent RNN: combines time-recurrence with latent reasoning.

    At each timestep, runs `reps` refinement iterations using the previous
    timestep's hidden state as additional context:

        e_t = Wi @ normalize(x_t) + Wh @ normalize(h_{t-1}) + b
        s_0 = 0
        s_i = tanh(Wr @ normalize(s_{i-1}) + e_t)
        h_t = s_reps
    """

    def __init__(self, input_size: int, size: int, reps: int):
        super().__init__()
        self.size = size
        self.reps = reps
        self.wi = nn.Linear(input_size, size)
        self.wh = nn.Linear(size, size, bias=False)
        self.wr = nn.Linear(size, size, bias=False)
        # Set to True in tests for deterministic output. The paper inits
        # the latent state randomly to encourage path independence; we
        # follow that in both train and eval modes.
        self._deterministic_init = False

    def init_state(self, device: torch.device | None = None) -> LayerState:
        return torch.zeros(self.size, device=device)

    def _init_latent(self, e: Tensor) -> Tensor:
        if self._deterministic_init:
            return torch.zeros_like(e)
        return torch.randn_like(e)

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        e = self.wi(_normalize(input)) + self.wh(_normalize(state))
        s = self._init_latent(e)
        for _ in range(self.reps):
            s = torch.tanh(self.wr(_normalize(s)) + e)
        return s, s

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, input_size)
        # Precompute the input contribution for all timesteps at once.
        wi_x = self.wi(_normalize(inputs))  # (batch, seq_len, size)

        batch, seq_len, _ = inputs.shape
        h = torch.zeros(
            batch, self.size, device=inputs.device, dtype=inputs.dtype)
        outputs = []
        for t in range(seq_len):
            e = wi_x[:, t] + self.wh(_normalize(h))
            s = self._init_latent(e)
            for _ in range(self.reps):
                s = torch.tanh(self.wr(_normalize(s)) + e)
            h = s
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class LatentDef(LayerDef):
    name = "latent"

    def __init__(self, size: int, reps: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size
        self.reps = reps

    def __str__(self) -> str:
        return f"latent.{self.size}.{self.reps}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.size) and self.size > 1
            and is_power2_int(self.reps) and self.reps >= 2
        )

    @property
    def num_weights(self) -> int:
        # Linear(input, size) + bias + Linear(size, size, bias=False)
        return self.size * self.input_size + self.size + self.size * self.size

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> LatentModule:
        module = LatentModule(self.input_size, self.size, self.reps)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module


class LrnnDef(LayerDef):
    name = "lrnn"

    def __init__(self, size: int, reps: int, input_size: int):
        super().__init__(input_size=input_size)
        self.size = size
        self.reps = reps

    def __str__(self) -> str:
        return f"lrnn.{self.size}.{self.reps}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.size) and self.size > 1
            and is_power2_int(self.reps) and self.reps >= 2
        )

    @property
    def num_weights(self) -> int:
        # Wi(input,size)+bias + Wh(size,size) + Wr(size,size)
        return (
            self.size * self.input_size + self.size
            + self.size * self.size + self.size * self.size
        )

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> LrnnModule:
        module = LrnnModule(self.input_size, self.size, self.reps)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
