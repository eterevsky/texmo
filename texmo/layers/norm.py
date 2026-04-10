import torch
import torch.nn.functional as F
from torch import Tensor

from ..layer import LayerDef, LayerModule, LayerState


class NormModule(LayerModule):
    """L2 normalization along the feature dimension.

    out = x / max(||x||_2, eps)
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def step(
        self, state: LayerState, input: Tensor
    ) -> tuple[LayerState, Tensor]:
        out = F.normalize(input, dim=-1, eps=1e-5)
        return None, out

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (batch, seq_len, size)
        return F.normalize(inputs, dim=-1, eps=1e-5)


class NormDef(LayerDef):
    name = "norm"

    def __init__(self, input_size: int):
        super().__init__(input_size=input_size)
        self.size = input_size

    def __str__(self) -> str:
        return "norm"

    def is_valid(self) -> bool:
        return self.input_size > 1

    @property
    def num_weights(self) -> int:
        return 0

    def build_module(
        self, state_dict: dict[str, Tensor] | None = None
    ) -> NormModule:
        module = NormModule(self.input_size)
        if state_dict is not None:
            module.load_state_dict(state_dict)
        return module
