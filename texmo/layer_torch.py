from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .common import power2_neighbors


LayerState = Any


class LayerModule(nn.Module):
    """Base class for layer nn.Modules that own weights.

    Produced by LayerDef.create_module().
    """

    def init_state(self, device: torch.device | None = None) -> LayerState:
        """Initialize recurrent state for a single sample.

        Returns None for stateless layers (e.g. dense).
        """
        return None

    def step(
        self, state: LayerState, input: torch.Tensor
    ) -> tuple[LayerState, torch.Tensor]:
        """Single timestep forward on a single input.

        Args:
            state: recurrent state, or None for stateless layers
            input: (input_size,)

        Returns:
            (new_state, output) where output is (size,)
        """
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Full-sequence forward on a batch.

        Args:
            inputs: (batch, seq_len, input_size)

        Returns:
            (batch, seq_len, size)
        """
        raise NotImplementedError


class LayerDef(object):
    """Descriptor for a layer. Lightweight, no weights."""
    name = "OVERRIDE"

    def __init__(self, input_size: int):
        self.input_size: int = input_size
        self.size: Optional[int] = None
        # The number of the output steps from the previous layers on which
        # this layer depends. Should be 1 for dense and various recursive
        # layers, and length of the suffix for suffix and attn layers.
        self.length: int = 1

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def num_weights(self) -> int:
        raise NotImplementedError

    def is_valid(self) -> bool:
        raise NotImplementedError

    def neighbors(self):
        """Yield spec strings for single-mutation neighbors.

        Includes size/length 2x changes and dense<->rnn type swaps.
        """
        if self.name in ("dense", "rnn"):
            for s in power2_neighbors(self.size):
                yield f"{self.name}.{s}.{self._activation}"
            # Type swap
            other = "rnn" if self.name == "dense" else "dense"
            yield f"{other}.{self.size}.{self._activation}"
        elif self.name == "suffix":
            for l in power2_neighbors(self.length):
                yield f"suffix.{l}"

    def build_module(self, state_dict: Optional[dict[str, Tensor]] = None) -> LayerModule:
        """Create an nn.Module for this layer.

        Args:
            state_dict: if provided, load these weights instead of
                initializing fresh ones.

        Returns:
            A LayerModule owning its weights.
        """
        raise NotImplementedError
