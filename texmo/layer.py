from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .common import power2_neighbors
from .layer_jax import LayerJax

LayerState = Any


class LayerModule(nn.Module):
    """Base class for layer nn.Modules that own weights.

    Produced by LayerDef.create_module().
    """

    def init_state(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> LayerState:
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

    @property
    def num_mults(self) -> int:
        """Multiplications per (batch=1, length=1) token through one
        forward pass. Used as a workload proxy — `num_weights` is a
        good first approximation (each non-bias weight does one mult
        per token; the bias adds count as 'honorary' mults). Layers
        that reuse the same weight matrix multiple times per token
        (latent, lrnn) override this."""
        return self.num_weights

    @property
    def num_layers(self) -> int:
        """Count of layers contributed by this LayerDef. Leaf layers
        report 1; structural layers (LayerSeqDef holds N children;
        SplitDef holds itself plus its branches) override to recurse
        through their contents so the total matches `len(layers)` on
        the equivalent flat-list ModelDef representation."""
        return 1

    def is_valid(self) -> bool:
        raise NotImplementedError

    def neighbors(self):
        """Yield spec strings for single-mutation neighbors.

        Size/length 2x changes and swaps between related layer types:
            dense <-> rnn (with activation preserved)
            rnn <-> {gru, mgru, mingru} (activation dropped/picked)
            gru, mgru, mingru, lstm are mutual neighbors (except
            lstm <-> mingru and lstm <-> rnn, which are not)
            dense.X.tanh <-> latent.X.2
            rnn.X.tanh <-> lrnn.X.2
            latent.X.Y <-> lrnn.X.Y (and size/reps 2x mutations)
        """
        if self.name == "suffix":
            for l in power2_neighbors(self.length):
                yield f"suffix.{l}"
            # Type swap: suffix.L <-> conv.L (same time-axis width;
            # suffix downsamples by L, conv preserves length).
            yield f"conv.{self.length}"
            return

        if self.name == "conv":
            for l in power2_neighbors(self.kernel):
                yield f"conv.{l}"
            # Type swap: conv.L <-> suffix.L (mirror of the suffix
            # case above).
            yield f"suffix.{self.kernel}"
            return

        if self.name in ("norm", "rmsnorm"):
            # Swap between the parameter-free L2 norm and the learned-
            # scale RMS norm; both are size-preserving with no other
            # metaparameter.
            yield "rmsnorm" if self.name == "norm" else "norm"
            return

        if self.name in ("latent", "lrnn", "lmgu"):
            # Size 2x (keep > 1)
            for s in power2_neighbors(self.size):
                if s > 1:
                    yield f"{self.name}.{s}.{self.reps}"
            # Reps 2x (keep >= 2)
            for r in power2_neighbors(self.reps):
                if r >= 2:
                    yield f"{self.name}.{self.size}.{r}"
            # Swaps among the depth-recurrent family at same (S, R).
            if self.name == "latent":
                yield f"lrnn.{self.size}.{self.reps}"
            elif self.name == "lrnn":
                yield f"latent.{self.size}.{self.reps}"
                yield f"lmgu.{self.size}.{self.reps}"
            elif self.name == "lmgu":
                yield f"lrnn.{self.size}.{self.reps}"
            # Collapse to non-iterating cousin at reps == 2.
            if self.reps == 2:
                if self.name == "latent":
                    yield f"dense.{self.size}.tanh"
                elif self.name == "lrnn":
                    yield f"rnn.{self.size}.tanh"
                elif self.name == "lmgu":
                    yield f"mgru.{self.size}"
            return

        if self.name == "msr":
            # 2x dim (must stay >= 2 — RoPE needs at least one pair).
            for d in power2_neighbors(self.dim):
                if d >= 2:
                    yield f"msr.{d}.{self.heads}"
            # 2x heads (must stay >= 1).
            for h in power2_neighbors(self.heads):
                yield f"msr.{self.dim}.{h}"
            # Single-head msr swaps with mgru of the same size.
            if self.heads == 1:
                yield f"mgru.{self.dim}"
            return

        if self.name == "rglru":
            # The only metaparameter is the block count (power of 2);
            # is_valid filters candidates that don't divide the width.
            for b in power2_neighbors(self.blocks):
                yield f"rglru.{b}"
            # Size-preserving swap with the minimal input-gated cells, at
            # the canonical single-block form: rglru.1 <-> mgru.X /
            # mingru.X (which preserve size when X == input). rglru.size
            # == input_size by construction, so the targets are size-safe.
            if self.blocks == 1:
                yield f"mgru.{self.size}"
                yield f"mingru.{self.size}"
            return

        # Size mutations (keep same layer type and activation)
        _LSTM_FAMILY = ("lstm", "matlstm", "slstm", "mullstm")
        if self.name in ("dense", "rnn"):
            for s in power2_neighbors(self.size):
                yield f"{self.name}.{s}.{self._activation}"
        elif self.name in ("gru", "mgru", "mingru", "lstm", "slstm", "mullstm"):
            for s in power2_neighbors(self.size):
                yield f"{self.name}.{s}"
        elif self.name == "matlstm":
            # matlstm needs size >= 2 (matrix cell isn't meaningful at D=1).
            for s in power2_neighbors(self.size):
                if s >= 2:
                    yield f"{self.name}.{s}"

        # Type swaps
        _RECURRENT = ("gru", "mgru", "mingru")
        if self.name == "dense":
            yield f"rnn.{self.size}.{self._activation}"
            if self._activation == "tanh" and self.size > 1:
                yield f"latent.{self.size}.2"
        elif self.name == "rnn":
            yield f"dense.{self.size}.{self._activation}"
            for other in _RECURRENT:
                yield f"{other}.{self.size}"
            if self._activation == "tanh" and self.size > 1:
                yield f"lrnn.{self.size}.2"
        elif self.name in _RECURRENT:
            for act in ("relu", "gelu", "tanh"):
                yield f"rnn.{self.size}.{act}"
            for other in _RECURRENT:
                if other != self.name:
                    yield f"{other}.{self.size}"
            # lstm <-> gru, mgru (but not mingru)
            if self.name in ("gru", "mgru"):
                yield f"lstm.{self.size}"
            # mgru <-> msr.X.1 (single-head retention has the same
            # vector state width as mgru.X). msr needs dim >= 2 for
            # at least one RoPE pair, so skip when mgru.1.
            if self.name == "mgru" and self.size >= 2:
                yield f"msr.{self.size}.1"
            # mgru <-> lmgu.X.2 (the depth-recurrent variant collapses
            # to mgru at reps=2). lmgu needs size > 1.
            if self.name == "mgru" and self.size > 1:
                yield f"lmgu.{self.size}.2"
            # mgru.X / mingru.X <-> rglru.1: the RG-LRU is size-preserving,
            # so the swap is only size-safe when this cell preserves size
            # (size == input). Lands on the canonical single-block form.
            if (self.name in ("mgru", "mingru")
                    and self.size == self.input_size):
                yield "rglru.1"
        elif self.name in _LSTM_FAMILY:
            # All-to-all within the LSTM family (lstm/matlstm/slstm/
            # mullstm). matlstm has size >= 2 constraint; skip the
            # swap target when current size is 1.
            for other in _LSTM_FAMILY:
                if other == self.name:
                    continue
                if other == "matlstm" and self.size < 2:
                    continue
                yield f"{other}.{self.size}"
            # lstm also keeps its existing swaps with gru/mgru.
            if self.name == "lstm":
                yield f"gru.{self.size}"
                yield f"mgru.{self.size}"

    def build_module(self, state_dict: Optional[dict[str, Tensor]] = None) -> LayerModule:
        """Create an nn.Module for this layer.

        Args:
            state_dict: if provided, load these weights instead of
                initializing fresh ones.

        Returns:
            A LayerModule owning its weights.
        """
        raise NotImplementedError

    def build_jax(self, dtype) -> LayerJax:
        """Create a JAX layer implementation."""
        raise NotImplementedError
