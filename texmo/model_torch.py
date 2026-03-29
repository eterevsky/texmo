import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .layer_torch import LayerDef, LayerModule, LayerState
from .layers.dense_torch import DenseDef, DenseModule
from .layers.input_bytes_torch import InputBytesDef, InputBytesModule

_1_BY_LOG2 = 1.0 / math.log(2.0)


class Model(nn.Module):
    """A runnable model that owns weights."""

    def __init__(
        self,
        input_module: InputBytesModule,
        layer_modules: list[LayerModule],
        output_module: DenseModule,
        ntokens: int,
    ):
        super().__init__()
        self.input_module = input_module
        self.layers = nn.ModuleList(layer_modules)
        self.output_module = output_module
        self.ntokens = ntokens

    @property
    def device(self) -> torch.device:
        return next(self.output_module.parameters()).device

    def initial_step(self) -> tuple[list[LayerState], Tensor]:
        """Predict the first token (before any input).

        Returns:
            (states, logits) where logits is (ntokens,)
        """
        states = []
        # Zero input for the first step
        v = torch.zeros(self.input_module.ntokens,
                        dtype=self.input_module.output_dtype,
                        device=self.device)
        for layer in self.layers:
            state = layer.init_state()
            state, v = layer.step(state, v)
            states.append(state)
        _, logits = self.output_module.step(None, v)
        return states, logits

    def step(
        self, states: list[LayerState], token: int
    ) -> tuple[list[LayerState], Tensor]:
        """Run one step of inference.

        Args:
            states: list of layer states from previous step
            token: input token index

        Returns:
            (new_states, logits) where logits is (ntokens,)
        """
        new_states = []
        v = self.input_module.step(token, device=self.device)
        for layer, state in zip(self.layers, states):
            state, v = layer.step(state, v)
            new_states.append(state)
        _, logits = self.output_module.step(None, v)
        return new_states, logits

    def step_prob(
        self, states: list[LayerState], token: int, temperature: float
    ) -> tuple[list[LayerState], Tensor]:
        """Run one step and return probabilities.

        Returns:
            (new_states, probs) where probs is (ntokens,)
        """
        states, logits = self.step(states, token)
        return states, F.softmax(logits / temperature, dim=-1)

    def step_sample(
        self, states: list[LayerState], token: int, temperature: float
    ) -> tuple[list[LayerState], int]:
        """Run one step and sample from the output distribution.

        Returns:
            (new_states, sampled_token_index)
        """
        states, probs = self.step_prob(states, token, temperature)
        sampled = torch.multinomial(probs, 1).item()
        return states, sampled

    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass on a batch for training.

        Args:
            batch: (batch_size, seq_len) int64 token indices

        Returns:
            logits: (batch_size, seq_len, ntokens)
        """
        # Input: shift right (predict next token)
        v = self.input_module(batch[:, :-1])
        # Prepend zeros for the first position
        batch_size = batch.shape[0]
        pad = torch.zeros(batch_size, 1, self.ntokens,
                          dtype=v.dtype, device=v.device)
        v = torch.cat([pad, v], dim=1)

        for layer in self.layers:
            v = layer(v)

        logits = self.output_module(v)
        return logits

    def loss_batch(self, batch: Tensor) -> Tensor:
        """Compute average cross-entropy loss in bits per token.

        Args:
            batch: (batch_size, seq_len) int64 token indices

        Returns:
            scalar loss in bits per token
        """
        logits = self.forward(batch)
        # logits: (batch, seq_len, ntokens), targets: (batch, seq_len)
        loss = F.cross_entropy(
            logits.reshape(-1, self.ntokens),
            batch.reshape(-1),
        )
        return _1_BY_LOG2 * loss


class ModelDef(object):
    """Lightweight model descriptor. No weights."""

    def __init__(self, spec: str, dtype: torch.dtype = torch.float32):
        self.spec = spec
        self.dtype = dtype

        spec_parts = spec.split("|")
        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise ValueError("Model spec can't contain more than one |")

        assert input_spec == '' or input_spec == 'bytes', \
            f"Only 'bytes' input is supported for now, got '{input_spec}'"

        self.input = InputBytesDef(dtype=dtype)

        self.layers: list[LayerDef] = []
        shape = self.input.output_size
        if layers_spec:
            for layer_spec in layers_spec.split("-"):
                layer = _build_layer_def(layer_spec, shape)
                self.layers.append(layer)
                shape = layer.output_size

        self.ntokens = self.input.ntokens
        self.output = DenseDef(self.ntokens, input_size=shape)

    def __str__(self) -> str:
        return self.spec

    def __eq__(self, other) -> bool:
        return self.spec == other.spec and self.dtype == other.dtype

    def __hash__(self) -> int:
        return hash((self.spec, self.dtype))

    @property
    def num_weights(self) -> int:
        return (
            self.input.num_weights
            + sum(l.num_weights for l in self.layers)
            + self.output.num_weights
        )

    def build_model(self, state_dict: Optional[dict[str, Tensor]] = None) -> Model:
        """Build a runnable Model (nn.Module).

        Args:
            state_dict: if provided, load weights from this dict.

        Returns:
            A Model instance.
        """
        input_module = self.input.build_module()
        layer_modules = [ld.build_module() for ld in self.layers]
        output_module = self.output.build_module()

        model = Model(input_module, layer_modules, output_module, self.ntokens)

        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            # Cast to target dtype (input module handles its own dtype)
            model.to(self.dtype)

        return model


def _build_layer_def(spec: str, input_size: int) -> LayerDef:
    """Parse a layer spec string and return a LayerDef."""
    parts = spec.split(".")
    name = parts[0]

    if name == "dense":
        size = int(parts[1])
        activation = parts[2] if len(parts) > 2 else None
        return DenseDef(
            size,
            relu=(activation == "relu"),
            tanh=(activation == "tanh"),
            gelu=(activation == "gelu"),
            input_size=input_size,
        )

    raise ValueError(f"Unknown layer type: {name}")


_cache: dict[tuple[str, torch.dtype], ModelDef] = {}


def build_model_def(spec: str, dtype: torch.dtype = torch.float32) -> ModelDef:
    key = (spec, dtype)
    model_def = _cache.get(key)
    if model_def is None:
        model_def = ModelDef(spec, dtype)
        _cache[key] = model_def
    return model_def
