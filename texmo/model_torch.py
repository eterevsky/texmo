import math
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Optional

from .layer_torch import LayerDef, LayerModule, LayerState
from .precision import Precision
from .layers.dense_torch import DenseDef, DenseModule
from .layers.input_bits_torch import InputBitsDef, InputBitsModule
from .layers.input_bytes_torch import InputBytesDef, InputBytesModule
from .layers.rnn_torch import RnnDef
from .layers.suffix_torch import SuffixDef

_1_BY_LOG2 = 1.0 / math.log(2.0)


class Model(nn.Module):
    """A runnable model that owns weights."""

    def __init__(
        self,
        input_module: InputBytesModule | InputBitsModule,
        layer_modules: list[LayerModule],
        output_module: DenseModule,
        ntokens: int,
        total_padding: int = 1,
    ):
        super().__init__()
        self.input_module = input_module
        self.layers = nn.ModuleList(layer_modules)
        self.output_module = output_module
        self.ntokens = ntokens
        self._total_padding = total_padding
        # For binary tokens (bits.1), the output layer produces a single logit.
        # Padding with 0 gives softmax([x, 0]) = [sigmoid(-x), sigmoid(x)],
        # so the single logit acts as the log-odds of class 1 vs class 0.
        self._pad_output = ntokens <= 2

    @property
    def device(self) -> torch.device:
        return next(self.output_module.parameters()).device

    def initial_step(self) -> tuple[list[LayerState], Tensor]:
        """Predict the first token (before any input).

        Returns:
            (states, logits) where logits is (ntokens,)
            states[0] is the input module state, states[1:] are layer states.
        """
        input_state = self.input_module.init_state()

        # Initialize layer states
        layer_states = [layer.init_state(device=self.device)
                        for layer in self.layers]

        # Feed total_padding initial vectors to warm up stateful layers
        # (e.g. suffix). Each iteration mirrors one padding position in
        # forward(), cycling through the correct bp positions.
        p = self._total_padding
        for i in range(p):
            v = self.input_module.initial_vector(
                device=self.device, position=-p + i)
            for j, layer in enumerate(self.layers):
                layer_states[j], v = layer.step(layer_states[j], v)

        states = [input_state] + layer_states
        _, logits = self.output_module.step(None, v)
        if self._pad_output:
            logits = F.pad(logits, (0, 1))
        return states, logits

    def step(
        self, states: list[LayerState], token: int
    ) -> tuple[list[LayerState], Tensor]:
        """Run one step of inference.

        Args:
            states: list of states; states[0] is input state,
                    states[1:] are layer states
            token: input token index

        Returns:
            (new_states, logits) where logits is (ntokens,)
        """
        input_state = states[0]
        new_states = []
        input_state, v = self.input_module.step(
            input_state, token, device=self.device)
        new_states.append(input_state)
        for layer, state in zip(self.layers, states[1:]):
            state, v = layer.step(state, v)
            new_states.append(state)
        _, logits = self.output_module.step(None, v)
        if self._pad_output:
            logits = F.pad(logits, (0, 1))
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
        # Input: shift right (predict next token). Extra padding beyond 1
        # is consumed by suffix-like layers that look back multiple steps.
        v = self.input_module(batch[:, :-1], padding=self._total_padding)

        for layer in self.layers:
            v = layer(v)

        logits = self.output_module(v)
        if self._pad_output:
            logits = F.pad(logits, (0, 1))
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

    def __init__(self, spec: str, precision: Precision):
        self.spec = spec
        self.precision = precision

        dtype = precision.dtype
        spec_parts = spec.split("|")
        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise ValueError("Model spec can't contain more than one |")

        if input_spec == '' or input_spec == 'bytes':
            self.input = InputBytesDef(dtype=dtype)
        elif input_spec.startswith('bits.'):
            self.input = InputBitsDef.from_spec(input_spec, dtype=dtype)
        else:
            raise ValueError(f"Unknown input type: '{input_spec}'")

        self.layers: list[LayerDef] = []
        shape = self.input.size
        if layers_spec:
            for layer_spec in layers_spec.split("-"):
                layer = _build_layer_def(layer_spec, shape)
                self.layers.append(layer)
                shape = layer.size

        self.ntokens = self.input.ntokens
        # 1 for the initial "no input" position, plus extra for layers
        # that look back multiple steps (e.g. suffix).
        self.total_padding = 1 + sum(l.length - 1 for l in self.layers)
        output_size = self.ntokens if self.ntokens > 2 else 1
        self.output = DenseDef(output_size, input_size=shape)

    def __str__(self) -> str:
        return self.spec

    def __eq__(self, other) -> bool:
        return self.spec == other.spec and self.precision == other.precision

    def __hash__(self) -> int:
        return hash((self.spec, self.precision))

    @property
    def num_weights(self) -> int:
        return (
            self.input.num_weights
            + sum(l.num_weights for l in self.layers)
            + self.output.num_weights
        )

    def is_valid(self) -> bool:
        if not self.input.is_valid():
            return False

        # Two suffix-like layers can't be one after another.
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            if l1.length > 1 and l2.length > 1:
                return False

        return all(l.is_valid() for l in self.layers)

    def _gen_neighbor_specs(self) -> Iterable[str]:
        """Yield spec strings for all single-mutation neighbors."""
        layers_str = [str(l) for l in self.layers]
        input_spec = str(self.input)

        layers_joined = "-".join(layers_str)

        def _make_spec(ls):
            return input_spec + "|" + "-".join(ls)

        # 0. Mutate input layer
        for input_neighbor in self.input.neighbors():
            yield input_neighbor + "|" + layers_joined

        # 1. Mutate each layer (size 2x, type swap — via LayerDef.neighbors)
        for i in range(len(layers_str)):
            for layer_neighbor in self.layers[i].neighbors():
                yield _make_spec(chain(
                    layers_str[:i], (layer_neighbor,), layers_str[i + 1:]))

        # 2. Append a new layer
        last_output = (
            self.layers[-1].size if self.layers
            else self.input.size
        )
        new_size = min(last_output, self.ntokens)
        for name in ("dense", "rnn"):
            for activation in ("relu", "gelu", "tanh"):
                yield _make_spec(chain(
                    layers_str, (f"{name}.{new_size}.{activation}",)))

        # 3. Remove last layer (symmetric with append)
        if self.layers:
            prev_output = (
                self.input.size if len(self.layers) == 1
                else self.layers[-2].size
            )
            expected_size = min(prev_output, self.ntokens)
            if self.layers[-1].size == expected_size:
                yield _make_spec(layers_str[:-1])

        # 4. Insert suffix.2 between any neighboring layers
        for i in range(len(layers_str) + 1):
            yield _make_spec(chain(
                layers_str[:i], ("suffix.2",), layers_str[i:]))

        # 5. Remove suffix.2 at any position (symmetric with insert)
        for i, ls in enumerate(layers_str):
            if ls == "suffix.2":
                yield _make_spec(chain(
                    layers_str[:i], layers_str[i + 1:]))

    def neighbors(self) -> Iterable['ModelDef']:
        # Precision neighbors
        for p in self.precision.neighbors:
            yield build_model_def(self.spec, precision=p)

        # Architecture neighbors
        seen = set()
        for spec in self._gen_neighbor_specs():
            if spec in seen:
                continue
            seen.add(spec)
            try:
                model = build_model_def(spec, precision=self.precision)
            except (ValueError, KeyError):
                continue
            if model.is_valid() and model != self:
                yield model

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

        model = Model(input_module, layer_modules, output_module,
                      self.ntokens, self.total_padding)

        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            # Cast to target dtype (input module handles its own dtype)
            model.to(self.precision.dtype)

        return model


def _build_layer_def(spec: str, input_size: int) -> LayerDef:
    """Parse a layer spec string and return a LayerDef."""
    parts = spec.split(".")
    name = parts[0]

    if name == "dense":
        size = int(parts[1])
        activation = parts[2] if len(parts) > 2 else None
        return DenseDef(size, activation=activation, input_size=input_size)

    if name == "rnn":
        size = int(parts[1])
        activation = parts[2] if len(parts) > 2 else None
        return RnnDef(size, activation=activation, input_size=input_size)

    if name == "suffix":
        length = int(parts[1])
        return SuffixDef(length, input_size=input_size)

    raise ValueError(f"Unknown layer type: {name}")


_cache: dict[tuple[str, Precision], ModelDef] = {}


def build_model_def(spec: str, precision: Precision) -> ModelDef:
    key = (spec, precision)
    model_def = _cache.get(key)
    if model_def is None:
        model_def = ModelDef(spec, precision)
        _cache[key] = model_def
    return model_def
