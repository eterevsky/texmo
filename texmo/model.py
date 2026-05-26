import math
from itertools import chain
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layer import LayerDef, LayerModule, LayerState
from .layers.dense import DenseDef, DenseModule
from .layers.gru import GruDef, MgruDef, MinGruDef
from .layers.input_bits import InputBitsDef, InputBitsModule
from .layers.input_bytes import InputBytesDef, InputBytesModule
from .layers.latent import LatentDef, LrnnDef
from .layers.lmgu import LmguDef
from .layers.lstm import LstmDef
from .layers.matlstm import MatLstmDef
from .layers.msr import MsrDef
from .layers.mullstm import MulLstmDef
from .layers.slstm import SLstmDef
from .layers.norm import NormDef
from .layers.rnn import RnnDef
from .layers.skip import SkipDef, SkipModule
from .layers.suffix import SuffixDef
from .model_jax import ModelJax
from .precision import Precision

_1_BY_LOG2 = 1.0 / math.log(2.0)


def _apply_merges(shape: int, merges: list[tuple[int, str, int]]) -> int:
    """Apply pending skip merges to `shape` (earliest start first).

    Each merge is (source_size, op, start_pos). For '.add' we take
    max(shape, source_size); for '.cat' we sum.
    """
    for source_size, op, _ in sorted(merges, key=lambda m: m[2]):
        if op == 'add':
            shape = max(shape, source_size)
        else:  # cat
            shape = shape + source_size
    return shape


def _skip_targets(
    layers: list[LayerDef],
) -> dict[int, list[tuple[int, str, int]]]:
    """Build target_position -> [(start_pos, op, shrinkage)] for all skips.

    `shrinkage` is the number of sequence positions that suffix-like
    layers between start and target consume (= sum of length-1). In
    forward mode the source must be sliced by this amount to align
    with the shortened main-path sequence. In step mode it's ignored
    (per-step processing is length-agnostic).

    Each target's list is sorted by start_pos (earliest first).
    """
    targets: dict[int, list[tuple[int, str, int]]] = {}
    for i, layer in enumerate(layers):
        if isinstance(layer, SkipDef):
            target = i + layer.distance + 1
            shrinkage = sum(
                l.length - 1 for l in layers[i + 1:target] if l.length > 1)
            targets.setdefault(target, []).append((i, layer.op, shrinkage))
    for t in targets:
        targets[t].sort()
    return targets


def _insert_with_skip_bumps(
    layers: list[LayerDef], pos: int, new_spec: str
) -> list[str]:
    """Insert new_spec at position pos in the spec strings, bumping the
    distance of any skip that strictly contains pos in its span.

    A skip at index i with distance d spans [i+1, i+d+1). If pos falls
    strictly between i and i+d+1 (i.e. i < pos < i+d+1), we bump the
    distance by 1 so the merge still happens at the same original
    layer. At pos == i+d+1 (exactly the merge point) we don't bump —
    the merge stays before the newly inserted layer.
    """
    result: list[str] = []
    for i, layer in enumerate(layers):
        if i == pos:
            result.append(new_spec)
        if isinstance(layer, SkipDef):
            old_target = i + layer.distance + 1
            if i < pos < old_target:
                result.append(f"skip.{layer.distance + 1}.{layer.op}")
            else:
                result.append(str(layer))
        else:
            result.append(str(layer))
    if pos == len(layers):
        result.append(new_spec)
    return result


def _remove_with_skip_bumps(
    layers: list[LayerDef], pos: int
) -> list[str] | None:
    """Remove layer at `pos`, decrementing distance for skips that
    strictly contained pos. Returns None if any skip would end up with
    distance < 1 (invalid mutation).
    """
    result: list[str] = []
    for i, layer in enumerate(layers):
        if i == pos:
            continue
        if isinstance(layer, SkipDef):
            old_target = i + layer.distance + 1
            if i < pos < old_target:
                new_d = layer.distance - 1
                if new_d < 1:
                    return None
                result.append(f"skip.{new_d}.{layer.op}")
            else:
                result.append(str(layer))
        else:
            result.append(str(layer))
    return result


def _merge_add(v, source):
    # v: (..., D_v); source: (..., D_s). Output shape: (..., max(D_v, D_s)).
    # Sum the overlap, append the tail from whichever is longer.
    d_v = v.shape[-1]
    d_s = source.shape[-1]
    if d_v == d_s:
        return v + source
    if d_v < d_s:
        head = v + source[..., :d_v]
        tail = source[..., d_v:]
    else:
        head = v[..., :d_s] + source
        tail = v[..., d_s:]
    return torch.cat([head, tail], dim=-1)


def _merge_cat(v, source):
    return torch.cat([v, source], dim=-1)


class Model(nn.Module):
    """A runnable model that owns weights."""

    def __init__(
        self,
        input_module: InputBytesModule | InputBitsModule,
        layer_modules: list[LayerModule],
        output_module: DenseModule,
        ntokens: int,
        total_padding: int = 1,
        skip_targets: dict[int, list[tuple[int, str, int]]] | None = None,
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
        # Maps target position -> [(start_pos, op), ...], sorted by start.
        self._skip_targets = skip_targets or {}

    @property
    def device(self) -> torch.device:
        return next(self.output_module.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.output_module.parameters()).dtype

    def _run_pipeline_step(self, v, layer_states):
        """Run v through all layers (with skip save/merge). Returns (v, new_states)."""
        pending: dict[int, Tensor] = {}
        new_states: list[LayerState] = []
        for i, (layer, state) in enumerate(zip(self.layers, layer_states)):
            # Apply any merges targeting this position (skip ends here).
            # In step mode shrinkage is ignored — the saved source is
            # already a single-timestep vector that aligns naturally.
            if i in self._skip_targets:
                for start_pos, op, _ in self._skip_targets[i]:
                    source = pending.pop(start_pos)
                    v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)
            # Save source if this is a skip pseudo-layer.
            if isinstance(layer, SkipModule):
                pending[i] = v
            state, v = layer.step(state, v)
            new_states.append(state)
        # Merges targeting the position after the last layer (into output dense).
        n = len(self.layers)
        if n in self._skip_targets:
            for start_pos, op, _ in self._skip_targets[n]:
                source = pending.pop(start_pos)
                v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)
        return v, new_states

    def initial_step(self) -> tuple[list[LayerState], Tensor]:
        """Predict the first token (before any input).

        Returns:
            (states, logits) where logits is (ntokens,)
            states[0] is the input module state, states[1:] are layer states.
        """
        input_state = self.input_module.init_state()

        layer_states = [
            layer.init_state(device=self.device, dtype=self.dtype)
            for layer in self.layers
        ]

        # Feed total_padding initial vectors to warm up stateful layers
        # (e.g. suffix). Each iteration mirrors one padding position in
        # forward(), cycling through the correct bp positions.
        p = self._total_padding
        v = None
        for i in range(p):
            v = self.input_module.initial_vector(
                device=self.device, position=-p + i)
            v, layer_states = self._run_pipeline_step(v, layer_states)

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
        input_state, v = self.input_module.step(
            input_state, token, device=self.device)
        v, new_layer_states = self._run_pipeline_step(v, states[1:])
        _, logits = self.output_module.step(None, v)
        if self._pad_output:
            logits = F.pad(logits, (0, 1))
        return [input_state] + new_layer_states, logits

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

        pending: dict[int, Tensor] = {}
        for i, layer in enumerate(self.layers):
            if i in self._skip_targets:
                for start_pos, op, shrinkage in self._skip_targets[i]:
                    source = pending.pop(start_pos)
                    if shrinkage > 0:
                        source = source[:, shrinkage:]
                    v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)
            if isinstance(layer, SkipModule):
                pending[i] = v
            v = layer(v)

        n = len(self.layers)
        if n in self._skip_targets:
            for start_pos, op, shrinkage in self._skip_targets[n]:
                source = pending.pop(start_pos)
                if shrinkage > 0:
                    source = source[:, shrinkage:]
                v = _merge_add(v, source) if op == 'add' else _merge_cat(v, source)

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

        spec_parts = spec.split("|")
        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise ValueError("Model spec can't contain more than one |")

        if input_spec == '' or input_spec == 'bytes':
            self.input = InputBytesDef(precision=precision)
        elif input_spec.startswith('bits.'):
            self.input = InputBitsDef.from_spec(input_spec, precision=precision)
        else:
            raise ValueError(f"Unknown input type: '{input_spec}'")

        self.layers: list[LayerDef] = []
        # Map from target position -> list of (source_size, op, start_pos).
        # Populated as we walk the spec; drained as we reach each target.
        pending_merges: dict[int, list[tuple[int, str, int]]] = {}
        shape = self.input.size
        if layers_spec:
            specs = layers_spec.split("-")
            for i, layer_spec in enumerate(specs):
                # Apply merges landing at position i to the shape that
                # feeds into layer i.
                if i in pending_merges:
                    shape = _apply_merges(shape, pending_merges.pop(i))

                layer = _build_layer_def(layer_spec, shape)
                self.layers.append(layer)

                if isinstance(layer, SkipDef):
                    target = i + layer.distance + 1
                    pending_merges.setdefault(target, []).append(
                        (shape, layer.op, i))

                shape = layer.size

            # Merges landing at position N (just before output dense).
            n = len(self.layers)
            if n in pending_merges:
                shape = _apply_merges(shape, pending_merges.pop(n))

        # Any remaining pending merges target positions past the output
        # dense, which means the skip's distance overshoots the pipeline.
        # Record this so is_valid() can reject the spec.
        self._skip_overshoots = bool(pending_merges)

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

    @property
    def num_mults(self) -> int:
        """Multiplications per (batch=1, length=1) token. Sum of
        `num_mults` across input, hidden layers, and output. Used as
        a workload proxy for cross-system throughput comparisons."""
        return (
            self.input.num_mults
            + sum(l.num_mults for l in self.layers)
            + self.output.num_mults
        )

    def is_valid(self) -> bool:
        if not self.input.is_valid():
            return False

        # Norm can't be the first layer.
        if self.layers and self.layers[0].name == "norm":
            return False

        # A skip whose distance went past the output dense is invalid.
        if self._skip_overshoots:
            return False

        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            # Two suffix-like layers can't be one after another.
            if l1.length > 1 and l2.length > 1:
                return False
            # Two normalization layers can't be adjacent.
            if l1.name == "norm" and l2.name == "norm":
                return False
            # Norm can't follow a suffix.
            if l1.name == "suffix" and l2.name == "norm":
                return False
            # Skip pseudo-layers can't be adjacent.
            if l1.name == "skip" and l2.name == "skip":
                return False
            # Skip source can't be right before a norm (layer after
            # the skip pseudo-layer is a norm).
            if l1.name == "skip" and l2.name == "norm":
                return False
            # Two msr layers can't be adjacent — both maintain matrix
            # state and serve the same architectural role.
            if l1.name == "msr" and l2.name == "msr":
                return False

        # Merge point can't be right after a suffix: the layer just
        # before each skip's target position (i.e. the last skipped
        # layer) can't be a suffix.
        for i, layer in enumerate(self.layers):
            if isinstance(layer, SkipDef):
                last_skipped = i + layer.distance
                if last_skipped < len(self.layers):
                    if self.layers[last_skipped].name == "suffix":
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

        # 2. Append a new layer. The new size matches the output layer size,
        # capped by the previous layer's (or input's) size — i.e. the
        # narrowest point of the pipe after the new layer.
        last_output = (
            self.layers[-1].size if self.layers
            else self.input.size
        )
        new_size = min(last_output, self.output.size)
        for name in ("dense", "rnn"):
            for activation in ("relu", "gelu", "tanh"):
                yield _make_spec(chain(
                    layers_str, (f"{name}.{new_size}.{activation}",)))
        for name in ("gru", "mgru", "mingru", "lstm", "slstm", "mullstm"):
            yield _make_spec(chain(
                layers_str, (f"{name}.{new_size}",)))
        # matlstm and msr both need size/dim >= 2.
        if new_size >= 2:
            yield _make_spec(chain(
                layers_str, (f"matlstm.{new_size}",)))
            yield _make_spec(chain(
                layers_str, (f"msr.{new_size}.1",)))

        # 3. Remove last layer (symmetric with append)
        if self.layers:
            prev_output = (
                self.input.size if len(self.layers) == 1
                else self.layers[-2].size
            )
            expected_size = min(prev_output, self.output.size)
            if self.layers[-1].size == expected_size:
                yield _make_spec(layers_str[:-1])

        # 4. Insert suffix.2 between any neighboring layers (bumping
        # skip distances that span the insertion point).
        for i in range(len(layers_str) + 1):
            bumped = _insert_with_skip_bumps(self.layers, i, "suffix.2")
            yield _make_spec(bumped)

        # 5. Remove suffix.2 at any position (symmetric with insert).
        for i, ls in enumerate(layers_str):
            if ls == "suffix.2":
                bumped = _remove_with_skip_bumps(self.layers, i)
                if bumped is not None:
                    yield _make_spec(bumped)

        # 6. Insert norm between existing layers (bumping skip distances).
        for i in range(1, len(layers_str) + 1):
            bumped = _insert_with_skip_bumps(self.layers, i, "norm")
            yield _make_spec(bumped)

        # 7. Remove norm at any position (symmetric with insert).
        for i, ls in enumerate(layers_str):
            if ls == "norm":
                bumped = _remove_with_skip_bumps(self.layers, i)
                if bumped is not None:
                    yield _make_spec(bumped)

        # 8. Add a skip at each valid insertion position. Preferred
        # distance is 1; if the layer at the insertion point is a
        # suffix, bump to 2.
        for i in range(len(layers_str)):
            # Need at least one layer after (for the merge point).
            # Source-before-norm: next layer in new spec (= original
            # layer at i) can't be norm.
            if layers_str[i] == "norm":
                continue
            # Distance: 1 normally, 2 if original layer at i is suffix
            # (merge-right-after-suffix rule).
            distance = 2 if layers_str[i].startswith("suffix") else 1
            if i + distance + 1 > len(layers_str) + 1:
                continue  # merge would overshoot the pipeline
            for op in ("add", "cat"):
                new_spec = f"skip.{distance}.{op}"
                yield _make_spec(chain(
                    layers_str[:i], (new_spec,), layers_str[i:]))

        # 9. Remove a skip (symmetric with add — only distances that
        # add would have produced at this position).
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SkipDef):
                continue
            if i + 1 >= len(self.layers):
                continue  # add never produces a skip with no layer after it
            next_spec = layers_str[i + 1]
            expected = 2 if next_spec.startswith("suffix") else 1
            if layer.distance != expected:
                continue
            yield _make_spec(chain(
                layers_str[:i], layers_str[i + 1:]))

        # 10. Mutate skip distance by ±1 (context-aware: skip over a
        # suffix that would end up right before the merge).
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, SkipDef):
                continue
            for delta in (-1, +1):
                new_d = layer.distance + delta
                if new_d < 1:
                    continue
                new_target = i + new_d + 1
                if new_target > len(self.layers) + 1:
                    continue
                # If the new merge lands right after a suffix, skip one
                # more in the same direction.
                if new_target - 1 < len(self.layers):
                    if self.layers[new_target - 1].name == "suffix":
                        new_d += delta
                        new_target = i + new_d + 1
                        if new_d < 1 or new_target > len(self.layers) + 1:
                            continue
                mutated_layer = f"skip.{new_d}.{layer.op}"
                yield _make_spec(chain(
                    layers_str[:i], (mutated_layer,), layers_str[i + 1:]))

        # 11. Prepend a dense layer. Other appendable layers
        # (rnn/gru/...) stay append-only -- they have their own
        # recurrence and benefit from being first. Dense doesn't, so
        # the search rarely starts with `<input>|dense` and never
        # discovers `<input>|dense-X-...` lead-ins. The new size is
        # `input.size` when the current first stack layer is a
        # passthrough (suffix/skip/norm don't define their own size),
        # otherwise `min(input.size, output.size)` -- the natural
        # narrow point for a compression-style projection.
        if (
            not self.layers
            or self.layers[0].name not in ("suffix", "skip", "norm")
        ):
            prepend_size = min(self.input.size, self.output.size)
        else:
            prepend_size = self.input.size
        for activation in ("tanh", "gelu"):
            bumped = _insert_with_skip_bumps(
                self.layers, 0,
                f"dense.{prepend_size}.{activation}")
            yield _make_spec(bumped)

        # 12. Remove the first layer when it's a prepend-shaped
        # dense (symmetric with section 11). The size and activation
        # must match what section 11 would have produced for the
        # *remaining* layers, otherwise removing it isn't a reachable-
        # by-prepend mutation.
        if self.layers and self.layers[0].name == "dense":
            next_layer = (
                self.layers[1] if len(self.layers) >= 2 else None)
            if (
                next_layer is None
                or next_layer.name not in ("suffix", "skip", "norm")
            ):
                expected_size = min(
                    self.input.size, self.output.size)
            else:
                expected_size = self.input.size
            first = self.layers[0]
            if (
                first.size == expected_size
                and first._activation in ("tanh", "gelu")
            ):
                bumped = _remove_with_skip_bumps(self.layers, 0)
                if bumped is not None:
                    yield _make_spec(bumped)

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
            model = build_model_def(spec, precision=self.precision)
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
                      self.ntokens, self.total_padding,
                      skip_targets=_skip_targets(self.layers))

        if state_dict is not None:
            model.load_state_dict(state_dict)
        else:
            # Cast to target dtype (input module handles its own dtype)
            model.to(self.precision.dtype)

        return model

    def build_jax(self) -> ModelJax:
        dtype = self.precision.jax_dtype
        input_layer = self.input.build_jax()
        layers = [ld.build_jax(dtype) for ld in self.layers]
        output = self.output.build_jax(dtype)
        return ModelJax(input_layer, layers, output,
                        self.ntokens, self.total_padding,
                        skip_targets=_skip_targets(self.layers))


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

    if name == "gru":
        return GruDef(int(parts[1]), input_size=input_size)

    if name == "mgru":
        return MgruDef(int(parts[1]), input_size=input_size)

    if name == "mingru":
        return MinGruDef(int(parts[1]), input_size=input_size)

    if name == "lstm":
        return LstmDef(int(parts[1]), input_size=input_size)

    if name == "matlstm":
        return MatLstmDef(int(parts[1]), input_size=input_size)

    if name == "mullstm":
        return MulLstmDef(int(parts[1]), input_size=input_size)

    if name == "slstm":
        return SLstmDef(int(parts[1]), input_size=input_size)

    if name == "msr":
        return MsrDef(
            int(parts[1]), int(parts[2]), input_size=input_size)

    if name == "norm":
        return NormDef(input_size=input_size)

    if name == "latent":
        return LatentDef(
            int(parts[1]), int(parts[2]), input_size=input_size)

    if name == "lrnn":
        return LrnnDef(
            int(parts[1]), int(parts[2]), input_size=input_size)

    if name == "lmgu":
        return LmguDef(
            int(parts[1]), int(parts[2]), input_size=input_size)

    if name == "suffix":
        length = int(parts[1])
        return SuffixDef(length, input_size=input_size)

    if name == "skip":
        distance = int(parts[1])
        op = parts[2]
        if op not in ('add', 'cat'):
            raise ValueError(f"Unknown skip op: {op}")
        return SkipDef(distance, op, input_size=input_size)

    raise ValueError(f"Unknown layer type: {name}")


def build_model_def(spec: str, precision: Precision) -> ModelDef:
    return ModelDef(spec, precision)
