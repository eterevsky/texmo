"""Skip (residual) pseudo-layer.

See docs/skip.md for the full design. The SkipModule / SkipJax classes
are essentially no-ops; all the save/merge logic lives in Model /
ModelJax.
"""

import jax
import torch.nn as nn

from ..layer import LayerDef, LayerJax, LayerModule, LayerState
from ..layer_jax import LayerWeights


class SkipModule(LayerModule):
    """Placeholder. Skip logic lives in Model.forward / Model.step."""

    def __init__(self, size: int, distance: int, op: str):
        super().__init__()
        self.size = size
        self.distance = distance
        self.op = op

    def init_state(self, device=None, dtype=None) -> LayerState:
        return None

    def step(self, state: LayerState, input):
        return None, input

    def forward(self, inputs):
        return inputs


class SkipJax(LayerJax):
    """Placeholder. Skip logic lives in ModelJax.forward / ModelJax.step."""

    def __init__(self, input_size: int, distance: int, op: str, dtype):
        super().__init__(input_size=input_size, size=input_size, dtype=dtype)
        self.distance = distance
        self.op = op

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        return None, x

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        return inputs


class SkipDef(LayerDef):
    name = "skip"

    def __init__(self, distance: int, op: str, input_size: int):
        super().__init__(input_size=input_size)
        self.distance = distance
        self.op = op
        # Skip is a pass-through; output size equals input size. The
        # merge later may change the size, but that's handled at the
        # merge point, not here.
        self.size = input_size

    def __str__(self) -> str:
        return f"skip.{self.distance}.{self.op}"

    def is_valid(self) -> bool:
        # Context-free checks only. Pipeline-level checks (adjacency,
        # bounds, source-before-norm, merge-after-suffix) live in
        # ModelDef.is_valid.
        return self.distance >= 1 and self.op in ('add', 'cat')

    @property
    def num_weights(self) -> int:
        return 0

    def neighbors(self):
        # Only the context-free add ↔ cat swap. Distance mutations
        # need the full pipeline context and live in ModelDef.
        other_op = 'cat' if self.op == 'add' else 'add'
        yield f"skip.{self.distance}.{other_op}"

    def build_module(self, state_dict=None) -> SkipModule:
        return SkipModule(self.size, self.distance, self.op)

    def build_jax(self, dtype) -> SkipJax:
        return SkipJax(self.input_size, self.distance, self.op, dtype)
