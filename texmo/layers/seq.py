"""Sequence-of-layers container (Def + Jax).

The fundamental building block of Model2's fork-and-merge layer
DAG. A LayerSeqDef holds an ordered list of LayerDefs; its
forward/step iterate through them, threading the activation
between them. Used at two levels: as Model2's top-level layer
chain, and as each branch of a SplitDef. Same class, both
levels -- which is the whole point of inheriting from LayerDef.

A LayerSeqDef directly nested inside another LayerSeqDef would
just be a longer flat sequence with the same semantics; that's
rejected as invalid. Branches of a SplitDef are the only place a
LayerSeqDef is meant to appear inside another layer.
"""

import jax
import jax.numpy as jnp

from ..layer import LayerDef, LayerState
from ..layer_jax import LayerJax, LayerWeights


class LayerSeqJax(LayerJax):
    """Runtime counterpart of LayerSeqDef.

    Weights and state are lists -- one entry per sub-layer. JAX
    treats nested lists/dicts as pytrees automatically, so grads,
    jit, and vmap all pass through cleanly.
    """

    def __init__(
        self, input_size: int, layers: list[LayerJax], dtype,
    ):
        size = layers[-1].size if layers else input_size
        super().__init__(input_size, size, dtype)
        self.layers = layers

    def init_weights(self, rng: jax.Array) -> list[LayerWeights]:
        if not self.layers:
            return []
        keys = jax.random.split(rng, len(self.layers))
        return [
            layer.init_weights(k)
            for layer, k in zip(self.layers, keys)
        ]

    def init_state(self) -> list[LayerState]:
        return [layer.init_state() for layer in self.layers]

    def step(
        self, weights: list[LayerWeights],
        state: list[LayerState], x: jax.Array,
    ) -> tuple[list[LayerState], jax.Array]:
        new_states: list[LayerState] = []
        v = x
        for layer, lw, ls in zip(self.layers, weights, state):
            new_ls, v = layer.step(lw, ls, v)
            new_states.append(new_ls)
        return new_states, v

    def forward(
        self, weights: list[LayerWeights], inputs: jax.Array,
    ) -> jax.Array:
        v = inputs
        for layer, lw in zip(self.layers, weights):
            v = layer.forward(lw, v)
        return v


class LayerSeqDef(LayerDef):
    """Ordered sequence of LayerDefs that behaves itself like a layer.

    `input_size`: input dim to the first sub-layer (or pass-through
    if the sequence is empty).
    `size`: output dim of the last sub-layer (or input_size if empty).
    `length`: 1 + sum(l.length - 1 for l in layers) -- how many input
    positions are needed per output position. Empty seq -> length 1.
    """
    name = "seq"

    def __init__(self, layers: list[LayerDef], input_size: int):
        super().__init__(input_size=input_size)
        self.layers = layers
        if layers:
            self.size = layers[-1].size
            self.length = 1 + sum(l.length - 1 for l in layers)
        else:
            self.size = input_size
            self.length = 1

    def __str__(self) -> str:
        # Spec-string view: just dash-joined child layers. The
        # enclosing context (Model2 spec, or a Split branch) takes
        # care of whatever syntax frames the sequence.
        return "-".join(str(l) for l in self.layers)

    @property
    def num_weights(self) -> int:
        return sum(l.num_weights for l in self.layers)

    @property
    def num_mults(self) -> int:
        return sum(l.num_mults for l in self.layers)

    @property
    def num_layers(self) -> int:
        # Sum of children's num_layers -- the sequence itself doesn't
        # add to the count, only its contents do. By recursion, this
        # walks through Split branches automatically.
        return sum(l.num_layers for l in self.layers)

    def is_valid(self) -> bool:
        # A LayerSeqDef directly inside another LayerSeqDef would
        # just be a longer flat sequence -- reject so the structure
        # stays unambiguous. SplitDef is the only place a
        # LayerSeqDef is meant to appear inside another layer.
        if any(isinstance(l, LayerSeqDef) for l in self.layers):
            return False
        return all(l.is_valid() for l in self.layers)

    def build_jax(self, dtype) -> LayerSeqJax:
        return LayerSeqJax(
            self.input_size,
            [l.build_jax(dtype) for l in self.layers],
            dtype,
        )
