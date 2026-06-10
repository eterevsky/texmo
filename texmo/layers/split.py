"""Split layer (Def + Jax): fork-and-merge over N parallel branches.

Each branch is a LayerSeqDef; all branches see the same input and
their outputs are combined channel-wise by `op` in {mul, add, cat}.

Data model uses `branches: list[LayerSeqDef]` from the start so
extending to multi-way (N > 2) later is a validity-rule change,
not a structural rewrite. For now `is_valid` enforces 2-way.

Sequence-length handling: branches may consume different numbers
of positions (e.g. one branch has a `suffix.2`, the other doesn't).
The merge truncates leading positions of the shorter-consuming
branches to align with the longest. The Split's own length
attribute is `max(branch.length for branch in branches)`, so the
enclosing total_padding accounting works the same as for any
other length-aware layer.
"""

import jax
import jax.numpy as jnp

from ..layer import LayerDef, LayerState
from ..layer_jax import LayerJax, LayerWeights
from .seq import LayerSeqDef, LayerSeqJax


_VALID_OPS = ('mul', 'add', 'cat')


# -- Pairwise merge primitives. Element-wise on the overlap; the
#    longer side passes through unchanged for `add` / `mul` and is
#    concatenated for `cat`. These match the legacy `_merge_add` and
#    `_merge_cat` semantics in model_jax.py, so existing skip-form
#    confs preserve loss exactly once the parser translates them.


def _merge_add(v: jax.Array, source: jax.Array) -> jax.Array:
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
    return jnp.concatenate([head, tail], axis=-1)


def _merge_mul(v: jax.Array, source: jax.Array) -> jax.Array:
    # Symmetric to _merge_add but element-wise multiply on the
    # overlap. Padding the smaller vector matches what the user
    # asked for ("start with the current approach of padding the
    # smaller vector"); a stricter equal-channel-size rule can land
    # later in is_valid without touching this code.
    d_v = v.shape[-1]
    d_s = source.shape[-1]
    if d_v == d_s:
        return v * source
    if d_v < d_s:
        head = v * source[..., :d_v]
        tail = source[..., d_v:]
    else:
        head = v[..., :d_s] * source
        tail = v[..., d_s:]
    return jnp.concatenate([head, tail], axis=-1)


def _merge_cat(v: jax.Array, source: jax.Array) -> jax.Array:
    return jnp.concatenate([v, source], axis=-1)


_MERGE_FNS = {
    'add': _merge_add,
    'mul': _merge_mul,
    'cat': _merge_cat,
}


def _merge_all(op: str, outs: list[jax.Array]) -> jax.Array:
    """Left-fold the per-branch outputs with the op's merge function.

    Multi-way generalisation lives here -- for now we only ever get
    two outputs in `outs`, but the same fold trivially handles N.
    """
    fn = _MERGE_FNS[op]
    out = outs[0]
    for branch_out in outs[1:]:
        out = fn(out, branch_out)
    return out


def _merged_size(op: str, branch_sizes: list[int]) -> int:
    if op == 'cat':
        return sum(branch_sizes)
    # add and mul both broadcast to the maximum.
    return max(branch_sizes)


# -- Sequence-length alignment. Every branch sees the same `inputs`
#    with shape (B, T, D); each branch.forward returns
#    (B, T - branch.consumed, D_branch). The least-consuming branch
#    has the longest output; we drop its leading positions to match
#    the most-consuming branch.


def _align_branch_outputs(
    branch_outs: list[jax.Array], branch_consumed: list[int],
) -> list[jax.Array]:
    target = max(branch_consumed)
    return [
        out if c == target else out[:, target - c:, :]
        for out, c in zip(branch_outs, branch_consumed)
    ]


class SplitJax(LayerJax):
    """Runtime counterpart of SplitDef.

    Weights and state are lists -- one entry per branch. JAX walks
    nested lists/dicts as pytrees, so grads / jit / vmap pass
    through without any per-layer plumbing.

    `branch_consumed[i]` is the number of leading sequence positions
    branch `i` consumes (i.e. how many positions shorter its forward
    output is than its input). Computed by `SplitDef` from the
    branches' `LayerSeqDef.length` and passed in here so the runtime
    doesn't have to inspect heterogeneous Jax-side `.length` /
    `.kernel` attributes.
    """

    def __init__(
        self, op: str, branches: list[LayerSeqJax],
        input_size: int, dtype, branch_consumed: list[int],
    ):
        size = _merged_size(op, [b.size for b in branches])
        super().__init__(input_size, size, dtype)
        self.op = op
        self.branches = branches
        self._branch_consumed = branch_consumed

    def init_weights(self, rng: jax.Array) -> list[LayerWeights]:
        if not self.branches:
            return []
        keys = jax.random.split(rng, len(self.branches))
        return [
            branch.init_weights(k)
            for branch, k in zip(self.branches, keys)
        ]

    def init_state(self) -> list[LayerState]:
        return [b.init_state() for b in self.branches]

    def step(
        self, weights: list[LayerWeights],
        state: list[LayerState], x: jax.Array,
    ) -> tuple[list[LayerState], jax.Array]:
        # Each branch sees the same x and updates its own state.
        # Outputs are merged element-wise (no temporal alignment
        # needed -- each step emits exactly one vector per branch).
        new_states: list[LayerState] = []
        outs: list[jax.Array] = []
        for branch, bw, bs in zip(self.branches, weights, state):
            new_bs, branch_out = branch.step(bw, bs, x)
            new_states.append(new_bs)
            outs.append(branch_out)
        return new_states, _merge_all(self.op, outs)

    def forward(
        self, weights: list[LayerWeights], inputs: jax.Array,
    ) -> jax.Array:
        # All branches run on the full `inputs` tensor. Branches
        # that consume more positions produce shorter outputs; we
        # trim the leading positions of the less-consuming ones to
        # align before the channel-axis merge.
        branch_outs = [
            branch.forward(bw, inputs)
            for branch, bw in zip(self.branches, weights)
        ]
        aligned = _align_branch_outputs(
            branch_outs, self._branch_consumed)
        return _merge_all(self.op, aligned)


class SplitDef(LayerDef):
    """Fork-and-merge layer over N (currently 2) parallel branches.

    Spec syntax: `split.{op}({branch_1}, {branch_2})`. Each branch
    is a `LayerSeqDef`; an empty branch is rendered as `pass`. The
    parser doesn't yet know this syntax -- for now SplitDefs are
    constructed in Python directly. The parser change is item 2 of
    the migration TODO.
    """
    name = "split"

    def __init__(
        self, op: str, branches: list[LayerSeqDef], input_size: int,
    ):
        super().__init__(input_size=input_size)
        if op not in _VALID_OPS:
            raise ValueError(f"unknown split op: {op!r}")
        self.op = op
        self.branches = branches
        self.size = _merged_size(op, [b.size for b in branches])
        # All branches see the same input; the split's length is
        # the *longest* per-position reach across branches. The
        # less-consuming branches' leading outputs get dropped at
        # the merge to align time anchors.
        self.length = (
            max(b.length for b in branches) if branches else 1
        )

    def __str__(self) -> str:
        def render_branch(b: LayerSeqDef) -> str:
            s = str(b)
            return s if s else "pass"
        parts = ", ".join(render_branch(b) for b in self.branches)
        return f"split.{self.op}({parts})"

    @property
    def num_weights(self) -> int:
        return sum(b.num_weights for b in self.branches)

    @property
    def num_mults(self) -> int:
        # Branch costs + a small per-merged-element op term. With
        # output shape (T_out, size), the merge adds roughly
        # size * (n_branches - 1) elementwise ops per token. Tiny
        # next to per-layer matmuls; we surface it so the timing
        # model has the right structural feature when we add the
        # split feature extractor.
        merge_cost = self.size * max(0, len(self.branches) - 1)
        return sum(b.num_mults for b in self.branches) + merge_cost

    def is_valid(self) -> bool:
        # 2-way constraint lives here, not in __init__, so multi-way
        # is a one-character relaxation when we're ready. Each
        # branch must also carry the split's input_size (every
        # branch consumes the same input).
        if len(self.branches) != 2:
            return False
        for b in self.branches:
            if not b.is_valid():
                return False
            if b.input_size != self.input_size:
                return False
        # add/mul require an output channel dim to be definable --
        # which the broadcast-friendly merge gives us for any pair
        # of sizes, so no further check is needed today.
        return True

    def build_jax(self, dtype) -> SplitJax:
        return SplitJax(
            self.op,
            [b.build_jax(dtype) for b in self.branches],
            self.input_size, dtype,
            branch_consumed=[b.length - 1 for b in self.branches],
        )
