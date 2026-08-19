"""Model2: the layer-DAG model representation (successor to the
retired flat ModelDef).

The layer chain is a `LayerSeqDef` rather than a flat
`list[LayerDef]`. SplitDef plugs into this by holding two
LayerSeqDefs as its branches.

Construction goes through `spec_parser.parse_model2(spec,
precision)`: that's where every string-to-tree decision lives,
including the `|`-split, the input-layer dispatch, and the
layer-list grammar. `Model2Def.__init__` only takes already-built
pieces and stores them.
"""

from collections.abc import Iterable
from typing import Self

from .common import floor_power2
from .layer import LayerDef, _ATTN_SWAP_WINDOW
from .layers.dense import DenseDef
from .layers.embedding_codec import EmbeddingCodecDef, TiedHead
from .layers.one_hot_codec import OneHotCodecDef
from .layers.seq import LayerSeqDef
from .layers.split import SplitDef
from .model2_jax import Model2Jax
from .precision import Precision
from .spec_parser import parse_model2

# The two codec implementations share one API (layers/codec.py).
CodecDef = OneHotCodecDef | EmbeddingCodecDef


class Model2Def:
    """Model spec descriptor built on LayerSeqDef.

    Stores already-parsed pieces (the `codec` owning both model ends
    and `layer_seq`) plus the original spec string for round-tripping.
    Use `spec_parser.parse_model2(spec, precision)` to construct from
    a string.

    Mirrors ModelDef's public surface (spec / num_weights /
    num_mults / is_valid / build_jax / equality / hashing). The
    `input` / `output` properties delegate into the codec so existing
    consumers (managers, predictors, neighbor generation) are
    unaffected: `input` is the codec itself, which exposes the input
    metadata surface (size / ntokens / tokens_name / neighbors /
    the pre-`|` spec as str()), and `output` is its dense head.
    """

    def __init__(
        self,
        *,
        spec: str,
        precision: Precision,
        codec: CodecDef,
        layer_seq: LayerSeqDef,
    ):
        self.spec = spec
        self.precision = precision
        self.codec = codec
        self.layer_seq = layer_seq
        self.ntokens = codec.ntokens
        # 1 for the autoregressive shift + extras consumed by the
        # layer chain. layer_seq.length is already 1 + sum(extras),
        # so it doubles as the model's total padding requirement.
        self.total_padding = layer_seq.length

    @property
    def input(self) -> CodecDef:
        return self.codec

    @property
    def output(self) -> DenseDef | TiedHead:
        return self.codec.head

    def __str__(self) -> str:
        return self.spec

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Model2Def)
            and self.spec == other.spec
            and self.precision == other.precision
        )

    def __hash__(self) -> int:
        return hash((self.spec, self.precision))

    @property
    def num_weights(self) -> int:
        return self.codec.num_weights + self.layer_seq.num_weights

    @property
    def num_mults(self) -> int:
        return self.codec.num_mults + self.layer_seq.num_mults

    @property
    def num_layers(self) -> int:
        """Hidden-layer count. Recursive: a Split counts as itself
        plus the layers inside its branches. For any skip spec that
        the parser translated to split form, the count matches the
        legacy ModelDef.num_layers (which counted the SkipDef
        pseudo-layer as 1 and each in-skip layer as 1)."""
        return self.layer_seq.num_layers

    def is_valid(self) -> bool:
        # The codec validates its input encoding; the head is valid
        # by construction (bare linear readout, like the old implicit
        # output dense). A trailing bare dense is legal exactly for
        # the tied codec, where it is the explicit adapter rather
        # than a projection the head would absorb.
        return self.codec.is_valid() and self.layer_seq.is_valid(
            bare_tail_ok=isinstance(self.codec, EmbeddingCodecDef))

    def _gen_neighbor_specs(self) -> Iterable[str]:
        """Yield raw spec strings for single-mutation neighbors."""
        input_spec = str(self.input)
        top = self.layer_seq.layers
        top_str = "-".join(_strs(top))
        # Input-layer mutations (keep the layer chain fixed). The
        # chain's final width rides along so oh -> emb mode swaps can
        # spell the slaved table width.
        for in_nb in self.input.neighbors(self.output.input_size):
            yield f"{in_nb}|{top_str}"
        # Layer-chain mutations (recurse into the tree).
        for variant in _seq_variants(top, self.input.size):
            yield f"{input_spec}|" + "-".join(variant)

    def neighbors(self) -> Iterable[Self]:
        """Yield single-mutation neighbor models (precision + arch).

        Mirrors ModelDef.neighbors: each architecture mutation is a
        spec string re-parsed through parse_model2 (which threads sizes
        and validates). Dedupes on the canonical spec so distinct raw
        mutations that normalize to the same model are yielded once, and
        self is excluded.
        """
        for p in self.precision.neighbors:
            yield parse_model2(self.spec, p)

        raw_seen: set[str] = set()
        seen: set[str] = {self.spec}
        for raw in self._gen_neighbor_specs():
            if raw in raw_seen:
                continue
            raw_seen.add(raw)
            model = parse_model2(raw, self.precision)
            model = self._sync_emb_width(model)
            if model.spec in seen:
                continue
            seen.add(model.spec)
            if model.is_valid():
                yield model

    def _sync_emb_width(self, model: Self) -> Self:
        """Structure mutations own the emb-width sync: a variant that
        resized the chain's last layer arrives with a stale X, because
        the parser never repairs specs (see embedding_codec.py).
        Re-spell the input at the mutated chain's final width and
        re-parse. Chains with no consistent width (suffix-scaled
        passthrough) still disagree afterwards and get filtered by
        the caller's is_valid check like any other invalid neighbor."""
        codec = model.codec
        if not isinstance(codec, EmbeddingCodecDef):
            return model
        if codec.last_width in (None, codec.emb_size):
            return model
        layers_spec = model.spec.split('|', 1)[1]
        input_spec = str(codec.with_emb_size(codec.last_width))
        return parse_model2(f'{input_spec}|{layers_spec}', self.precision)

    def build_jax(self) -> Model2Jax:
        return Model2Jax(
            self.codec.build_jax(),
            self.layer_seq.build_jax(self.precision.jax_dtype),
            total_padding=self.total_padding,
        )


# --- neighbor generation -------------------------------------------------
#
# Recursive, skip-free analog of ModelDef._gen_neighbor_specs. Each
# mutation is emitted as a canonical spec string and re-parsed (so size
# threading + is_valid live in one place). Split mutations form two
# independent families:
#   - residual: split.add / split.cat with a `pass` second path -- the
#     skip analog (add<->cat swap, span grow/shrink via wrap/unwrap).
#   - gate: split.mul with a dense (or pass) second path slaved to the
#     main path's size -- GeGLU/SwiGLU/self-gating.

# The search-valid activations (relu retired 2026-07, silu added
# 2026-08). Kept in step with the `is_valid` whitelists in
# layers/dense.py and layers/rnn.py.
_APPEND_ACTS = ("gelu", "tanh", "silu")
_APPEND_RECURRENT = ("gru", "mgru", "mingru", "lstm", "slstm", "mullstm")
# Same set plus the bare linear gate: `mul(dense.X.silu, dense.X)` is
# SwiGLU, `mul(dense.X.gelu, dense.X)` GeGLU.
_GATE_ACTS = (None, "gelu", "tanh", "silu")


def _strs(layers: list[LayerDef]) -> list[str]:
    return [str(l) for l in layers]


def _render_branch(layer_strs: list[str]) -> str:
    return "-".join(layer_strs) if layer_strs else "pass"


def _split_str(op: str, branch_strs: list[list[str]]) -> str:
    inner = ", ".join(_render_branch(b) for b in branch_strs)
    return f"split.{op}({inner})"


def _split_variants(split: SplitDef) -> Iterable[str]:
    """Neighbor split-strings: op-swap (residual only), branch
    mutations, and gate-path mutations (mul)."""
    bsl = [_strs(b.layers) for b in split.branches]
    if split.op in ("add", "cat"):
        other = "cat" if split.op == "add" else "add"
        yield _split_str(other, bsl)
        # Mutate each non-empty branch; leave `pass` branches as pass.
        for idx, br in enumerate(split.branches):
            if not br.layers:
                continue
            for variant in _seq_variants(br.layers, br.input_size):
                new = list(bsl)
                new[idx] = variant
                yield _split_str(split.op, new)
    elif split.op == "mul":  # gate family
        main = split.branches[0]
        main_strs = bsl[0]
        # Second-path (gate) variants: activation toggle + self-gate.
        for act in _GATE_ACTS:
            gate = f"dense.{main.size}" + (f".{act}" if act else "")
            yield _split_str("mul", [main_strs, [gate]])
        # self-gate: pass is the value (first); the gate slot (second)
        # is never pass under the canonical form.
        yield _split_str("mul", [[], main_strs])
        # Mutate the main path only; the gate stays fixed. A mutation
        # that changes the main's output size yields an unequal-size
        # mul that is_valid rejects, so a single-step coupled resize
        # (gate follows the main) is NOT offered here -- deferred TODO.
        # It's still reachable in a couple of hops (the invalid
        # intermediate's own neighbors resize the other side), which a
        # depth>=2 neighbor walk finds for free.
        for variant in _seq_variants(main.layers, main.input_size):
            yield _split_str("mul", [variant] + bsl[1:])


def _seq_variants(
    layers: list[LayerDef], input_size: int,
) -> Iterable[list[str]]:
    """Neighbor layer-string-lists for one sequence (top chain or a
    split branch). The caller composes them into a full spec.

    Sizing rule: appended and prepended layers are SIZE-PRESERVING,
    snapped down to the power-of-2 grid (17 -> 16 for the odd one-hot
    input widths; powers of 2 unchanged). Width exploration is the
    job of the in-place resize mutations, not of appends -- and for
    tied-codec models a size-preserving append leaves the table width
    alone instead of dragging it around via the X sync. The output
    head's size plays no role in chain mutations anymore.
    """
    strs = _strs(layers)
    n = len(layers)

    # 1. Mutate each layer in place (recurse into nested splits).
    for i, layer in enumerate(layers):
        if isinstance(layer, SplitDef):
            for sv in _split_variants(layer):
                yield strs[:i] + [sv] + strs[i + 1:]
        else:
            for nb in layer.neighbors():
                yield strs[:i] + [nb] + strs[i + 1:]

    last_output = layers[-1].size if layers else input_size
    new_size = floor_power2(last_output)

    # 2. Append a new layer (size-preserving, snapped to the grid).
    #    The bare dense is included: for tied-codec models it is the
    #    explicit adapter (the sanctioned trailing linear); elsewhere
    #    validity filters it like any other inapplicable append.
    yield strs + [f"dense.{new_size}"]
    for act in _APPEND_ACTS:
        for name in ("dense", "rnn"):
            yield strs + [f"{name}.{new_size}.{act}"]
    for name in _APPEND_RECURRENT:
        yield strs + [f"{name}.{new_size}"]
    # rglru is size-preserving (no size arg). Append both block extremes
    # as candidates: rglru.1 (a single full DxD gate) and rglru.{width}
    # (blocks == width, i.e. per-channel scalar gates, block_width 1).
    # Both are valid for the power-of-2 widths the search uses, and
    # reachable here directly rather than only via the mgru/mingru swap.
    yield strs + ["rglru.1"]
    if last_output > 1:
        yield strs + [f"rglru.{last_output}"]
    # Attention keeps the exact stream width (no snap: heads/window
    # mutate from there; tiny widths (head_dim < 4) are filtered by
    # is_valid).
    yield strs + [f"attn.{last_output}.1.{_ATTN_SWAP_WINDOW}"]
    if new_size >= 2:
        yield strs + [f"matlstm.{new_size}"]
        yield strs + [f"msr.{new_size}.1"]
    yield strs + ["conv.2"]

    # 3. Remove the last layer (symmetric with append: only a layer
    #    an append could have produced is removable).
    if layers:
        prev_output = input_size if n == 1 else layers[-2].size
        if last_output == floor_power2(prev_output):
            yield strs[:-1]

    # 4./5. Insert / remove suffix.2 (no skip-distance bumps in a tree).
    for i in range(n + 1):
        yield strs[:i] + ["suffix.2"] + strs[i:]
    for i, s in enumerate(strs):
        if s == "suffix.2":
            yield strs[:i] + strs[i + 1:]

    # 6./7. Insert / remove a normalization -- norm or rmsnorm -- never
    #       as the first layer.
    for i in range(1, n + 1):
        for nm in ("norm", "rmsnorm"):
            yield strs[:i] + [nm] + strs[i:]
    for i, s in enumerate(strs):
        if s in ("norm", "rmsnorm"):
            yield strs[:i] + strs[i + 1:]

    # 8./9. Prepend a dense lead-in / remove it (symmetric). Same
    #       sizing rule as appends: size-preserving on the grid.
    prepend_size = floor_power2(input_size)
    # The removal below is the exact inverse of this prepend, so the
    # two activation sets have to stay identical.
    for act in _APPEND_ACTS:
        yield [f"dense.{prepend_size}.{act}"] + strs
    if layers and layers[0].name == "dense":
        first = layers[0]
        if (first.size == floor_power2(input_size)
                and first._activation in _APPEND_ACTS):
            yield strs[1:]

    # 10. Introduce a residual split around a 1- or 2-layer span.
    for i in range(n):
        for d in (1, 2):
            if i + d > n:
                continue
            for op in ("add", "cat"):
                wrapped = _split_str(op, [strs[i:i + d], []])
                yield strs[:i] + [wrapped] + strs[i + d:]

    # 11. Introduce a mul gate around a single (non-split) layer.
    for i, layer in enumerate(layers):
        if isinstance(layer, SplitDef):
            continue
        gate = f"dense.{layer.size}"
        yield strs[:i] + [_split_str("mul", [[strs[i]], [gate]])] + strs[i + 1:]

    # 12. Remove a split (unwrap to its main branch).
    for i, layer in enumerate(layers):
        if isinstance(layer, SplitDef):
            main = _strs(layer.branches[0].layers)
            yield strs[:i] + main + strs[i + 1:]

    # 13. Grow / shrink a split's main branch by moving its right
    #     boundary across the adjacent layer -- the skip distance +-1
    #     analog. Grow pulls the following (non-split) layer into the
    #     main path; shrink pushes the main path's last layer back out
    #     after the split. A 1-layer main shrinks via unwrap (sec. 12);
    #     the other branches (pass / gate) are carried unchanged.
    #     The boundary moves by exactly one: a step that would end the
    #     branch on a suffix-like layer is simply filtered by is_valid,
    #     rather than bumped over it the way old skip-distance did --
    #     deferred, so a residual spanning a suffix needs a longer walk.
    for i, layer in enumerate(layers):
        if not isinstance(layer, SplitDef):
            continue
        others = [_strs(b.layers) for b in layer.branches[1:]]
        main_strs = _strs(layer.branches[0].layers)
        # grow: consume layers[i+1] into the main path.
        if i + 1 < n and not isinstance(layers[i + 1], SplitDef):
            grown = _split_str(layer.op, [main_strs + [strs[i + 1]]] + others)
            yield strs[:i] + [grown] + strs[i + 2:]
        # shrink: release the main path's last layer after the split.
        if len(main_strs) >= 2:
            shrunk = _split_str(layer.op, [main_strs[:-1]] + others)
            yield strs[:i] + [shrunk, main_strs[-1]] + strs[i + 1:]

    # 14. Append a self-gate on the running activation:
    #     split.mul(pass, dense.{size}[.act]). Value path is the
    #     identity (pass), gate is a learned dense of the matching size;
    #     the activations cover the GLU family. Unlike the wrap-a-layer
    #     gate (sec. 11) this can fire on the empty chain. Its inverse
    #     -- dropping a trailing self-gate -- is the section-12 unwrap
    #     (a mul whose main branch is pass unwraps to empty).
    #     Caveat: the gate must match the activation width, so for
    #     non-power-of-2 inputs (bits.2.oh+bp=6, bits.4.oh+bp=17) the
    #     size-matched gate dense is rejected by the power-of-2 rule and
    #     this is filtered from the empty chain -- it's still reachable
    #     after any real (power-of-2) layer. Fine for bits.1+bp / bytes.
    for act in _GATE_ACTS:
        gate = f"dense.{last_output}" + (f".{act}" if act else "")
        yield strs + [_split_str("mul", [[], [gate]])]
