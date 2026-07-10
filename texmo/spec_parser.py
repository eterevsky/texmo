"""Spec parser for Model2.

One entry point: `parse_model2(spec, precision) -> Model2Def`.
The full Model2 grammar:

    spec        := [input_spec] "|" layer_list
    input_spec  := "" | "bytes" | "bits." ...
    layer_list  := layer ("-" layer)*
    layer       := simple_layer | split_layer
    split_layer := "split." op "(" branch ("," branch)* ")"
    op          := "mul" | "add" | "cat"
    branch      := layer_list | "pass"

`simple_layer` covers everything still parsed by
`model._build_layer_def` (dense, gru, suffix, conv, ...). Skip
syntax flows through too -- it returns a SkipDef which
`parse_model2` rejects until parser-level skip-to-split
translation lands (TODO item 3).

Branch arity is unbounded at parse level; `SplitDef.is_valid`
enforces 2-way for now. Multi-way is a one-character relaxation.
"""

from .layer import LayerDef
from .layers.one_hot_codec import OneHotCodecDef
from .layers.seq import LayerSeqDef
from .layers.skip import SkipDef
from .layers.split import SplitDef
from .model import _apply_merges, _build_layer_def
from .precision import Precision


def parse_model2(spec: str, precision: Precision, cap: bool = False):
    """Top-level spec parser. Returns a `Model2Def`.

    Splits the spec on `|` into the codec's input spec and the layer
    chain, builds each part, and stitches them together. Legacy
    `skip.D.op` syntax in the layer chain is translated to the
    equivalent `split.op(...)` form at parse time, recursively at
    every nesting level. The runtime sees a skip-free tree.

    `cap` enables logit soft-capping (see layers/codec.py). Off by
    default until its effect is validated against the result DB.
    """
    # Imported here to break the circular module dependency:
    # model2.py imports the parser indirectly via downstream code
    # paths, and Model2Def itself only needs to exist when this
    # function is called.
    from .model2 import Model2Def

    input_spec, layers_spec = _split_input_and_layers(spec)
    codec = OneHotCodecDef.from_spec(input_spec, precision, cap=cap)
    layers = parse_layer_list(layers_spec, codec.size)

    layer_seq = LayerSeqDef(layers, input_size=codec.size)
    codec.set_head_width(layers[-1].size if layers else codec.size)

    # Canonical spec is rebuilt from the parsed tree -- so callers
    # see split.op(...) form whether the user wrote split.op or the
    # legacy skip.D.op (which gets translated in parse_layer_list).
    # Parsing the canonical spec round-trips to an equivalent
    # Model2Def.
    canonical = f'{codec}|{layer_seq}'

    return Model2Def(
        spec=canonical, precision=precision,
        codec=codec, layer_seq=layer_seq,
    )


def _split_input_and_layers(spec: str) -> tuple[str, str]:
    parts = spec.split("|")
    if len(parts) == 1:
        return "", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("Model spec can't contain more than one |")


def _translate_skips(
    layers: list[LayerDef],
) -> list[LayerDef]:
    """Rewrite SkipDef pseudo-layers as SplitDefs, recursively.

    Each `skip.D.op` at position `i` spans the next D layers
    (positions i+1 ... i+D) and merges the source (input at
    position i) with the output of layer i+D via `op`. The
    equivalent split is:

        layers[i] = split.op(
            translate(layers[i+1]-...-layers[i+D]), pass)

    and the consumed positions i+1 ... i+D are dropped from the
    rewritten list. The main branch is translated recursively, so a
    skip nested inside another skip's span becomes a nested split.
    The merge target (layer i+D+1) consumes the split's output
    directly -- its input_size already accounts for the merged shape
    because the original parser set it that way.

    Two failure modes:
      - Crossing skip spans: a partial overlap where neither span
        contains the other. Can't be expressed as a tree. ValueError.
      - Span overshoots the end of the layer list. ValueError.

    Nested (laminar) spans are fine -- they form a forest and
    translate into nested splits.
    """
    if not any(isinstance(l, SkipDef) for l in layers):
        return layers

    n = len(layers)
    skips: list[tuple[int, int, str]] = [
        (i, layer.distance, layer.op)
        for i, layer in enumerate(layers)
        if isinstance(layer, SkipDef)
    ]
    # Overshoot check: a skip at i with distance D claims positions
    # [i, i+D+1); that range must fit within the list.
    for i, d, _ in skips:
        if i + d + 1 > n:
            raise ValueError(
                f"skip at position {i} (distance {d}) overshoots "
                f"the layer list of length {n}")

    # Laminar check: sorted by start, any two spans must be disjoint
    # or nested, never crossing (partial overlap). Nested skips become
    # nested splits via the recursion below; crossing ones can't be a
    # tree.
    spans = sorted((i, i + d + 1) for i, d, _ in skips)
    for a, (lo1, hi1) in enumerate(spans):
        for lo2, hi2 in spans[a + 1:]:
            if lo2 >= hi1:
                break  # disjoint -- and so are all later spans
            if hi2 > hi1:
                raise ValueError(
                    f"crossing skip spans can't be expressed as a "
                    f"tree: [{lo1}, {hi1}) crosses [{lo2}, {hi2})")

    # Walk the list. The walk only lands on outermost skips: a nested
    # skip lives inside some outer skip's [i+1, i+D+1) range, which is
    # consumed -- and recursively translated -- when the outer skip is
    # reached.
    skip_at = {i: (d, op) for i, d, op in skips}
    out: list[LayerDef] = []
    pos = 0
    while pos < n:
        if pos in skip_at:
            d, op = skip_at[pos]
            source_size = layers[pos].input_size
            main_seq = LayerSeqDef(
                _translate_skips(list(layers[pos + 1: pos + d + 1])),
                input_size=source_size,
            )
            side_seq = LayerSeqDef([], input_size=source_size)
            out.append(SplitDef(
                op, [main_seq, side_seq], input_size=source_size))
            pos += d + 1
        else:
            out.append(layers[pos])
            pos += 1
    return out


def parse_layer_list(
    layers_spec: str, input_size: int,
) -> list[LayerDef]:
    """Parse the layer-chain portion of a Model2 spec.

    `layers_spec` is the string AFTER the `|` (or the whole spec if
    there's no input layer). `input_size` is the channel dim coming
    into the first layer. Returns the parsed `LayerDef`s; channel
    sizes are threaded through automatically. Any `skip.D.op`
    pseudo-layers in the parsed list are rewritten as the
    equivalent `split.op(...)` before return, so callers see a
    skip-free tree. Recurses naturally into split branches because
    `_parse_split` calls back into this function for each branch.
    """
    if not layers_spec.strip():
        return []
    layers: list[LayerDef] = []
    shape = input_size
    # Track pending skip merges per target position so the merge-
    # target layer's input_size reflects the post-merge shape (the
    # original `model.py` parser does the same).
    pending_merges: dict[int, list[tuple[int, str, int]]] = {}
    pieces = list(_split_at_depth_0(layers_spec, '-'))
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            raise ValueError(
                f"empty layer in spec: {layers_spec!r}")
        if i in pending_merges:
            shape = _apply_merges(shape, pending_merges.pop(i))
        layer = _parse_layer(piece, shape)
        layers.append(layer)
        if isinstance(layer, SkipDef):
            target = i + layer.distance + 1
            pending_merges.setdefault(target, []).append(
                (shape, layer.op, i))
        shape = layer.size
    return _translate_skips(layers)


def _parse_layer(spec: str, input_size: int) -> LayerDef:
    if spec.startswith('split.'):
        return _parse_split(spec, input_size)
    return _build_layer_def(spec, input_size)


def _parse_split(spec: str, input_size: int) -> SplitDef:
    open_idx = spec.find('(')
    if open_idx < 0:
        raise ValueError(f"split missing '(': {spec!r}")
    if not spec.endswith(')'):
        raise ValueError(f"split missing ')': {spec!r}")

    head = spec[:open_idx]              # "split.{op}"
    body = spec[open_idx + 1:-1]         # contents between parens

    head_parts = head.split('.')
    if len(head_parts) != 2 or head_parts[0] != 'split':
        raise ValueError(f"invalid split head: {head!r}")
    op = head_parts[1]

    if not body.strip():
        raise ValueError(
            f"split has no branches (need at least one): {spec!r}")

    branches: list[LayerSeqDef] = []
    for branch_spec in _split_at_depth_0(body, ','):
        branch_spec = branch_spec.strip()
        if branch_spec == 'pass':
            branch_layers: list[LayerDef] = []
        elif branch_spec:
            branch_layers = parse_layer_list(branch_spec, input_size)
        else:
            raise ValueError(
                f"empty branch in split (use 'pass' for "
                f"identity): {spec!r}")
        branches.append(
            LayerSeqDef(branch_layers, input_size=input_size))

    # SplitDef constructor validates `op`. Branch arity is checked
    # at is_valid time, not here.
    return SplitDef(op, branches, input_size=input_size)


def _split_at_depth_0(s: str, delim: str) -> list[str]:
    """Split `s` on `delim` characters that aren't inside parens.

    Used both to split the top-level layer chain on `-` (skipping
    `-` characters inside a split's branches) and to split a
    split's body on `,` (skipping `,` inside nested splits).
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"unbalanced ')' in spec: {s!r}")
        elif c == delim and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    if depth != 0:
        raise ValueError(f"unbalanced '(' in spec: {s!r}")
    parts.append(s[start:])
    return parts
