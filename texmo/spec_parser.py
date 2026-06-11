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
from .layers.dense import DenseDef
from .layers.input_bits import InputBitsDef
from .layers.input_bytes import InputBytesDef
from .layers.seq import LayerSeqDef
from .layers.skip import SkipDef
from .layers.split import SplitDef
from .model import _build_layer_def
from .precision import Precision


def parse_model2(spec: str, precision: Precision):
    """Top-level spec parser. Returns a `Model2Def`.

    Splits the spec on `|` into the input encoding and the layer
    chain, builds each part, and stitches them together. Legacy
    `skip.D.op` syntax in the layer chain is translated to the
    equivalent `split.op(...)` form at parse time, recursively at
    every nesting level. The runtime sees a skip-free tree.
    """
    # Imported here to break the circular module dependency:
    # model2.py imports the parser indirectly via downstream code
    # paths, and Model2Def itself only needs to exist when this
    # function is called.
    from .model2 import Model2Def

    input_spec, layers_spec = _split_input_and_layers(spec)
    input_layer = _parse_input(input_spec, precision)
    layers = parse_layer_list(layers_spec, input_layer.size)

    layer_seq = LayerSeqDef(layers, input_size=input_layer.size)
    last_shape = layers[-1].size if layers else input_layer.size
    output_size = (
        input_layer.ntokens if input_layer.ntokens > 2 else 1)
    output = DenseDef(output_size, input_size=last_shape)

    return Model2Def(
        spec=spec, precision=precision,
        input=input_layer, layer_seq=layer_seq, output=output,
    )


def _split_input_and_layers(spec: str) -> tuple[str, str]:
    parts = spec.split("|")
    if len(parts) == 1:
        return "", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("Model spec can't contain more than one |")


def _parse_input(input_spec: str, precision: Precision):
    if input_spec == '' or input_spec == 'bytes':
        return InputBytesDef(precision=precision)
    if input_spec.startswith('bits.'):
        return InputBitsDef.from_spec(
            input_spec, precision=precision)
    raise ValueError(f"Unknown input type: '{input_spec}'")


def _translate_skips(
    layers: list[LayerDef],
) -> list[LayerDef]:
    """Rewrite SkipDef pseudo-layers as SplitDefs.

    Each `skip.D.op` at position `i` spans the next D layers
    (positions i+1 ... i+D) and merges the source (input at
    position i) with the output of layer i+D via `op`. The
    equivalent split is:

        layers[i] = split.op(layers[i+1]-...-layers[i+D], pass)

    and the consumed positions i+1 ... i+D are dropped from the
    rewritten list. The merge target (layer i+D+1) consumes the
    split's output directly -- its input_size already accounts for
    the merged shape because the original parser set it that way.

    Two failure modes:
      - Overlapping skip spans (can't be a tree). ValueError.
      - Span overshoots the end of the layer list. ValueError.
    """
    if not any(isinstance(l, SkipDef) for l in layers):
        return layers

    skips: list[tuple[int, int, str]] = [
        (i, layer.distance, layer.op)
        for i, layer in enumerate(layers)
        if isinstance(layer, SkipDef)
    ]
    # Overlap check: a skip at i with distance D claims positions
    # [i, i+D+1). Sort by start; adjacent claims may touch but not
    # overlap (e.g. one ends exactly where the next begins).
    spans = sorted((i, i + d + 1) for i, d, _ in skips)
    for (lo1, hi1), (lo2, hi2) in zip(spans, spans[1:]):
        if hi1 > lo2:
            raise ValueError(
                f"overlapping skip spans can't be expressed as a "
                f"tree: [{lo1}, {hi1}) overlaps [{lo2}, {hi2})")
    # Overshoot check.
    n = len(layers)
    for i, d, _ in skips:
        if i + d + 1 > n:
            raise ValueError(
                f"skip at position {i} (distance {d}) overshoots "
                f"the layer list of length {n}")

    skip_at = {i: (d, op) for i, d, op in skips}
    out: list[LayerDef] = []
    pos = 0
    while pos < n:
        if pos in skip_at:
            d, op = skip_at[pos]
            source_size = layers[pos].input_size
            main_seq = LayerSeqDef(
                list(layers[pos + 1: pos + d + 1]),
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
    for piece in _split_at_depth_0(layers_spec, '-'):
        piece = piece.strip()
        if not piece:
            raise ValueError(
                f"empty layer in spec: {layers_spec!r}")
        layer = _parse_layer(piece, shape)
        layers.append(layer)
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
