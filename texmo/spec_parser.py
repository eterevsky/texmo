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
    chain, builds each part, and stitches them together. Skip
    syntax is rejected anywhere in the tree (including inside
    Split branches) until parser-level translation to `split`
    form lands.
    """
    # Imported here to break the circular module dependency:
    # model2.py imports the parser indirectly via downstream code
    # paths, and Model2Def itself only needs to exist when this
    # function is called.
    from .model2 import Model2Def

    input_spec, layers_spec = _split_input_and_layers(spec)
    input_layer = _parse_input(input_spec, precision)
    layers = parse_layer_list(layers_spec, input_layer.size)
    if _has_skip(layers):
        raise NotImplementedError(
            "Model2 doesn't handle `skip` yet; the parser will "
            "translate it to `split` in a follow-up.")

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


def _has_skip(layers: list[LayerDef]) -> bool:
    for layer in layers:
        if isinstance(layer, SkipDef):
            return True
        if isinstance(layer, SplitDef):
            for branch in layer.branches:
                if _has_skip(branch.layers):
                    return True
    return False


def parse_layer_list(
    layers_spec: str, input_size: int,
) -> list[LayerDef]:
    """Parse the layer-chain portion of a Model2 spec.

    `layers_spec` is the string AFTER the `|` (or the whole spec if
    there's no input layer). `input_size` is the channel dim coming
    into the first layer. Returns the parsed `LayerDef`s; channel
    sizes are threaded through automatically.
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
    return layers


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
