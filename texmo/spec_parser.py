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

`simple_layer` covers everything parsed by `_build_layer_def`
below (dense, gru, suffix, conv, ...). The legacy `skip.D.op`
syntax was retired in 2026-07 after the result DB was migrated to
split form; residuals are spelled `split.op(span, pass)` directly.

Branch arity is unbounded at parse level; `SplitDef.is_valid`
enforces 2-way for now. Multi-way is a one-character relaxation.
"""

from .layer import LayerDef
from .layers.attn import AttnDef
from .layers.conv import ConvDef
from .layers.dense import DenseDef
from .layers.embedding_codec import EmbeddingCodecDef
from .layers.gru import GruDef, MgruDef, MinGruDef
from .layers.latent import LatentDef, LrnnDef
from .layers.lmgu import LmguDef
from .layers.lstm import LstmDef
from .layers.matlstm import MatLstmDef
from .layers.msr import MsrDef
from .layers.mullstm import MulLstmDef
from .layers.norm import NormDef
from .layers.one_hot_codec import OneHotCodecDef
from .layers.rglru import RglruDef
from .layers.rmsnorm import RmsNormDef
from .layers.rnn import RnnDef
from .layers.seq import LayerSeqDef
from .layers.slstm import SLstmDef
from .layers.split import SplitDef
from .layers.suffix import SuffixDef
from .precision import Precision


def parse_model2(spec: str, precision: Precision, cap: bool = True):
    """Top-level spec parser. Returns a `Model2Def`.

    Splits the spec on `|` into the codec's input spec and the layer
    chain, builds each part, and stitches them together.

    `cap` enables logit soft-capping (see layers/codec.py). On by
    default; cap=False is for experiments and for comparing against
    the legacy uncapped runtime.
    """
    # Imported here to break the circular module dependency:
    # model2.py imports the parser indirectly via downstream code
    # paths, and Model2Def itself only needs to exist when this
    # function is called.
    from .model2 import Model2Def

    input_spec, layers_spec = _split_input_and_layers(spec)
    # `.emb.` selects the tied-embedding codec; everything else is a
    # fixed codebook. Same API either way.
    if '.emb.' in input_spec:
        codec = EmbeddingCodecDef.from_spec(input_spec, precision, cap=cap)
    else:
        codec = OneHotCodecDef.from_spec(input_spec, precision, cap=cap)
    layers = parse_layer_list(layers_spec, codec.size)

    layer_seq = LayerSeqDef(layers, input_size=codec.size)
    codec.set_head_width(layers[-1].size if layers else codec.size)

    # Canonical spec is rebuilt from the parsed tree. Parsing the
    # canonical spec round-trips to an equivalent Model2Def.
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


def parse_layer_list(
    layers_spec: str, input_size: int,
) -> list[LayerDef]:
    """Parse the layer-chain portion of a Model2 spec.

    `layers_spec` is the string AFTER the `|` (or the whole spec if
    there's no input layer). `input_size` is the channel dim coming
    into the first layer. Returns the parsed `LayerDef`s; channel
    sizes are threaded through automatically. Recurses naturally
    into split branches because `_parse_split` calls back into this
    function for each branch.
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


def _parse_nope(spec: str, parts: list[str], index: int) -> bool:
    """Read the optional trailing `nope` qualifier on attn / msr.

    Bare means RoPE; there is no explicit `.rope` spelling, so each
    configuration has exactly one canonical form and existing DB specs
    stay textually identical. Anything else in the slot is a typo and
    fails here rather than silently parsing as the rotated form.
    """
    if len(parts) <= index:
        return False
    if parts[index] != "nope":
        raise ValueError(
            f"'{spec}': expected 'nope' or nothing after position "
            f"{index - 1}, got '{parts[index]}'")
    return True


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
            int(parts[1]), int(parts[2]), input_size=input_size,
            nope=_parse_nope(spec, parts, 3))

    if name == "norm":
        return NormDef(input_size=input_size)

    if name == "rmsnorm":
        return RmsNormDef(input_size=input_size)

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

    if name == "conv":
        kernel = int(parts[1])
        return ConvDef(kernel, input_size=input_size)

    if name == "rglru":
        return RglruDef(int(parts[1]), input_size=input_size)

    if name == "attn":
        return AttnDef(
            int(parts[1]), int(parts[2]), int(parts[3]),
            input_size=input_size, nope=_parse_nope(spec, parts, 4))

    raise ValueError(f"Unknown layer type: {name}")


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
