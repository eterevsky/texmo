"""Model2: layer-DAG-shaped reimplementation of ModelDef.

The layer chain is a `LayerSeqDef` rather than a flat
`list[LayerDef]`. SplitDef plugs into this by holding two
LayerSeqDefs as its branches.

Construction goes through `spec_parser.parse_model2(spec,
precision)`: that's where every string-to-tree decision lives,
including the `|`-split, the input-layer dispatch, and the
layer-list grammar. `Model2Def.__init__` only takes already-built
pieces and stores them.

JAX only. No PyTorch counterpart.
"""

from .layers.dense import DenseDef
from .layers.input_bits import InputBitsDef
from .layers.input_bytes import InputBytesDef
from .layers.seq import LayerSeqDef
from .model2_jax import Model2Jax
from .precision import Precision

# Type alias for the input layers parse_model2 can produce -- kept
# loose because the input-layer hierarchy doesn't have a single
# base class today.
InputLayer = InputBytesDef | InputBitsDef


class Model2Def:
    """Model spec descriptor built on LayerSeqDef.

    Stores already-parsed pieces (`input`, `layer_seq`, `output`)
    plus the original spec string for round-tripping. Use
    `spec_parser.parse_model2(spec, precision)` to construct from a
    string.

    Mirrors ModelDef's public surface (spec / num_weights /
    num_mults / is_valid / build_jax / equality / hashing).
    """

    def __init__(
        self,
        *,
        spec: str,
        precision: Precision,
        input: InputLayer,
        layer_seq: LayerSeqDef,
        output: DenseDef,
    ):
        self.spec = spec
        self.precision = precision
        self.input = input
        self.layer_seq = layer_seq
        self.output = output
        self.ntokens = input.ntokens
        # 1 for the autoregressive shift + extras consumed by the
        # layer chain. layer_seq.length is already 1 + sum(extras),
        # so it doubles as the model's total padding requirement.
        self.total_padding = layer_seq.length

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
        return (
            self.input.num_weights
            + self.layer_seq.num_weights
            + self.output.num_weights
        )

    @property
    def num_mults(self) -> int:
        return (
            self.input.num_mults
            + self.layer_seq.num_mults
            + self.output.num_mults
        )

    @property
    def num_layers(self) -> int:
        """Hidden-layer count. Recursive: a Split counts as itself
        plus the layers inside its branches. For any skip spec that
        the parser translated to split form, the count matches the
        legacy ModelDef.num_layers (which counted the SkipDef
        pseudo-layer as 1 and each in-skip layer as 1)."""
        return self.layer_seq.num_layers

    def is_valid(self) -> bool:
        return (
            self.input.is_valid()
            and self.layer_seq.is_valid()
            and self.output.is_valid()
        )

    def build_jax(self) -> Model2Jax:
        return Model2Jax(
            self.input.build_jax(),
            self.layer_seq.build_jax(self.precision.jax_dtype),
            self.output.build_jax(self.precision.jax_dtype),
            ntokens=self.ntokens,
            total_padding=self.total_padding,
        )
