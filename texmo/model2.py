"""Model2: layer-DAG-shaped reimplementation of ModelDef.

The layer chain is a `LayerSeqDef` rather than a flat
`list[LayerDef]`. SplitDef will eventually plug into this by
holding two LayerSeqDefs as its branches; Model2 today only
handles plain sequences without splits or skips.

JAX only. No PyTorch counterpart.
"""

from .layers.dense import DenseDef
from .layers.input_bits import InputBitsDef
from .layers.input_bytes import InputBytesDef
from .layers.seq import LayerSeqDef
from .layers.skip import SkipDef
from .model import _build_layer_def
from .model2_jax import Model2Jax
from .precision import Precision


class Model2Def:
    """Model spec descriptor built on LayerSeqDef.

    Mirrors ModelDef's public surface (spec / num_weights /
    num_mults / is_valid / build_jax / equality / hashing). Skip
    syntax is rejected -- the migration path is parser-level
    translation to `split.op(...)` form, so Model2's runtime stays
    skip-free.
    """

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
            self.input = InputBitsDef.from_spec(
                input_spec, precision=precision)
        else:
            raise ValueError(f"Unknown input type: '{input_spec}'")

        layers = []
        shape = self.input.size
        if layers_spec:
            for layer_spec in layers_spec.split("-"):
                layer = _build_layer_def(layer_spec, shape)
                if isinstance(layer, SkipDef):
                    raise NotImplementedError(
                        "Model2 doesn't handle `skip` yet; it'll be "
                        "translated to `split` by the parser in the "
                        "next iteration.")
                layers.append(layer)
                shape = layer.size

        self.layer_seq = LayerSeqDef(layers, input_size=self.input.size)

        self.ntokens = self.input.ntokens
        # 1 for the autoregressive shift + extras consumed by the
        # layer chain. layer_seq.length is already 1 + sum(extras),
        # so it doubles as the model's total padding requirement.
        self.total_padding = self.layer_seq.length
        output_size = self.ntokens if self.ntokens > 2 else 1
        self.output = DenseDef(output_size, input_size=shape)

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
