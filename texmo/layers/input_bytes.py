from ..precision import Precision
from .input_bits import InputBitsJax


class InputBytesDef(object):
    """Descriptor for the bytes input layer."""

    def __init__(self, precision: Precision = Precision.FP32):
        self.ntokens = 256
        self.size = 256
        self.tokens_name = 'bytes'
        self.num_weights = 0
        self.num_mults = 0
        self.precision = precision

    def __str__(self):
        return 'bytes'

    def is_valid(self):
        return True

    def neighbors(self):
        return ('bits.4.oh+bp',)

    def build_jax(self) -> InputBitsJax:
        # bytes == bits.8.oh
        return InputBitsJax(nbits=8, one_hot=True, bp=False,
                            dtype=self.precision.jax_dtype)
