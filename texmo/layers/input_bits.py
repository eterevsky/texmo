import jax
import jax.numpy as jnp

from ..precision import Precision

# Number of bits used to encode the bit chunk position within a byte.
_BP = {1: 3, 2: 2, 4: 1, 8: 0}

# Only a small subset of input specs participates in search. Other
# specs remain constructible (you can run them manually) but won't be
# discovered as neighbors during exploration.
_INPUT_NEIGHBORS = {
    'bits.1+bp': ('bits.2.oh+bp',),
    'bits.2.oh+bp': ('bits.1+bp', 'bits.4.oh+bp'),
    'bits.4.oh+bp': ('bits.2.oh+bp', 'bytes'),
}


def _to_bit_array(n: int, nbits: int) -> list[float]:
    enc = []
    for i in range(nbits):
        enc.append(float(n % 2))
        n //= 2
    return enc


class InputBitsJax:
    """JAX input encoding for bit-chunks."""

    def __init__(self, nbits: int, one_hot: bool, bp: bool, dtype):
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = 2 ** nbits
        self.dtype = dtype

        if one_hot:
            enc_size = self.ntokens
        else:
            enc_size = nbits
        self.size = enc_size
        if bp:
            self.size += _BP[nbits]

        encodings = []
        for i in range(self.ntokens):
            if one_hot:
                row = [0.0] * self.ntokens
                row[i] = 1.0
            else:
                row = _to_bit_array(i, nbits)
            encodings.append(row)
        self.encodings = jnp.array(encodings, dtype=dtype)

        if bp:
            self.positions = 8 // nbits
            bp_bits = _BP[nbits]
            pos_encodings = []
            for i in range(self.positions):
                pos_encodings.append(_to_bit_array(i, bp_bits))
            self.pos_encodings = jnp.array(pos_encodings, dtype=dtype)

    def init_weights(self, rng: jax.Array) -> None:
        # Parameter-free encoding. All input layers share the weighted
        # signature (weights as first argument, None here) so Model2Jax
        # calls them uniformly.
        return None

    def init_state(self) -> int | None:
        if self.bp:
            return 0
        return None

    def _initial_vector(self, weights, position: int = -1) -> jax.Array:
        """Input vector for 'no token observed yet' (max entropy)."""
        if self.one_hot:
            v = jnp.full((self.ntokens,), 1.0 / self.ntokens, dtype=self.dtype)
        else:
            v = jnp.full((self.nbits,), 0.5, dtype=self.dtype)
        if self.bp:
            v = jnp.concatenate([v, self.pos_encodings[position % self.positions]])
        return v

    def step(self, weights, state, token: int) -> tuple:
        """Encode a single token, returning (new_state, output)."""
        out = self.encodings[token]
        if self.bp:
            out = jnp.concatenate([out, self.pos_encodings[state]])
            state = (state + 1) % self.positions
        return state, out

    def forward(
        self, weights, tokens: jax.Array, padding: int = 0,
    ) -> jax.Array:
        """Encode a batch of token sequences.

        Args:
            tokens: (batch, seq_len) int array.
            padding: initial positions to prepend (max-entropy input
                with correct position encoding).

        Returns:
            (batch, seq_len + padding, size).
        """
        batch, seq_len = tokens.shape
        out = self.encodings[tokens]  # (batch, seq_len, enc_size)

        if self.bp:
            reps = (seq_len + self.positions - 1) // self.positions
            pos = jnp.tile(self.pos_encodings, (reps, 1))[:seq_len]
            pos = jnp.broadcast_to(pos, (batch, seq_len, pos.shape[-1]))
            out = jnp.concatenate([out, pos], axis=-1)

        if padding > 0:
            pad_vecs = jnp.stack(
                [self._initial_vector(weights, position=-padding + i)
                 for i in range(padding)])
            pad = jnp.broadcast_to(pad_vecs, (batch, padding, self.size))
            out = jnp.concatenate([pad, out], axis=1)

        return out


class InputBitsDef:
    """Descriptor for the bits input layer."""

    def __init__(self, nbits: int, one_hot: bool, bp: bool,
                 precision: Precision = Precision.FP32):
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = 2 ** nbits
        self.precision = precision
        self.tokens_name = f'bits.{nbits}'
        self.num_weights = 0
        self.num_mults = 0

        if one_hot:
            self.size = self.ntokens
        else:
            self.size = nbits
        if bp:
            self.size += _BP[nbits]

    @staticmethod
    def from_spec(spec: str, precision: Precision = Precision.FP32) -> 'InputBitsDef':
        if spec == 'bytes':
            spec = 'bits.8.oh'

        parts = spec.split('+')
        bp = False
        if len(parts) > 1:
            assert len(parts) == 2
            assert parts[1] == 'bp'
            bp = True

        main = parts[0].split('.')
        assert main[0] == 'bits'
        nbits = int(main[1])
        assert nbits in (1, 2, 4, 8)

        one_hot = len(main) > 2 and main[2] == 'oh'

        return InputBitsDef(nbits, one_hot, bp, precision)

    def __str__(self):
        if self.nbits == 8 and self.one_hot and not self.bp:
            return 'bytes'
        s = f'bits.{self.nbits}'
        if self.one_hot:
            s += '.oh'
        if self.bp:
            s += '+bp'
        return s

    def is_valid(self):
        if self.nbits not in (1, 2, 4, 8):
            return False
        if self.bp and self.nbits >= 8:
            return False
        if self.one_hot and self.nbits == 1:
            return False
        return True

    def neighbors(self):
        spec = str(self)
        return _INPUT_NEIGHBORS.get(spec, ())

    def build_jax(self) -> InputBitsJax:
        return InputBitsJax(self.nbits, self.one_hot, self.bp, self.precision.jax_dtype)
