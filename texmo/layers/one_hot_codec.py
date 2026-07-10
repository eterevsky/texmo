"""OneHotCodec: the fixed-codebook codec implementation.

Values are encoded by a fixed (parameter-free) code -- one-hot or
binary bits, optionally with binary position-within-byte bits (`+bp`)
-- and decoded by an independent learnable dense head, optionally
soft-capped (see codec.py). The name reflects the specs in actual use
(`bytes`, `bits.1+bp`, `bits.2.oh+bp`, `bits.4.oh+bp`, `tokens.*.oh`);
the grammar still parses the other bits variants, but `is_valid`
rejects them.

Input specs:

    ""|"bytes"                          one-hot over 256 (= bits.8.oh)
    bits.{1|2|4|8}[.oh][+bp]            bit chunks, binary or one-hot
    tokens.{ntokens}.{variation}.oh     one-hot over a tokenset

The binary special case (ntokens <= 2) produces a single logit x
treated as [x, 0]; the padding happens after the cap (tanh(0) == 0, so
the order is immaterial).

JAX only.
"""
import jax
import jax.numpy as jnp

from ..precision import Precision
from .codec import cap_logits
from .dense import DenseDef, DenseJax

# Number of bits used to encode the bit-chunk position within a byte.
_BP = {1: 3, 2: 2, 4: 1, 8: 0}

# The blessed input set: exactly these participate in search, and
# exactly these are valid. Other bits variants (non-oh bits.2/bits.4,
# oh-less combos, ...) still parse and run for manual experiments but
# count as invalid. The neighbor ladder is defined over the same set.
_INPUT_NEIGHBORS = {
    'bytes': ('bits.4.oh+bp',),
    'bits.1+bp': ('bits.2.oh+bp',),
    'bits.2.oh+bp': ('bits.1+bp', 'bits.4.oh+bp'),
    'bits.4.oh+bp': ('bits.2.oh+bp', 'bytes'),
}
_VALID_INPUTS = frozenset(_INPUT_NEIGHBORS)


def _to_bit_array(n: int, nbits: int) -> list[float]:
    enc = []
    for i in range(nbits):
        enc.append(float(n % 2))
        n //= 2
    return enc


class OneHotCodecJax:
    """Runtime for the fixed-codebook codec.

    Encoding: a codebook lookup for bit-chunk inputs (`nbits` set), or
    `jax.nn.one_hot` for tokenized inputs (`nbits is None` -- a 256k
    vocabulary never materializes a table). Decoding: the dense head,
    optionally soft-capped, padded to two logits in the binary case.

    The input side is parameter-free; weights live in the model
    pytree's slots 0 (always None here) and 2 (the head), so existing
    checkpoints keep their layout.
    """

    def __init__(
        self,
        *,
        nbits: int | None,
        one_hot: bool,
        bp: bool,
        ntokens: int,
        head: DenseJax,
        cap: bool,
        dtype,
    ):
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = ntokens
        self.head = head
        self.cap = cap
        self.dtype = dtype
        self._pad_output = ntokens <= 2

        if nbits is None:
            self.size = ntokens
            self.encodings = None
        else:
            self.size = ntokens if one_hot else nbits
            encodings = []
            for i in range(ntokens):
                if one_hot:
                    row = [0.0] * ntokens
                    row[i] = 1.0
                else:
                    row = _to_bit_array(i, nbits)
                encodings.append(row)
            self.encodings = jnp.array(encodings, dtype=dtype)
            if bp:
                self.positions = 8 // nbits
                self.pos_encodings = jnp.array(
                    [_to_bit_array(i, _BP[nbits])
                     for i in range(self.positions)],
                    dtype=dtype)
                self.size += _BP[nbits]

    # -- weights ----------------------------------------------------------

    def init_input_weights(self, rng: jax.Array) -> None:
        # Fixed codebook: no learned input weights. The slot stays in
        # the pytree (None) so the layout matches EmbeddingCodec and
        # existing checkpoints.
        return None

    def init_head_weights(self, rng: jax.Array):
        return self.head.init_weights(rng)

    # -- input side -------------------------------------------------------

    def init_state(self) -> int | None:
        """Step-mode state: bit-chunk position within the byte, or
        None when position encoding is not used."""
        if self.bp:
            return 0
        return None

    def initial_vector(self, weights, position: int = -1) -> jax.Array:
        """Input vector for 'no token observed yet' (max entropy):
        uniform 1/n over one-hot codes, 0.5 per value bit otherwise.
        `position` indexes the bp cycle and may be negative (padding
        counts down toward the first real chunk)."""
        if self.nbits is None or self.one_hot:
            v = jnp.full((self.ntokens,), 1.0 / self.ntokens,
                         dtype=self.dtype)
        else:
            v = jnp.full((self.nbits,), 0.5, dtype=self.dtype)
        if self.bp:
            v = jnp.concatenate(
                [v, self.pos_encodings[position % self.positions]])
        return v

    def encode(
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
        if self.encodings is None:
            out = jax.nn.one_hot(tokens, self.ntokens, dtype=self.dtype)
        else:
            out = self.encodings[tokens]

        if self.bp:
            reps = (seq_len + self.positions - 1) // self.positions
            pos = jnp.tile(self.pos_encodings, (reps, 1))[:seq_len]
            pos = jnp.broadcast_to(pos, (batch, seq_len, pos.shape[-1]))
            out = jnp.concatenate([out, pos], axis=-1)

        if padding > 0:
            pad_vecs = jnp.stack(
                [self.initial_vector(weights, position=-padding + i)
                 for i in range(padding)])
            pad = jnp.broadcast_to(pad_vecs, (batch, padding, self.size))
            out = jnp.concatenate([pad, out], axis=1)

        return out

    def encode_step(self, weights, state, token) -> tuple:
        """Encode a single token, returning (new_state, vector)."""
        if self.encodings is None:
            return state, jax.nn.one_hot(
                token, self.ntokens, dtype=self.dtype)
        out = self.encodings[token]
        if self.bp:
            out = jnp.concatenate([out, self.pos_encodings[state]])
            state = (state + 1) % self.positions
        return state, out

    # -- output side ------------------------------------------------------

    def logits(self, weights, h: jax.Array) -> jax.Array:
        """(..., K) hidden activations -> (..., ntokens) logits."""
        out = self.head.forward(weights, h)
        if self.cap:
            out = cap_logits(out)
        if self._pad_output:
            pad_widths = [(0, 0)] * (out.ndim - 1) + [(0, 1)]
            out = jnp.pad(out, pad_widths)
        return out

    def logits_step(self, weights, h: jax.Array) -> jax.Array:
        _, out = self.head.step(weights, None, h)
        if self.cap:
            out = cap_logits(out)
        if self._pad_output:
            out = jnp.pad(out, (0, 1))
        return out


class OneHotCodecDef:
    """Descriptor for the fixed-codebook codec.

    Owns the full pre-`|` grammar plus the head. Construction is
    two-phase because the head's width isn't known until the layer
    chain is parsed: `from_spec(...)` first (this fixes `size`, which
    the chain parse needs), then `set_head_width(last_width)`. Both
    calls live in `spec_parser.parse_model2`.

    `str(codec)` renders the canonical input spec (`bits.8.oh` ->
    `bytes`), which keeps neighbor-spec generation and the timing
    predictor's input type_id unchanged.
    """

    def __init__(
        self,
        *,
        nbits: int | None,
        one_hot: bool,
        bp: bool,
        ntokens: int,
        variation: str | None = None,
        precision: Precision = Precision.FP32,
        cap: bool = False,
    ):
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = ntokens
        self.variation = variation
        self.precision = precision
        self.cap = cap
        self.head: DenseDef | None = None

        if nbits is None:
            self.size = ntokens
            self.tokens_name = f'tokens{ntokens}_{variation}'
        else:
            self.size = ntokens if one_hot else nbits
            if bp:
                self.size += _BP[nbits]
            self.tokens_name = 'bytes' if nbits == 8 else f'bits.{nbits}'

    @staticmethod
    def from_spec(
        spec: str,
        precision: Precision = Precision.FP32,
        cap: bool = False,
    ) -> 'OneHotCodecDef':
        if spec == '' or spec == 'bytes':
            spec = 'bits.8.oh'
        if spec.startswith('bits.'):
            parts = spec.split('+')
            bp = False
            if len(parts) > 1:
                if len(parts) != 2 or parts[1] != 'bp':
                    raise ValueError(f"bad input spec: '{spec}'")
                bp = True
            main = parts[0].split('.')
            nbits = int(main[1])
            if nbits not in (1, 2, 4, 8):
                raise ValueError(f"bad chunk width in '{spec}'")
            one_hot = len(main) > 2 and main[2] == 'oh'
            return OneHotCodecDef(
                nbits=nbits, one_hot=one_hot, bp=bp, ntokens=2 ** nbits,
                precision=precision, cap=cap)
        if spec.startswith('tokens.'):
            parts = spec.split('.')
            if len(parts) != 4 or parts[3] != 'oh':
                raise ValueError(
                    f"'{spec}': only tokens.N.variation.oh is a "
                    f"OneHotCodec spec; *.emb.X belongs to "
                    f"EmbeddingCodec (not implemented yet)")
            return OneHotCodecDef(
                nbits=None, one_hot=True, bp=False, ntokens=int(parts[1]),
                variation=parts[2], precision=precision, cap=cap)
        raise ValueError(f"Unknown input type: '{spec}'")

    def set_head_width(self, last_width: int) -> None:
        """Second construction phase: create the head once the layer
        chain's output width is known. Single logit for the binary
        case (the runtime pads the reference logit)."""
        head_size = self.ntokens if self.ntokens > 2 else 1
        self.head = DenseDef(head_size, input_size=last_width)

    def __str__(self) -> str:
        if self.nbits is None:
            return f'tokens.{self.ntokens}.{self.variation}.oh'
        if self.nbits == 8 and self.one_hot and not self.bp:
            return 'bytes'
        s = f'bits.{self.nbits}'
        if self.one_hot:
            s += '.oh'
        if self.bp:
            s += '+bp'
        return s

    @property
    def num_weights(self) -> int:
        return self.head.num_weights

    @property
    def num_mults(self) -> int:
        return self.head.num_weights

    def is_valid(self) -> bool:
        # The head is valid by construction (bare linear readout);
        # only the input encoding constrains. Tokenized inputs are
        # data-determined; bit-chunk inputs must be on the whitelist.
        if self.nbits is None:
            return self.ntokens >= 2
        return str(self) in _VALID_INPUTS

    def neighbors(self) -> tuple[str, ...]:
        """Input-spec strings one mutation away (the search ladder)."""
        return _INPUT_NEIGHBORS.get(str(self), ())

    def build_jax(self) -> OneHotCodecJax:
        return OneHotCodecJax(
            nbits=self.nbits, one_hot=self.one_hot, bp=self.bp,
            ntokens=self.ntokens,
            head=self.head.build_jax(self.precision.jax_dtype),
            cap=self.cap, dtype=self.precision.jax_dtype)
