"""PairCodec: the additive hex-pair codec -- `bits.4.pair.add`.

A WHOLE-BYTE codec (one position per byte, like `bytes`) that splits
each byte into its two hexadecimal digits, `hi = byte >> 4` and
`lo = byte & 15`, on both ends of the model:

    input   v_t = concat(onehot16(hi), onehot16(lo))     width 32
    output  t = W1 @ h + b1                              16 hi logits
            s = W2 @ h + b2, offset by D[:, hi]          16 lo logits

The output is a factorized 256-way byte distribution:

    P(byte) = P(hi) * P(lo | hi)
    P(hi)      = softmax(t)[hi]
    P(lo | hi) = softmax(s + D[:, hi])[lo]

`D` is a (16, 16) coupling matrix -- `D[:, hi]` is the additive offset
over `lo` given `hi`. It is the static byte inventory: which low
nibbles follow which high nibble in this corpus at all (the ASCII
letter block, the digit block, the UTF-8 continuation range). What it
cannot express is CONTEXT-dependent hi/lo coupling, since the offset
does not depend on `h`; that is the measurable gap of the additive
arm (see docs/io.md), and the escalation if it bites is a small
nonlinear mixer on the second head.

The `.add` in the name is load-bearing: it marks the ADDITIVE arm of
the pair family, the k=0-style baseline. Its sibling
`bits.4.pair.K` -- pure-multiplicative coupling through k shared
gated channels -- is a separate kind with its own spec, so the
family name is never spelled bare: `bits.4.pair` on its own is a
parse error rather than a silent default to this arm.

**The contract stays unchanged.** Rather than exposing two heads to
the training loop, the codec COMPOSES them into ordinary 256-way byte
logits:

    logits[hi*16 + lo] = log_softmax(t)[hi]
                       + log_softmax(s + D[:, hi])[lo]

Those 256 numbers are already log-probabilities (they sum to 1 in
probability space), so `logsumexp(logits) == 0`. Everything
downstream -- cross-entropy, sampling, eval, generate -- works
verbatim: by the chain rule the softmax cross-entropy of a byte
against these logits is EXACTLY `CE(hi) + CE(lo | true hi)`, so
teacher forcing comes free, and sampling from the composed 256-way
softmax is exactly ancestral sampling of the pair. No two-step
dispatch anywhere in the runtime.

Weights (all in the head slot, slot 2; the input side is a fixed
parameter-free codebook and keeps None in slot 0):

    w1 (16, X), b1 (16,)     the hi head
    w2 (16, X), b2 (16,)     the lo base head
    d  (16, 16)              the hi -> lo coupling, zero-initialized

    num_weights = 2*16*X + 16*16 + 32 = 32X + 288

Against the 256-way head it replaces (`256X + 256`), the crossover is
at X = 1/7: the pair head is cheaper at every width the search can
reach, by 4.2x at X = 8 (544 vs 2304), 6.4x at X = 32 (1312 vs 8448),
and 8x asymptotically. The 288 constant is the price of the fixed
`16x16 + biases` block, which no width amortizes away.

Tokens are raw bytes (`tokens_name = 'bytes'`, the same BytesTokenizer
`bytes` uses): lossless, 1.0 bytes per token, no residual charge and
none of the `+bp` phase bookkeeping the sub-byte `bits.N` kinds need.
There is no `+bp`/`+bm` suffix and no `emb` variant -- both are
parse errors, as is the bare family name.
"""
import jax
import jax.numpy as jnp

from ..layer_jax import xavier_uniform
from ..precision import Precision

# One hex digit: 16 values per position, two positions per byte.
_NHEX = 16
# Fixed sizes derived from _NHEX: 32-wide input, 256-way output.
_INPUT_SIZE = 2 * _NHEX
_NTOKENS = _NHEX * _NHEX

# The only spelling. `+bp`/`+bm` have no meaning here (one position
# per byte), there is no tied/embedded variant, and the bare family
# name `bits.4.pair` is NOT accepted: the multiplicative sibling
# `bits.4.pair.K` shares the family, so the arm is always explicit.
_SPEC = 'bits.4.pair.add'
_FAMILY = 'bits.4.pair'

# The IO ladder, v1: a single toggle bridge to the one-hot hex input
# (`bits.4.oh+bp`), which is the same 16-symbol alphabet read
# sub-byte. The reverse edge lives in
# one_hot_codec._INPUT_NEIGHBORS; keep the two in sync.
_NEIGHBORS = ('bits.4.oh+bp',)


class PairHead:
    """Head metadata stand-in for `Model2Def.output` consumers.

    The pair head is not a plain dense, but managers/predictors read
    `output.size` (logit count), `output.input_size` and
    `output.num_weights` -- this carries them. Mirrors
    `embedding_codec.TiedHead`, except the parameters really do live
    in the head slot, so `num_weights` is nonzero (which is also how
    the loss predictor duck-types "not tied").
    """

    def __init__(self, input_size: int):
        self.size = _NTOKENS
        self.input_size = input_size
        self.num_weights = 2 * _NHEX * input_size + _NHEX * _NHEX + 2 * _NHEX


class PairCodecJax:
    """Runtime for the additive hex-pair codec."""

    def __init__(self, *, last_width: int, dtype):
        self.ntokens = _NTOKENS
        self.size = _INPUT_SIZE
        self.last_width = last_width
        self.dtype = dtype
        ids = jnp.arange(_NTOKENS)
        self.encodings = jnp.concatenate(
            [jax.nn.one_hot(ids // _NHEX, _NHEX, dtype=dtype),
             jax.nn.one_hot(ids % _NHEX, _NHEX, dtype=dtype)],
            axis=-1)

    # -- weights ----------------------------------------------------------

    def init_input_weights(self, rng: jax.Array) -> None:
        # Fixed codebook: no learned input weights. The slot stays in
        # the pytree (None) so the layout matches the other codecs.
        return None

    def init_head_weights(self, rng: jax.Array):
        k1, k2 = jax.random.split(rng)
        return {
            'w1': xavier_uniform(
                k1, (_NHEX, self.last_width), dtype=self.dtype),
            'b1': jnp.zeros(_NHEX, dtype=self.dtype),
            'w2': xavier_uniform(
                k2, (_NHEX, self.last_width), dtype=self.dtype),
            'b2': jnp.zeros(_NHEX, dtype=self.dtype),
            # Zero coupling to start: the conditional is softmax(s)
            # for every hi, and loss pressure differentiates it.
            'd': jnp.zeros((_NHEX, _NHEX), dtype=self.dtype),
        }

    # -- input side -------------------------------------------------------

    def init_state(self) -> None:
        """One position per byte: nothing to remember between steps."""
        return None

    def initial_vector(self, weights, position: int = -1) -> jax.Array:
        """Input vector for 'no byte observed yet' (max entropy):
        uniform 1/16 within each hex half. `position` is accepted for
        API uniformity and ignored -- there is no position cycle."""
        return jnp.full((_INPUT_SIZE,), 1.0 / _NHEX, dtype=self.dtype)

    def encode(
        self, weights, tokens: jax.Array, padding: int = 0,
    ) -> jax.Array:
        """Encode a batch of byte sequences.

        Args:
            tokens: (batch, seq_len) int array of byte values.
            padding: initial positions to prepend (max-entropy input).

        Returns:
            (batch, seq_len + padding, 32).
        """
        batch = tokens.shape[0]
        out = self.encodings[tokens]
        if padding > 0:
            pad = jnp.broadcast_to(
                self.initial_vector(weights),
                (batch, padding, _INPUT_SIZE))
            out = jnp.concatenate([pad, out], axis=1)
        return out

    def encode_step(self, weights, state, token) -> tuple:
        """Encode a single byte, returning (new_state, vector)."""
        return state, self.encodings[token]

    # -- output side ------------------------------------------------------
    #
    # Both logits methods take the input-side weights first (unused
    # here) so the call is uniform across codecs.

    def _compose(self, weights, h: jax.Array) -> jax.Array:
        """(..., X) -> (..., 256) composed log-probabilities."""
        t = h @ weights['w1'].T + weights['b1']          # (..., 16) hi
        s = h @ weights['w2'].T + weights['b2']          # (..., 16) lo base
        lsm_hi = jax.nn.log_softmax(t, axis=-1)          # (..., 16)
        # d[lo, hi] is the offset on `lo` given `hi`, so d.T is the
        # (hi, lo) grid of offsets that broadcasts onto s.
        lsm_lo = jax.nn.log_softmax(
            s[..., jnp.newaxis, :] + weights['d'].T, axis=-1)  # (..., 16, 16)
        grid = lsm_hi[..., jnp.newaxis] + lsm_lo         # (..., hi, lo)
        return grid.reshape(grid.shape[:-2] + (_NTOKENS,))

    def logits(self, input_weights, weights, h: jax.Array) -> jax.Array:
        """(..., X) hidden activations -> (..., 256) byte logits.

        These are normalized log-probabilities, not raw scores: the
        composition is where the chain rule happens, so a plain
        softmax cross-entropy over them equals CE(hi) + CE(lo | hi).
        """
        return self._compose(weights, h)

    def logits_step(self, input_weights, weights, h: jax.Array) -> jax.Array:
        return self._compose(weights, h)


class PairCodecDef:
    """Descriptor for the additive hex-pair codec.

    Same two-phase construction as the other codecs: `from_spec`
    fixes the chain input width (a constant 32 here), then
    `set_head_width(last_width)` builds the head once the chain's
    final width is known. Both calls live in
    `spec_parser.parse_model2`.
    """

    def __init__(self, *, precision: Precision = Precision.FP32):
        self.precision = precision
        self.ntokens = _NTOKENS
        self.size = _INPUT_SIZE
        # Raw bytes, one token per byte: the same tokenizer `bytes`
        # uses. Lossless, 1.0 bytes/token, no extra_weights.
        self.tokens_name = 'bytes'
        self.variation = None
        self.head: PairHead | None = None
        self.last_width: int | None = None

    @staticmethod
    def from_spec(
        spec: str,
        precision: Precision = Precision.FP32,
    ) -> 'PairCodecDef':
        if spec != _SPEC:
            if spec == _FAMILY:
                raise ValueError(
                    f"'{spec}' names the pair FAMILY, not a codec: "
                    f"spell the arm explicitly ('{_SPEC}' for the "
                    f"additive coupling, 'bits.4.pair.K' for the "
                    f"multiplicative one)")
            raise ValueError(
                f"bad input spec: '{spec}'; the additive hex-pair "
                f"codec has exactly one spelling, '{_SPEC}' (no "
                f"+bp/+bm -- one position per byte -- and no emb "
                f"variant)")
        return PairCodecDef(precision=precision)

    def set_head_width(self, last_width: int) -> None:
        self.last_width = last_width
        self.head = PairHead(input_size=last_width)

    def __str__(self) -> str:
        return _SPEC

    @property
    def num_weights(self) -> int:
        # 32X + 288: two 16-way heads plus the 16x16 coupling. Far
        # below the 256X + 256 of the byte head it replaces at every
        # reachable width (see the module docstring). Raw bytes carry
        # no stored corpus knowledge, so there is nothing to add.
        return self.head.num_weights

    @property
    def num_mults(self) -> int:
        return self.head.num_weights

    def is_valid(self) -> bool:
        # No metaparameters: the spec is a constant and the head
        # follows the chain's final width, so validity is purely
        # structural -- anything that parsed is valid.
        return True

    def neighbors(self, last_width: int) -> tuple[str, ...]:
        """Input-spec strings one mutation away. v1 is deliberately
        minimal: the toggle to the one-hot hex input at the same
        16-symbol alphabet. `last_width` is unused (no width follows
        the chain here); the parameter keeps the codec API uniform."""
        return _NEIGHBORS

    def build_jax(self) -> PairCodecJax:
        return PairCodecJax(
            last_width=self.last_width, dtype=self.precision.jax_dtype)
