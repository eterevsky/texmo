"""PairCodec: the hex-pair codec family -- `bits.4.pair.{add,K}`.

A WHOLE-BYTE codec (one position per byte, like `bytes`) that splits
each byte into its two hexadecimal digits, `hi = byte >> 4` and
`lo = byte & 15`, on both ends of the model:

    input   v_t = concat(onehot16(hi), onehot16(lo))     width 32
    output  t = W1 @ h + b1                              16 hi logits
            u(hi)                                        16 lo logits

The output is a factorized 256-way byte distribution:

    P(byte) = P(hi) * P(lo | hi)
    P(hi)      = softmax(t)[hi]
    P(lo | hi) = softmax(u(hi))[lo]

The two arms differ ONLY in how `u(hi)` is formed. Everything else --
the input codebook, the hi head, the composition into 256 logits --
is shared code, which is why they live in one class.

**Additive arm, `bits.4.pair.add`**:

    u(hi) = W2 @ h + b2 + D[:, hi]        D is (16, 16)

`D[:, hi]` is a STATIC additive offset over `lo`. It learns the byte
inventory -- which low nibbles follow which high nibble in this
corpus at all (the ASCII letter blocks, the digit block, the UTF-8
continuation range) -- but it never sees `h`, so it cannot express
CONTEXT-dependent hi/lo coupling. That gap is what the other arm
attacks.

**Multiplicative arm, `bits.4.pair.K`** (K a literal integer, e.g.
`bits.4.pair.16`):

    u(hi) = U @ ((V @ h + b_v) * A[:, hi]) + b_u

k shared channels, gated per high nibble: `V` reads k features off
`h`, `A[:, hi]` scales them for this hi, `U` writes the result back
onto the 16 lo logits. There is NO W2, NO b2 and NO D -- the coupling
is purely multiplicative, which is what makes it context-dependent:
the effect of `hi` on `lo` now depends on `h`.

The gate is raw bilinear, with no activation -- matching `split.mul`
and mullstm, and keeping the 256-grid closed-form (see `_lo_logits`).

**The biases here are load-bearing, not decoration.** `b_v * A[:, hi]`
is an `h`-independent term, so it reproduces exactly the rank-k
static coupling that `D` is in the additive arm; `b_u` is the lo
prior. Take them away and the arm loses the additive arm's whole
capability rather than generalizing it. There is deliberately NO
bias on the A side: a channel whose `A` row is near-constant across
hi already emulates an h-linear, hi-independent term, so an explicit
A-bias is a marginal freedom we skip.

**The contract stays unchanged** in both arms. Rather than exposing
two sequential heads to the training loop, the codec COMPOSES them
into ordinary 256-way byte logits:

    logits[hi*16 + lo] = log_softmax(t)[hi] + log_softmax(u(hi))[lo]

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

    both    w1 (16, X), b1 (16,)          the hi head
    .add    w2 (16, X), b2 (16,)          the lo base head
            d  (16, 16)                   static coupling, zero-init
    .K      v  (k, X), b_v (k,)           channel readout
            a  (k, 16)                    per-hi gate
            u  (16, k), b_u (16,)         channel writeback

    .add    num_weights = 2*16*X + 16*16 + 32 = 32X + 288
    .K      num_weights = (16 + k)*X + 33k + 32

At k = 16 the two are weight-comparable and share the same X-slope
(32X + 288 vs 32X + 560), which is why that is the k the arm-toggle
neighbor edge uses.

Against the 256-way head both replace (`256X + 256`), `.add`'s
crossover is at X = 1/7: cheaper at every width the search can reach,
by 4.2x at X = 8 (544 vs 2304), 6.4x at X = 32 (1312 vs 8448), and 8x
asymptotically. The fixed `16x16 + biases` constant is what no width
amortizes away.

Tokens are raw bytes (`tokens_name = 'bytes'`, the same BytesTokenizer
`bytes` uses): lossless, 1.0 bytes per token, no residual charge and
none of the `+bp` phase bookkeeping the sub-byte `bits.N` kinds need.
There is no `+bp`/`+bm` suffix and no `emb` variant -- both are parse
errors, as is the bare family name `bits.4.pair`: the arm is always
explicit, so nothing silently defaults.
"""
import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer_jax import xavier_uniform
from ..precision import Precision

# One hex digit: 16 values per position, two positions per byte.
_NHEX = 16
# Fixed sizes derived from _NHEX: 32-wide input, 256-way output.
_INPUT_SIZE = 2 * _NHEX
_NTOKENS = _NHEX * _NHEX

# Spellings. `+bp`/`+bm` have no meaning here (one position per byte),
# there is no tied/embedded variant, and the bare family name is NOT
# accepted: both arms share it, so neither may claim it by default.
_SPEC_ADD = 'bits.4.pair.add'
_FAMILY = 'bits.4.pair'

# k floor: below 4 channels the lo digit barely sees `h` at all, and
# the arm degenerates toward a worse-parameterized `.add`. No upper
# cap -- max_weights prunes the top end like any other width.
_MIN_K = 4
# The k at which `.K` is weight-comparable with `.add` (32X + 560 vs
# 32X + 288, same X-slope), hence the only k that carries the
# arm-toggle edge and the bridge down to the one-hot ladder.
_TOGGLE_K = 16

# The IO ladder, out of the family.
#
# `bits.4.oh+bp` is the same 16-symbol alphabet read sub-byte.
#
# `tokens.32.hexbpe.oh` is the other genuine neighbor: 32-wide IO on
# both sides, and hexbpe's own encoding is a MIX of whole-byte tokens
# and hex-digit fallbacks -- exactly the two granularities this codec
# holds at once. The representations are adjacent enough that the
# search should be able to cross between them in one step rather than
# routing through the bit ladder.
#
# The reverse edges live in one_hot_codec._INPUT_NEIGHBORS; keep the
# two sides in sync.
_OH_LADDER = 'bits.4.oh+bp'
_HEXBPE_LADDER = 'tokens.32.hexbpe.oh'
_OUT_OF_FAMILY = (_OH_LADDER, _HEXBPE_LADDER)


class PairHead:
    """Head metadata stand-in for `Model2Def.output` consumers.

    The pair head is not a plain dense, but managers/predictors read
    `output.size` (logit count), `output.input_size` and
    `output.num_weights` -- this carries them. Mirrors
    `embedding_codec.TiedHead`, except the parameters really do live
    in the head slot, so `num_weights` is nonzero (which is also how
    the loss predictor duck-types "not tied").

    `k is None` marks the additive arm.
    """

    def __init__(self, input_size: int, k: int | None):
        self.size = _NTOKENS
        self.input_size = input_size
        self.k = k
        if k is None:
            # w1 + w2 (2*16*X), d (16*16), b1 + b2 (32).
            self.num_weights = (
                2 * _NHEX * input_size + _NHEX * _NHEX + 2 * _NHEX)
        else:
            # w1 (16X) + b1 (16) + v (kX) + b_v (k) + a (16k)
            # + u (16k) + b_u (16)  =  (16 + k)X + 33k + 32.
            self.num_weights = (
                (_NHEX + k) * input_size + (2 * _NHEX + 1) * k + 2 * _NHEX)


class PairCodecJax:
    """Runtime for the hex-pair codec (both arms; `k is None` is
    the additive one)."""

    def __init__(self, *, k: int | None, last_width: int, dtype):
        self.k = k
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
        if self.k is None:
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
        k1, kv, ka, ku = jax.random.split(rng, 4)
        return {
            'w1': xavier_uniform(
                k1, (_NHEX, self.last_width), dtype=self.dtype),
            'b1': jnp.zeros(_NHEX, dtype=self.dtype),
            'v': xavier_uniform(
                kv, (self.k, self.last_width), dtype=self.dtype),
            # b_v starts at zero: the STATIC part of the coupling
            # begins flat (the additive arm's d = 0 start), while the
            # h-dependent part starts at the xavier scale like any
            # other head matrix. Gradients still reach b_v -- its
            # partial is u[:, j] * a[j, hi], nonzero for random u, a.
            'b_v': jnp.zeros(self.k, dtype=self.dtype),
            'a': xavier_uniform(ka, (self.k, _NHEX), dtype=self.dtype),
            'u': xavier_uniform(ku, (_NHEX, self.k), dtype=self.dtype),
            'b_u': jnp.zeros(_NHEX, dtype=self.dtype),
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

    def _lo_logits(self, weights, h: jax.Array) -> jax.Array:
        """(..., X) -> (..., 16 hi, 16 lo): the raw conditional
        logits `u(hi)` for every high nibble at once. This is the only
        place the two arms differ."""
        if self.k is None:
            # Additive: a shared lo head plus a static per-hi offset.
            # d[lo, hi] is the offset on `lo` given `hi`, so d.T is the
            # (hi, lo) grid of offsets that broadcasts onto s.
            s = h @ weights['w2'].T + weights['b2']      # (..., 16) lo base
            return s[..., jnp.newaxis, :] + weights['d'].T
        # Multiplicative: k channels read off h, gated per hi, written
        # back onto the lo logits. Closed form for all 16 his at once
        # -- no per-hi loop, no second dispatch:
        #     out[..., hi, lo] = sum_j u[lo, j] * g[..., j] * a[j, hi]
        # which is exactly u @ (diag(g) @ a) with the (hi, lo) axes in
        # the order `_compose` wants.
        g = h @ weights['v'].T + weights['b_v']          # (..., k)
        return jnp.einsum(
            '...j,jh,lj->...hl', g, weights['a'], weights['u'],
        ) + weights['b_u']

    def _compose(self, weights, h: jax.Array) -> jax.Array:
        """(..., X) -> (..., 256) composed log-probabilities."""
        t = h @ weights['w1'].T + weights['b1']          # (..., 16) hi
        lsm_hi = jax.nn.log_softmax(t, axis=-1)          # (..., 16)
        lsm_lo = jax.nn.log_softmax(
            self._lo_logits(weights, h), axis=-1)        # (..., 16, 16)
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
    """Descriptor for the hex-pair codec, both arms.

    `k is None` is the additive arm (`bits.4.pair.add`); an integer k
    is the multiplicative one (`bits.4.pair.{k}`).

    Same two-phase construction as the other codecs: `from_spec`
    fixes the chain input width (a constant 32 here), then
    `set_head_width(last_width)` builds the head once the chain's
    final width is known. Both calls live in
    `spec_parser.parse_model2`.
    """

    def __init__(
        self,
        *,
        k: int | None = None,
        precision: Precision = Precision.FP32,
    ):
        self.k = k
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
        """Parse `bits.4.pair.add` or `bits.4.pair.{k}`.

        Unlike the dense/emb widths -- where the power-of-2 rule is a
        search rule and off-grid values stay load-only -- k is
        rejected outright here. There is no legacy population spelled
        with an off-grid k and no pretrained model to load, so a bad k
        is a typo, and the family's strict single-spelling style says
        to fail on it rather than build a model that `is_valid` will
        silently drop.
        """
        if spec == _SPEC_ADD:
            return PairCodecDef(k=None, precision=precision)
        if spec == _FAMILY:
            raise ValueError(
                f"'{spec}' names the pair FAMILY, not a codec: spell "
                f"the arm explicitly -- '{_SPEC_ADD}' for the additive "
                f"coupling, '{_FAMILY}.{_TOGGLE_K}' (or any other k) "
                f"for the multiplicative one")
        arm = (spec[len(_FAMILY) + 1:] if spec.startswith(_FAMILY + '.')
               else '')
        if not arm.isdigit():
            raise ValueError(
                f"bad input spec: '{spec}'; the hex-pair codec is "
                f"spelled '{_SPEC_ADD}' or '{_FAMILY}.K' with K an "
                f"integer (no +bp/+bm -- one position per byte -- and "
                f"no emb variant)")
        k = int(arm)
        if not is_power2_int(k) or k < _MIN_K:
            raise ValueError(
                f"'{spec}': k must be a power of two >= {_MIN_K}; "
                f"below that the lo digit barely sees the hidden "
                f"state. No upper cap -- max_weights prunes the top.")
        return PairCodecDef(k=k, precision=precision)

    def set_head_width(self, last_width: int) -> None:
        self.last_width = last_width
        self.head = PairHead(input_size=last_width, k=self.k)

    def __str__(self) -> str:
        if self.k is None:
            return _SPEC_ADD
        return f'{_FAMILY}.{self.k}'

    @property
    def num_weights(self) -> int:
        # .add: 32X + 288 (two 16-way heads + the 16x16 coupling).
        # .K:   (16 + k)X + 33k + 32.
        # Both far below the 256X + 256 of the byte head they replace
        # at every reachable width (see the module docstring). Raw
        # bytes carry no stored corpus knowledge, so nothing to add.
        return self.head.num_weights

    @property
    def num_mults(self) -> int:
        return self.head.num_weights

    def is_valid(self) -> bool:
        # Validity is purely structural: the head follows the chain's
        # final width, and k was already checked at parse time (see
        # `from_spec`), so anything that parsed is valid.
        return True

    def neighbors(self, last_width: int) -> tuple[str, ...]:
        """Input-spec strings one mutation away.

        Both arms bridge out of the family to the two adjacent
        32-wide representations (`_OUT_OF_FAMILY`) and across to each
        other. The multiplicative arm carries those cross-family
        edges and the arm toggle only at the weight-comparable
        k = _TOGGLE_K; every k additionally moves along the k ladder
        by doubling/halving -- the same 2x rule layer widths use,
        with a floor at _MIN_K and no cap.

        `last_width` is unused (no width follows the chain here); the
        parameter keeps the codec API uniform.
        """
        if self.k is None:
            return _OUT_OF_FAMILY + (f'{_FAMILY}.{_TOGGLE_K}',)
        nbs = []
        if self.k == _TOGGLE_K:
            nbs += [*_OUT_OF_FAMILY, _SPEC_ADD]
        if self.k > _MIN_K:
            nbs.append(f'{_FAMILY}.{self.k // 2}')
        nbs.append(f'{_FAMILY}.{self.k * 2}')
        return tuple(nbs)

    def build_jax(self) -> PairCodecJax:
        return PairCodecJax(
            k=self.k, last_width=self.last_width,
            dtype=self.precision.jax_dtype)
