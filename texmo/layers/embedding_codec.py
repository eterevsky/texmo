"""EmbeddingCodec: the tied-embedding codec implementation.

One learned table serves both ends of the model (docs/tied_io.md):
on the way in, token ids look up their rows; on the way out, the
chain's LAST ACTIVATION is scored against the same rows directly,

    input   v_t = exp(y) * (E[token] + P[t mod npositions])
    output  logits = cap(h @ E.T)

- `E` is the (ntokens, X) value table; `P` is the (npositions, X)
  position table for sub-byte chunk inputs (positions are ADDED, so
  the `+bp`/`+bm` flags of OneHotCodec disappear; bytes and tokens
  have npositions == 1 and no P). Only value rows are scored at the
  output -- position rows there would cancel in the softmax.
- `exp(y)` is the learned input scale, initialized to sqrt(X)
  (y0 = ln(X)/2): rows sized to give O(1) logits have RMS ~1/sqrt(X),
  and the scale repairs the input-side magnitude -- Gemma's exact
  convention (inputs * sqrt(d), head unscaled).
- There is NO implicit adapter and no `.direct` mode: scoring is
  always direct, which requires the chain's final width to equal X.
  A model that wants a projection before the table spells it as an
  explicit trailing bare `dense.X` -- the exact same bias-linear map
  an implicit adapter would have been, but visible in the spec and
  reachable by the ordinary dense mutations. (OneHotCodec keeps its
  implicit dense head: a fixed codebook has nothing to score
  against, so there the head is the only learnable output map.)

The X == final-width constraint is checked by `is_valid`, not
repaired by the parser: a spec whose X disagrees with its chain is
simply an invalid model, the same as any other inconsistent spec.
Keeping X in sync belongs to structure MUTATIONS:
`Model2Def._sync_emb_width` re-spells the input at the mutated
chain's final width when neighbor generation resizes the last layer.
Width-preserving chains (the Gemma shape: the whole stack inherits
its width FROM the input) satisfy the constraint for any X, so X is
load-bearing exactly where it must be.

Input specs (X >= 1; power of 2 required only for search validity,
so e.g. tokens.256000.gemma.emb.2560 parses and runs but is
load-only):

    bytes.emb.{X}                       = bits.8
    bits.{1|2|4}.emb.{X}                npositions = 8/N
    tokens.{ntokens}.{variation}.emb.{X}

The binary case (bits.1) scores against its 2 value rows and emits 2
logits directly -- no single-logit padding trick. The empty chain
(`bytes.emb.4|`) is allowed: it is the symmetric bigram
(logits_j ~ (E_t + P_p) . E_j), handicapped on real text but not
broken.

Weights layout: slot 0 holds {'emb', 'pos'?, 'y'}; slot 2 is always
None (no head parameters) -- the model pytree keeps its
[input, seq, head] shape.

JAX only.
"""
import math

import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..precision import Precision
from .codec import cap_logits

# Mode swaps back to the blessed fixed-codebook inputs, per chunk
# width. Input swaps carry the chain unchanged, so the final width --
# and hence the emb X -- is carried unchanged too.
_OH_SWAPS = {
    1: ('bits.1+bp', 'bits.1+bm'),
    2: ('bits.2.oh+bp',),
    4: ('bits.4.oh+bp',),
    8: ('bytes',),
}
# The emb domain ladder, mirroring the one-hot 1 <-> 2 <-> 4 <-> 8
# ladder at the same table width.
_DOMAIN_LADDER = {1: (2,), 2: (1, 4), 4: (2, 8), 8: (4,)}


def _domain_spec(nbits: int, emb_size: int) -> str:
    if nbits == 8:
        return f'bytes.emb.{emb_size}'
    return f'bits.{nbits}.emb.{emb_size}'


class TiedHead:
    """Head metadata stand-in for `Model2Def.output` consumers.

    The tied head has no parameters of its own (its matrix is the
    slot-0 table), but managers/predictors read `output.size` (logit
    count), `output.input_size` and `output.num_weights` -- this
    carries them.
    """

    def __init__(self, size: int, input_size: int):
        self.size = size
        self.input_size = input_size
        self.num_weights = 0


class EmbeddingCodecJax:
    """Runtime for the tied-embedding codec."""

    def __init__(
        self,
        *,
        ntokens: int,
        npositions: int,
        emb_size: int,
        cap: bool,
        dtype,
    ):
        self.ntokens = ntokens
        self.npositions = npositions
        self.size = emb_size
        self.cap = cap
        self.dtype = dtype

    # -- weights ----------------------------------------------------------

    def init_input_weights(self, rng: jax.Array):
        k_emb, k_pos = jax.random.split(rng)
        # Rows ~ N(0, 1/X) per element: h @ E.T logits come out O(1),
        # and the exp(y) input scale restores O(1) input activations.
        scale = 1.0 / math.sqrt(self.size)
        w = {
            'emb': scale * jax.random.normal(
                k_emb, (self.ntokens, self.size), dtype=self.dtype),
            'y': jnp.asarray(0.5 * math.log(self.size), dtype=self.dtype),
        }
        if self.npositions > 1:
            w['pos'] = scale * jax.random.normal(
                k_pos, (self.npositions, self.size), dtype=self.dtype)
        return w

    def init_head_weights(self, rng: jax.Array) -> None:
        # The head is the slot-0 table; no parameters of its own.
        return None

    # -- input side -------------------------------------------------------

    def init_state(self) -> int | None:
        if self.npositions > 1:
            return 0
        return None

    def initial_vector(self, weights, position: int = -1) -> jax.Array:
        """'No token observed yet': zero value embedding, with the
        position embedding continuing the cycle into the negative
        direction (padding counts down toward the first real chunk)."""
        if self.npositions > 1:
            v = weights['pos'][position % self.npositions]
        else:
            v = jnp.zeros((self.size,), dtype=self.dtype)
        return jnp.exp(weights['y']) * v

    def encode(
        self, weights, tokens: jax.Array, padding: int = 0,
    ) -> jax.Array:
        batch, seq_len = tokens.shape
        out = weights['emb'][tokens]
        if self.npositions > 1:
            reps = (seq_len + self.npositions - 1) // self.npositions
            pos = jnp.tile(weights['pos'], (reps, 1))[:seq_len]
            out = out + pos  # broadcasts over the batch dim
        out = jnp.exp(weights['y']) * out
        if padding > 0:
            pad_vecs = jnp.stack(
                [self.initial_vector(weights, position=-padding + i)
                 for i in range(padding)])
            pad = jnp.broadcast_to(pad_vecs, (batch, padding, self.size))
            out = jnp.concatenate([pad, out], axis=1)
        return out

    def encode_step(self, weights, state, token) -> tuple:
        out = weights['emb'][token]
        if self.npositions > 1:
            out = out + weights['pos'][state]
            state = (state + 1) % self.npositions
        return state, jnp.exp(weights['y']) * out

    # -- output side ------------------------------------------------------

    def logits(self, input_weights, weights, h: jax.Array) -> jax.Array:
        """(..., X) last activations -> (..., ntokens) logits scored
        against the value rows of the shared table. `weights` (the
        head slot) is always None here; the parameter keeps the codec
        API uniform."""
        out = h @ input_weights['emb'].T
        if self.cap:
            out = cap_logits(out)
        return out

    def logits_step(self, input_weights, weights, h: jax.Array) -> jax.Array:
        return self.logits(input_weights, weights, h)


class EmbeddingCodecDef:
    """Descriptor for the tied-embedding codec.

    Same two-phase construction as OneHotCodecDef: `from_spec` fixes
    the chain input width (X), `set_head_width` records the chain's
    final width for the X == final-width consistency check in
    `is_valid`.
    """

    def __init__(
        self,
        *,
        nbits: int | None,
        ntokens: int,
        emb_size: int,
        variation: str | None = None,
        precision: Precision = Precision.FP32,
        cap: bool = True,
    ):
        self.nbits = nbits
        self.ntokens = ntokens
        self.emb_size = emb_size
        self.variation = variation
        self.precision = precision
        self.cap = cap
        self.size = emb_size  # width fed to the layer chain
        self.head: TiedHead | None = None
        self.last_width: int | None = None

        if nbits is None:
            self.npositions = 1
            self.tokens_name = f'tokens{ntokens}_{variation}'
        else:
            self.npositions = 8 // nbits if nbits < 8 else 1
            self.tokens_name = 'bytes' if nbits == 8 else f'bits.{nbits}'

    @staticmethod
    def from_spec(
        spec: str,
        precision: Precision = Precision.FP32,
        cap: bool = True,
    ) -> 'EmbeddingCodecDef':
        parts = spec.split('.')
        if parts[0] == 'tokens':
            if len(parts) != 5 or parts[3] != 'emb':
                raise ValueError(f"bad input spec: '{spec}'")
            return EmbeddingCodecDef(
                nbits=None, ntokens=int(parts[1]), emb_size=int(parts[4]),
                variation=parts[2], precision=precision, cap=cap)
        if parts[0] == 'bytes':
            if len(parts) != 3 or parts[1] != 'emb':
                raise ValueError(f"bad input spec: '{spec}'")
            nbits, emb_size = 8, int(parts[2])
        elif parts[0] == 'bits':
            if len(parts) != 4 or parts[2] != 'emb':
                raise ValueError(f"bad input spec: '{spec}'")
            nbits, emb_size = int(parts[1]), int(parts[3])
            if nbits not in (1, 2, 4, 8):
                raise ValueError(f"bad chunk width in '{spec}'")
        else:
            raise ValueError(f"Unknown input type: '{spec}'")
        if emb_size < 1:
            raise ValueError(f"bad embedding size in '{spec}'")
        return EmbeddingCodecDef(
            nbits=nbits, ntokens=2 ** nbits, emb_size=emb_size,
            precision=precision, cap=cap)

    def with_emb_size(self, emb_size: int) -> 'EmbeddingCodecDef':
        """Copy with a different table width. Used by neighbor
        generation to re-spell the input when a structure mutation
        changed the chain's final width."""
        return EmbeddingCodecDef(
            nbits=self.nbits, ntokens=self.ntokens, emb_size=emb_size,
            variation=self.variation, precision=self.precision,
            cap=self.cap)

    def set_head_width(self, last_width: int) -> None:
        self.last_width = last_width
        self.head = TiedHead(size=self.ntokens, input_size=last_width)

    def __str__(self) -> str:
        if self.nbits is None:
            return (f'tokens.{self.ntokens}.{self.variation}'
                    f'.emb.{self.emb_size}')
        if self.nbits == 8:
            return f'bytes.emb.{self.emb_size}'
        return f'bits.{self.nbits}.emb.{self.emb_size}'

    @property
    def num_weights(self) -> int:
        pos = self.npositions if self.npositions > 1 else 0
        table = (self.ntokens + pos) * self.emb_size
        return table + 1  # +1: the exp(y) input scale

    @property
    def num_mults(self) -> int:
        # Table lookups are free; the head costs the scoring matmul.
        return self.emb_size * self.ntokens

    def is_valid(self) -> bool:
        # Power-of-2 X is a search rule, not a parse rule: off-grid
        # widths (tokens.256000.gemma.emb.2560) are load-only.
        if self.ntokens < 2:
            return False
        if not is_power2_int(self.emb_size):
            return False
        # Direct scoring: the chain must end at the table width.
        return self.last_width == self.emb_size

    def neighbors(self, last_width: int) -> tuple[str, ...]:
        """Input-spec strings one mutation away: the mode swap back to
        the fixed-codebook input(s) and the emb domain ladder at the
        same table width. X itself has no independent mutations -- it
        follows the chain's final width (`last_width` equals
        `emb_size` on any valid model, so it isn't consulted here;
        the parameter keeps the codec API uniform)."""
        if self.nbits is None:
            return (f'tokens.{self.ntokens}.{self.variation}.oh',)
        nbs = list(_OH_SWAPS[self.nbits])
        nbs += [_domain_spec(n, self.emb_size)
                for n in _DOMAIN_LADDER[self.nbits]]
        return tuple(nbs)

    def build_jax(self) -> EmbeddingCodecJax:
        return EmbeddingCodecJax(
            ntokens=self.ntokens, npositions=self.npositions,
            emb_size=self.emb_size, cap=self.cap,
            dtype=self.precision.jax_dtype)
