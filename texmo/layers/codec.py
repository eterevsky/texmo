"""Shared pieces of the codec component -- the pairing that owns both
ends of a model: token ids to vectors at the front, hidden activations
to logits at the back. See docs/io.md for the concepts and
docs/tied_io.md for the design record.

Implementations (same API, chosen by the pre-`|` spec):

- `one_hot_codec.OneHotCodec{Def,Jax}` -- fixed codebook (one-hot or
  binary bits) in, independent learnable dense head out.
- `embedding_codec.EmbeddingCodec{Def,Jax}` (upcoming) -- learned
  embedding table tied between input and output (`*.emb.X[.direct]`).

This module holds what they share. Logit soft-capping:

    logits = CAP * tanh(logits / CAP),   CAP = 30

(the RecurrentGemma value). Monotone, so greedy sampling is unchanged;
bounds the maximum loss (~86 bits/token instead of inf) and kills
gradients on runaway logits. ON by default, validated against the
result DB (2026-07, ~500 seed-paired confs retrained both ways):
loss-neutral on healthy confs (rank of capped runs among DB losses
uniform, mean 0.504 +- 0.018; paired geometric-mean loss ratio
0.998 +- 0.004), no change in overall divergence rate (its mechanism
only tames runaway logits, not hidden-state NaNs), but it rescues the
logit-blow-up failure mode -- including confs that had never converged
in the DB -- and bounds the loss when divergence happens anyway.
`parse_model2(..., cap=False)` opts out for experiments and for
comparison with the legacy uncapped runtime.
"""
import jax
import jax.numpy as jnp

from ..tokens import get_tokenizer

_LOGIT_CAP = 30.0

# tokens_name -> stats['extra_weights'], memoized because num_weights
# is called constantly during search and must stay a dict lookup.
_EXTRA_WEIGHTS: dict[str, int] = {}


def tokenset_extra_weights(ntokens: int, variation: str | None) -> int:
    """Corpus knowledge baked into a tokenset, counted as model
    weights so sets that carry it compare honestly with bits/bytes
    (docs/io.md). A fold set stores per-token head frequencies; a
    hexbpe set stores its selected bytes and merge table. The number
    is the tokenset's own `stats.extra_weights`; sets that carry
    nothing (bit-chunk inputs, the shift set) have no such stat and
    cost 0.

    Raises RuntimeError when the tokenset can't be loaded. num_weights
    is part of a conf's identity across the whole fleet, so a silent 0
    on a machine with a missing tokens dir would fork that identity.
    """
    if variation is None:
        return 0
    name = f'tokens.{ntokens}.{variation}'
    extra = _EXTRA_WEIGHTS.get(name)
    if extra is not None:
        return extra
    tokenizer = get_tokenizer(name)
    if tokenizer is None:
        raise RuntimeError(
            f"can't load tokenset {name!r} (tokens dir unset or file "
            f"missing); its extra weights are part of the conf "
            f"identity, so guessing 0 would fork it across machines")
    extra = int((tokenizer.tokenset.stats or {}).get('extra_weights', 0))
    _EXTRA_WEIGHTS[name] = extra
    return extra


def cap_logits(logits: jax.Array) -> jax.Array:
    """Soft-cap logits to (-CAP, CAP)."""
    return _LOGIT_CAP * jnp.tanh(logits / _LOGIT_CAP)
