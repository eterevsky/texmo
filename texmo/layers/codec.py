"""Shared pieces of the codec component -- the pairing that owns both
ends of a model: token ids to vectors at the front, hidden activations
to logits at the back. See docs/io.md for the concepts and
docs/tied_io.md for the design record.

Implementations (same API, chosen by the pre-`|` spec):

- `one_hot_codec.OneHotCodec{Def,Jax}` -- fixed codebook (one-hot or
  binary bits) in, independent learnable dense head out.
- `embedding_codec.EmbeddingCodec{Def,Jax}` -- learned embedding
  table tied between input and output (`*.emb.X`).
- `pair_codec.PairCodec{Def,Jax}` -- whole-byte hex pair
  (`bits.4.pair.add` / `bits.4.pair.K`): two 16-way one-hots in, a
  16-way hi head plus a per-hi lo conditional out (additive or
  gated-multiplicative), composed into 256 byte log-probabilities.

This module holds what they share. Logits are the raw head output:
the logit soft-cap that lived here was removed 2026-08-19 (see
docs/io.md for the measurement that retired it).
"""
from ..tokens import get_tokenizer

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
