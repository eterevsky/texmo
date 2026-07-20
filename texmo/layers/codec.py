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

_LOGIT_CAP = 30.0

# Stored per-token corpus knowledge a fold tokenset carries, counted
# as model weights so lossy sets compare honestly with bits/bytes
# (docs/io.md, "Fold tokensets"). Keyed by (ntokens, variation) --
# the codec never loads the tokenset file, so the count lives here.
# Uniform-bucket fold sets (tokens64) store nothing and default to 0.
_FOLD_EXTRA_WEIGHTS = {(32, 'raw_fold'): 32}


def fold_extra_weights(ntokens: int, variation: str | None) -> int:
    if variation is None:
        return 0
    return _FOLD_EXTRA_WEIGHTS.get((ntokens, variation), 0)


def cap_logits(logits: jax.Array) -> jax.Array:
    """Soft-cap logits to (-CAP, CAP)."""
    return _LOGIT_CAP * jnp.tanh(logits / _LOGIT_CAP)
