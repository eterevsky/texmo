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
gradients on runaway logits, so fewer runs should diverge. Currently
OPT-IN (`parse_model2(..., cap=True)`): the search DB's half-million
runs were trained uncapped, so the default stays off until the cap's
effect is measured against them; flip the default after that
experiment.
"""
import jax
import jax.numpy as jnp

_LOGIT_CAP = 30.0


def cap_logits(logits: jax.Array) -> jax.Array:
    """Soft-cap logits to (-CAP, CAP)."""
    return _LOGIT_CAP * jnp.tanh(logits / _LOGIT_CAP)
