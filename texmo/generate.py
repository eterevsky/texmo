"""Sampling a continuation from a built JAX model.

The generation loop itself, independent of Manager, Configuration and
dataset: it needs only the built `Model2Jax`, its weights and the name
of the tokenset the model was trained on. `ManagerJax.continue_prefix`
and the `generate` CLI are both thin wrappers around
`continue_prefix` here.
"""
import functools
import random

import jax

from .model2_jax import Model2Jax
from .tokens import get_tokenizer


@functools.lru_cache(maxsize=8)
def jit_step(model: Model2Jax):
    """The model's per-token inference step, JIT-compiled.

    Cached on the model object: without jit every `model.step` call
    pays JAX's dispatch overhead (~10 ms), which dominates for any
    reasonable continuation length, and without the cache each call
    of `continue_prefix` (once per temperature, typically) would
    recompile.
    """
    return jax.jit(model.step)


def sample_tokens(
    model: Model2Jax,
    weights,
    tokenizer,
    prefix: str,
    temperature: float,
):
    """Prefill `prefix`, then yield sampled token ids indefinitely.

    A generator, so the caller decides when to stop -- at a token
    count (`continue_prefix`) or at a turn boundary (`texmo.chat`).
    The states live in the frame between yields, so nothing is
    recomputed per token.
    """
    prefix_tokens = tokenizer.tokenize(prefix.encode())
    step = jit_step(model)
    states, _ = model.initial_step(weights)
    for c in prefix_tokens[:-1]:
        states, _ = step(weights, states, int(c))

    c = int(prefix_tokens[-1])
    rng = jax.random.PRNGKey(random.randrange(2**32))
    while True:
        states, logits = step(weights, states, c)
        probs = jax.nn.softmax(logits / temperature)
        rng, sub = jax.random.split(rng)
        c = int(jax.random.choice(sub, model.ntokens, p=probs))
        yield c


def continue_prefix(
    model: Model2Jax,
    weights,
    tokens_name: str,
    prefix: str,
    length: int,
    temperature: float,
) -> bytes:
    """Sample a `length`-token continuation of `prefix`.

    Returns the prefix and the continuation together, as bytes (the
    tokenization is byte-level, and a continuation can end mid
    character).
    """
    tokenizer = get_tokenizer(tokens_name)
    out = list(tokenizer.tokenize(prefix.encode()))
    tokens = sample_tokens(model, weights, tokenizer, prefix, temperature)
    for _ in range(length):
        out.append(next(tokens))

    return tokenizer.untokenize(out)
