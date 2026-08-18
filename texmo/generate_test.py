"""Tests for the standalone sampling loop.

Uses a tiny untrained model -- the output is noise, so the assertions
are about the contract (bytes out, prefix kept, RNG seeding), not the
text.
"""

import random

import jax

from texmo.generate import continue_prefix, jit_step
from texmo.precision import Precision
from texmo.spec_parser import parse_model2

SPEC = 'bits.4.oh+bp|dense.4.gelu'


def _make_model(spec: str = SPEC):
    md = parse_model2(spec, precision=Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    return model, weights, md.codec.tokens_name


def test_continue_prefix_returns_bytes():
    model, weights, tokens_name = _make_model()
    out = continue_prefix(
        model, weights, tokens_name, 'hi', length=8, temperature=1.0)
    assert isinstance(out, bytes)
    assert len(out) > 0
    assert out.startswith(b'hi')


def test_continue_prefix_deterministic_under_seed():
    """The sampling key comes from the `random` module, so seeding it
    pins the continuation."""
    model, weights, tokens_name = _make_model()
    random.seed(1234)
    first = continue_prefix(
        model, weights, tokens_name, 'hi', length=16, temperature=1.0)
    random.seed(1234)
    second = continue_prefix(
        model, weights, tokens_name, 'hi', length=16, temperature=1.0)
    assert first == second


def test_continue_prefix_length_grows_with_tokens():
    model, weights, tokens_name = _make_model()
    random.seed(7)
    short = continue_prefix(
        model, weights, tokens_name, 'hi', length=4, temperature=1.0)
    random.seed(7)
    long = continue_prefix(
        model, weights, tokens_name, 'hi', length=64, temperature=1.0)
    assert len(long) > len(short)


def test_jit_step_cached_per_model():
    model, _, _ = _make_model()
    other, _, _ = _make_model()
    assert jit_step(model) is jit_step(model)
    assert jit_step(model) is not jit_step(other)
