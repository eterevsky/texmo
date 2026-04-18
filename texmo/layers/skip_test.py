"""Tests for skip (residual) layer.

Most of the interesting behavior lives at the Model / ModelDef level;
see also model_test.py for end-to-end tests involving skips.
"""

import jax
import jax.numpy as jnp
import torch

from texmo.layers.skip import SkipDef, SkipJax, SkipModule


def test_def_basic():
    d = SkipDef(2, "add", input_size=8)
    assert d.distance == 2
    assert d.op == "add"
    assert d.size == 8  # pass-through
    assert d.num_weights == 0
    assert str(d) == "skip.2.add"
    assert d.is_valid()


def test_def_cat():
    d = SkipDef(1, "cat", input_size=4)
    assert str(d) == "skip.1.cat"
    assert d.is_valid()


def test_def_invalid_distance_zero():
    d = SkipDef(0, "add", input_size=8)
    assert not d.is_valid()


def test_def_invalid_op():
    d = SkipDef(1, "bogus", input_size=8)
    assert not d.is_valid()


def test_def_neighbors_only_op_swap():
    # Distance mutations are handled at the ModelDef level, not here.
    d = SkipDef(2, "add", input_size=8)
    assert list(d.neighbors()) == ["skip.2.cat"]

    d = SkipDef(3, "cat", input_size=8)
    assert list(d.neighbors()) == ["skip.3.add"]


def test_module_is_passthrough():
    m = SkipModule(8, distance=2, op="add")
    x = torch.randn(2, 5, 8)
    out = m(x)
    assert torch.equal(out, x)

    state, out = m.step(None, torch.randn(8))
    assert state is None


def test_jax_is_passthrough():
    layer = SkipJax(8, distance=2, op="add", dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 8), dtype=jnp.float32)
    out = layer.forward(None, x)
    assert jnp.array_equal(out, x)

    state, out = layer.step(None, None, x[0, 0])
    assert state is None
