import jax.numpy as jnp
import numpy as np

from texmo.layers.norm import NormDef
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def test_def_properties():
    d = NormDef(input_size=16)
    assert d.size == 16
    assert str(d) == "norm"
    assert d.num_weights == 0
    assert d.is_valid()


def test_invalid_size1():
    assert not NormDef(input_size=1).is_valid()


# -- Model-level validity constraints --

def _md(spec):
    return parse_model2(spec, Precision.FP32)


def test_norm_first_layer_invalid():
    assert not _md("bytes|norm-dense.32.gelu").is_valid()


def test_two_consecutive_norms_invalid():
    assert not _md("bytes|dense.32.gelu-norm-norm-dense.16.tanh").is_valid()


def test_norm_after_suffix_invalid():
    assert not _md("bytes|dense.32.gelu-suffix.2-norm").is_valid()


def test_valid_norm_placement():
    md = _md("bytes|dense.32.gelu-norm-dense.16.tanh")
    assert md.is_valid()


def test_norm_size1_invalid():
    # bits.1 has size 1, so norm immediately after would be invalid.
    # But we can't put norm as the first layer anyway. Check via dense.1.
    assert not _md("bytes|dense.1.gelu-norm").is_valid()


# -- Neighbors --

def test_neighbors_insert_norm():
    md = _md("bytes|dense.32.gelu-dense.16.tanh")
    neighbor_specs = [str(n) for n in md.neighbors()]
    assert "bytes|dense.32.gelu-norm-dense.16.tanh" in neighbor_specs
    # But not before the first layer
    assert "bytes|norm-dense.32.gelu-dense.16.tanh" not in neighbor_specs


def test_neighbors_remove_norm():
    md = _md("bytes|dense.32.gelu-norm-dense.16.tanh")
    neighbor_specs = [str(n) for n in md.neighbors()]
    assert "bytes|dense.32.gelu-dense.16.tanh" in neighbor_specs


# -- JAX --

def test_jax_forward_unit_norm():
    layer = NormDef(input_size=8).build_jax(jnp.float32)
    inputs = jnp.array(
        np.random.RandomState(0).randn(3, 4, 8), dtype=jnp.float32)
    out = layer.forward(None, inputs)
    assert out.shape == (3, 4, 8)
    norms = jnp.linalg.norm(out, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_jax_step_known_values():
    layer = NormDef(input_size=4).build_jax(jnp.float32)
    state, out = layer.step(None, None, jnp.array([3.0, 4.0, 0.0, 0.0]))
    assert state is None
    np.testing.assert_allclose(out, [0.6, 0.8, 0.0, 0.0], atol=1e-6)


def test_jax_step_zero_vector():
    layer = NormDef(input_size=4).build_jax(jnp.float32)
    _, out = layer.step(None, None, jnp.zeros(4))
    assert jnp.all(jnp.isfinite(out))


def test_jax_step_matches_forward():
    layer = NormDef(input_size=4).build_jax(jnp.float32)
    inputs = jnp.array(
        np.random.RandomState(1).randn(1, 6, 4), dtype=jnp.float32)
    fwd_out = layer.forward(None, inputs)
    for t in range(inputs.shape[1]):
        _, step_out = layer.step(None, None, inputs[0, t])
        np.testing.assert_allclose(fwd_out[0, t], step_out, atol=1e-6)
