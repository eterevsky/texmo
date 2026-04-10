import torch

from texmo.layers.norm import NormDef
from texmo.model import ModelDef
from texmo.precision import Precision


def test_def_properties():
    d = NormDef(input_size=16)
    assert d.size == 16
    assert str(d) == "norm"
    assert d.num_weights == 0
    assert d.is_valid()


def test_invalid_size1():
    assert not NormDef(input_size=1).is_valid()


def test_step():
    d = NormDef(input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert state is None

    input = torch.tensor([3.0, 4.0, 0.0, 0.0])
    new_state, out = module.step(state, input)
    assert new_state is None
    # ||[3,4,0,0]|| = 5, so output should be [0.6, 0.8, 0, 0]
    assert torch.allclose(out, torch.tensor([0.6, 0.8, 0.0, 0.0]))


def test_step_zero_vector():
    d = NormDef(input_size=4)
    module = d.build_module()
    input = torch.zeros(4)
    _, out = module.step(None, input)
    # Zero vector should stay near zero (not NaN)
    assert torch.all(torch.isfinite(out))


def test_forward_shape():
    d = NormDef(input_size=8)
    module = d.build_module()
    inputs = torch.randn(2, 5, 8)
    out = module(inputs)
    assert out.shape == (2, 5, 8)


def test_forward_unit_norm():
    d = NormDef(input_size=8)
    module = d.build_module()
    inputs = torch.randn(3, 4, 8)
    out = module(inputs)
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_matches_step():
    d = NormDef(input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        for t in range(inputs.shape[1]):
            _, step_out = module.step(None, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-6)


# -- Model-level validity constraints --

def _md(spec):
    return ModelDef(spec, Precision.FP32)


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
