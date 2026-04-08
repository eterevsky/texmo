import torch

from texmo.layers.rnn_torch import RnnDef


def test_def_properties_tanh():
    d = RnnDef(32, activation="tanh", input_size=16)
    assert d.size == 32
    assert d.output_size == 32
    assert str(d) == "rnn.32.tanh"
    assert d.is_valid()
    # nn.RNN weights: W_ih + W_hh + b_ih + b_hh
    assert d.num_weights == 32 * 16 + 32 * 32 + 2 * 32


def test_def_properties_gelu():
    d = RnnDef(16, activation="gelu", input_size=8)
    assert str(d) == "rnn.16.gelu"
    assert d.is_valid()
    # Linear weights: W(input+size, size) + b(size)
    assert d.num_weights == 16 * (8 + 16) + 16


def test_def_no_activation_invalid():
    d = RnnDef(32, input_size=16)
    assert not d.is_valid()


def test_def_non_power2_invalid():
    d = RnnDef(13, activation="tanh", input_size=8)
    assert not d.is_valid()


def test_step_tanh():
    d = RnnDef(8, activation="tanh", input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert state.shape == (8,)
    assert state.sum() == 0

    input = torch.randn(4)
    new_state, out = module.step(state, input)
    assert new_state.shape == (8,)
    assert out.shape == (8,)
    assert torch.equal(new_state, out)


def test_step_gelu():
    d = RnnDef(8, activation="gelu", input_size=4)
    module = d.build_module()
    state = module.init_state()

    input = torch.randn(4)
    new_state, out = module.step(state, input)
    assert new_state.shape == (8,)
    assert torch.equal(new_state, out)


def test_forward_shape():
    d = RnnDef(16, activation="relu", input_size=8)
    module = d.build_module()

    inputs = torch.randn(2, 5, 8)
    out = module(inputs)
    assert out.shape == (2, 5, 16)


def test_forward_shape_gelu():
    d = RnnDef(16, activation="gelu", input_size=8)
    module = d.build_module()

    inputs = torch.randn(2, 5, 8)
    out = module(inputs)
    assert out.shape == (2, 5, 16)


def test_forward_matches_step_tanh():
    """forward() and step() produce the same results for nn.RNN-backed module."""
    d = RnnDef(8, activation="tanh", input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)

        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_forward_matches_step_gelu():
    """forward() and step() produce the same results for GELU fallback module."""
    d = RnnDef(8, activation="gelu", input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)

        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_state_dict_roundtrip():
    d = RnnDef(16, activation="tanh", input_size=8)
    module1 = d.build_module()

    saved = module1.state_dict()
    module2 = d.build_module(state_dict=saved)

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(module1(inputs), module2(inputs))


def test_state_dict_roundtrip_gelu():
    d = RnnDef(16, activation="gelu", input_size=8)
    module1 = d.build_module()

    saved = module1.state_dict()
    module2 = d.build_module(state_dict=saved)

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(module1(inputs), module2(inputs))


def test_forward_batch():
    d = RnnDef(8, activation="relu", input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(3, 5, 4)
    with torch.no_grad():
        out = module(inputs)
        out0 = module(inputs[0:1])
        out1 = module(inputs[1:2])
        out2 = module(inputs[2:3])
    expected = torch.cat([out0, out1, out2], dim=0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_neighbors():
    d = RnnDef(32, activation="tanh", input_size=16)
    neighbors = list(d.neighbors())
    assert "dense.32.tanh" in neighbors
    assert "rnn.32.relu" in neighbors
    assert "rnn.16.tanh" in neighbors
    assert "rnn.64.tanh" in neighbors
