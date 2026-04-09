import torch

from texmo.layers.dense_torch import DenseDef


def _make_module():
    layer_def = DenseDef(2, input_size=4)
    module = layer_def.build_module({
        "linear.weight": torch.tensor([[2, 0, 1, 0], [0, 1, 0, 2]], dtype=torch.float32),
        "linear.bias": torch.tensor([10, 20], dtype=torch.float32),
    })
    return layer_def, module


def test_step():
    _, module = _make_module()
    input = torch.tensor([1, 2, 3, 4], dtype=torch.float32)

    state, out = module.step(None, input)
    assert state is None
    assert torch.equal(out, torch.tensor([15, 30], dtype=torch.float32))


def test_forward_single():
    _, module = _make_module()
    # (batch=1, seq=1, 4)
    input = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)

    out = module(input)
    assert out.shape == (1, 1, 2)

    _, step_out = module.step(None, input[0, 0])
    assert torch.equal(out[0, 0], step_out)


def test_forward_sequence():
    _, module = _make_module()
    # (batch=1, seq=2, 4)
    input = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=torch.float32)

    out = module(input)
    assert out.shape == (1, 2, 2)

    _, out0 = module.step(None, input[0, 0])
    _, out1 = module.step(None, input[0, 1])
    expected = torch.stack([out0, out1])
    assert torch.equal(out[0], expected)


def test_forward_batch():
    _, module = _make_module()
    # (batch=2, seq=2, 4)
    input = torch.tensor([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
    ], dtype=torch.float32)

    out = module(input)
    assert out.shape == (2, 2, 2)

    out0 = module(input[0:1])
    out1 = module(input[1:2])
    expected = torch.cat([out0, out1], dim=0)
    assert torch.equal(out, expected)


def test_forward_matches_step():
    """forward() and step() produce the same results."""
    _, module = _make_module()
    input = torch.tensor([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
    ], dtype=torch.float32)

    fwd_out = module(input)

    for b in range(input.shape[0]):
        for t in range(input.shape[1]):
            _, step_out = module.step(None, input[b, t])
            assert torch.equal(fwd_out[b, t], step_out)


def test_activation_relu():
    layer_def = DenseDef(2, activation="relu", input_size=4)
    module = layer_def.build_module({
        "linear.weight": torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),
        "linear.bias": torch.tensor([-5, -5], dtype=torch.float32),
    })

    input = torch.tensor([[[3, 0, 0, 10]]], dtype=torch.float32)
    out = module(input)
    # relu(-5 + 3) = 0, relu(-5 + 10) = 5
    assert torch.equal(out, torch.tensor([[[0, 5]]], dtype=torch.float32))


def test_build_module_fresh():
    """build_module() without state_dict produces a module with random weights."""
    layer_def = DenseDef(8, activation="gelu", input_size=4)
    module = layer_def.build_module()

    assert module.linear.weight.shape == (8, 4)
    assert module.linear.bias.shape == (8,)

    input = torch.randn(2, 3, 4)
    out = module(input)
    assert out.shape == (2, 3, 8)


def test_roundtrip_state_dict():
    """Weights survive a save/load roundtrip."""
    layer_def = DenseDef(2, activation="tanh", input_size=4)
    module1 = layer_def.build_module()

    saved = module1.state_dict()
    module2 = layer_def.build_module(state_dict=saved)

    input = torch.randn(1, 3, 4)
    assert torch.equal(module1(input), module2(input))


def test_num_weights():
    layer_def = DenseDef(8, activation="gelu", input_size=4)
    assert layer_def.num_weights == 8 * 4 + 8


def test_neighbors():
    d = DenseDef(32, activation="relu", input_size=16)
    neighbors = list(d.neighbors())
    assert "dense.16.relu" in neighbors
    assert "dense.64.relu" in neighbors
    assert "rnn.32.relu" in neighbors


def test_neighbors_size1():
    d = DenseDef(1, activation="tanh", input_size=8)
    neighbors = list(d.neighbors())
    assert "dense.2.tanh" in neighbors
    assert "rnn.1.tanh" in neighbors
