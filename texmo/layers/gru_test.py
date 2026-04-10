import torch

from texmo.layers.gru import GruDef, MgruDef, MinGruDef


# -- GRU --

def test_gru_def_properties():
    d = GruDef(32, input_size=16)
    assert d.size == 32
    assert str(d) == "gru.32"
    assert d.is_valid()
    # nn.GRU: 3 gates * (input_weights + hidden_weights + 2 biases)
    assert d.num_weights == 3 * 32 * (16 + 32) + 6 * 32
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_gru_non_power2_invalid():
    assert not GruDef(13, input_size=8).is_valid()


def test_gru_step():
    d = GruDef(8, input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert state.shape == (8,)
    assert state.sum() == 0

    new_state, out = module.step(state, torch.randn(4))
    assert new_state.shape == (8,)
    assert out.shape == (8,)
    assert torch.equal(new_state, out)


def test_gru_forward_shape():
    d = GruDef(16, input_size=8)
    module = d.build_module()
    inputs = torch.randn(2, 5, 8)
    out = module(inputs)
    assert out.shape == (2, 5, 16)


def test_gru_forward_matches_step():
    d = GruDef(8, input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_gru_state_dict_roundtrip():
    d = GruDef(16, input_size=8)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


# -- MGRU --

def test_mgru_def_properties():
    d = MgruDef(32, input_size=16)
    assert d.size == 32
    assert str(d) == "mgru.32"
    assert d.is_valid()
    # 2 linear(48, 32): 2 * (32*48 + 32)
    assert d.num_weights == 2 * (32 * 48 + 32)
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_mgru_step():
    d = MgruDef(8, input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert state.shape == (8,)

    new_state, out = module.step(state, torch.randn(4))
    assert new_state.shape == (8,)
    assert torch.equal(new_state, out)


def test_mgru_forward_shape():
    d = MgruDef(16, input_size=8)
    module = d.build_module()
    out = module(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 16)


def test_mgru_forward_matches_step():
    d = MgruDef(8, input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_mgru_state_dict_roundtrip():
    d = MgruDef(16, input_size=8)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


# -- MinGRU --

def test_mingru_def_properties():
    d = MinGruDef(32, input_size=16)
    assert d.size == 32
    assert str(d) == "mingru.32"
    assert d.is_valid()
    # linear(16, 64): 64*16 + 64
    assert d.num_weights == 2 * 32 * 16 + 2 * 32
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_mingru_step():
    d = MinGruDef(8, input_size=4)
    module = d.build_module()
    state = module.init_state()

    new_state, out = module.step(state, torch.randn(4))
    assert new_state.shape == (8,)
    assert torch.equal(new_state, out)


def test_mingru_forward_shape():
    d = MinGruDef(16, input_size=8)
    module = d.build_module()
    out = module(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 16)


def test_mingru_forward_matches_step():
    d = MinGruDef(8, input_size=4)
    module = d.build_module()
    module.eval()

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_mingru_state_dict_roundtrip():
    d = MinGruDef(16, input_size=8)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


# -- Neighbors --

def test_gru_neighbors():
    d = GruDef(32, input_size=16)
    neighbors = list(d.neighbors())
    # Size mutations
    assert "gru.16" in neighbors
    assert "gru.64" in neighbors
    # Swap to rnn with every activation
    assert "rnn.32.relu" in neighbors
    assert "rnn.32.gelu" in neighbors
    assert "rnn.32.tanh" in neighbors
    # Swap to other recurrent types
    assert "mgru.32" in neighbors
    assert "mingru.32" in neighbors
    assert "gru.32" not in neighbors


def test_mgru_neighbors():
    d = MgruDef(16, input_size=8)
    neighbors = list(d.neighbors())
    assert "mgru.8" in neighbors
    assert "mgru.32" in neighbors
    assert "rnn.16.tanh" in neighbors
    assert "gru.16" in neighbors
    assert "mingru.16" in neighbors


def test_mingru_neighbors():
    d = MinGruDef(8, input_size=4)
    neighbors = list(d.neighbors())
    assert "mingru.4" in neighbors
    assert "mingru.16" in neighbors
    assert "rnn.8.relu" in neighbors
    assert "gru.8" in neighbors
    assert "mgru.8" in neighbors
