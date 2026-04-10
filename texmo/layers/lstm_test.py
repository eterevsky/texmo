import torch

from texmo.layers.lstm import LstmDef


def test_def_properties():
    d = LstmDef(32, input_size=16)
    assert d.size == 32
    assert str(d) == "lstm.32"
    assert d.is_valid()
    # nn.LSTM: 4 gates * (input_weights + hidden_weights + 2 biases)
    assert d.num_weights == 4 * 32 * (16 + 32) + 8 * 32
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_non_power2_invalid():
    assert not LstmDef(13, input_size=8).is_valid()


def test_step():
    d = LstmDef(8, input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert isinstance(state, tuple)
    h, c = state
    assert h.shape == (8,)
    assert c.shape == (8,)
    assert h.sum() == 0
    assert c.sum() == 0

    new_state, out = module.step(state, torch.randn(4))
    new_h, new_c = new_state
    assert new_h.shape == (8,)
    assert new_c.shape == (8,)
    assert out.shape == (8,)
    assert torch.equal(out, new_h)


def test_forward_shape():
    d = LstmDef(16, input_size=8)
    module = d.build_module()
    inputs = torch.randn(2, 5, 8)
    out = module(inputs)
    assert out.shape == (2, 5, 16)


def test_forward_matches_step():
    d = LstmDef(8, input_size=4)
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
    d = LstmDef(16, input_size=8)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())

    inputs = torch.randn(2, 4, 8)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


def test_neighbors():
    d = LstmDef(32, input_size=16)
    neighbors = list(d.neighbors())
    # Size mutations
    assert "lstm.16" in neighbors
    assert "lstm.64" in neighbors
    # Type swaps (gru and mgru only)
    assert "gru.32" in neighbors
    assert "mgru.32" in neighbors
    # Not a neighbor of mingru or rnn
    assert "mingru.32" not in neighbors
    for act in ("relu", "gelu", "tanh"):
        assert f"rnn.32.{act}" not in neighbors


def test_gru_has_lstm_neighbor():
    from texmo.layers.gru import GruDef, MgruDef, MinGruDef
    g = list(GruDef(32, input_size=16).neighbors())
    assert "lstm.32" in g

    m = list(MgruDef(32, input_size=16).neighbors())
    assert "lstm.32" in m

    mg = list(MinGruDef(32, input_size=16).neighbors())
    assert "lstm.32" not in mg
