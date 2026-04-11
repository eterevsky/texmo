import torch

from texmo.layers.latent import LatentDef, LrnnDef


# -- LatentDef --

def test_latent_def_properties():
    d = LatentDef(16, 4, input_size=8)
    assert d.size == 16
    assert d.reps == 4
    assert str(d) == "latent.16.4"
    assert d.is_valid()
    # Wi(8,16) + bias + Wr(16,16)
    assert d.num_weights == 16 * 8 + 16 + 16 * 16
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_latent_invalid_size1():
    assert not LatentDef(1, 2, input_size=8).is_valid()


def test_latent_invalid_reps1():
    assert not LatentDef(8, 1, input_size=4).is_valid()


def test_latent_invalid_non_power2():
    assert not LatentDef(13, 4, input_size=8).is_valid()
    assert not LatentDef(8, 3, input_size=8).is_valid()


def test_latent_step():
    d = LatentDef(8, 2, input_size=4)
    module = d.build_module()
    state, out = module.step(None, torch.randn(4))
    assert state is None
    assert out.shape == (8,)


def test_latent_forward_shape():
    d = LatentDef(16, 4, input_size=8)
    module = d.build_module()
    out = module(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 16)


def test_latent_forward_matches_step():
    d = LatentDef(8, 4, input_size=4)
    module = d.build_module()
    module.eval()
    module._deterministic_init = True

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        for t in range(inputs.shape[1]):
            _, step_out = module.step(None, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_latent_state_dict_roundtrip():
    d = LatentDef(8, 2, input_size=4)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())
    m1._deterministic_init = True
    m2._deterministic_init = True
    inputs = torch.randn(2, 4, 4)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


def test_latent_random_init():
    """By default the latent state is initialized randomly, so two
    forward passes on identical inputs should differ."""
    d = LatentDef(8, 4, input_size=4)
    module = d.build_module()
    inputs = torch.randn(1, 3, 4)
    with torch.no_grad():
        out1 = module(inputs)
        out2 = module(inputs)
    assert not torch.equal(out1, out2)


def test_latent_deterministic_init_flag():
    """Setting _deterministic_init=True makes the output reproducible."""
    d = LatentDef(8, 4, input_size=4)
    module = d.build_module()
    module._deterministic_init = True
    inputs = torch.randn(1, 3, 4)
    with torch.no_grad():
        out1 = module(inputs)
        out2 = module(inputs)
    assert torch.equal(out1, out2)


def test_latent_neighbors():
    d = LatentDef(16, 4, input_size=8)
    neighbors = list(d.neighbors())
    # Size 2x
    assert "latent.8.4" in neighbors
    assert "latent.32.4" in neighbors
    # Reps 2x
    assert "latent.16.2" in neighbors
    assert "latent.16.8" in neighbors
    # Type swap
    assert "lrnn.16.4" in neighbors


def test_latent_neighbors_reps2_collapse():
    d = LatentDef(16, 2, input_size=8)
    neighbors = list(d.neighbors())
    # When reps=2, dense.X.tanh is a neighbor
    assert "dense.16.tanh" in neighbors
    # And we shouldn't drop reps below 2
    assert "latent.16.1" not in neighbors


def test_dense_tanh_has_latent_neighbor():
    from texmo.layers.dense import DenseDef
    d = DenseDef(16, input_size=8, activation="tanh")
    neighbors = list(d.neighbors())
    assert "latent.16.2" in neighbors


def test_dense_relu_no_latent_neighbor():
    from texmo.layers.dense import DenseDef
    d = DenseDef(16, input_size=8, activation="relu")
    neighbors = list(d.neighbors())
    assert "latent.16.2" not in neighbors


# -- LrnnDef --

def test_lrnn_def_properties():
    d = LrnnDef(16, 4, input_size=8)
    assert d.size == 16
    assert d.reps == 4
    assert str(d) == "lrnn.16.4"
    assert d.is_valid()
    # Wi(8,16) + bias + Wh(16,16) + Wr(16,16)
    assert d.num_weights == 16 * 8 + 16 + 16 * 16 + 16 * 16
    module = d.build_module()
    assert sum(p.numel() for p in module.parameters()) == d.num_weights


def test_lrnn_step():
    d = LrnnDef(8, 2, input_size=4)
    module = d.build_module()
    state = module.init_state()
    assert state.shape == (8,)
    assert state.sum() == 0

    new_state, out = module.step(state, torch.randn(4))
    assert new_state.shape == (8,)
    assert out.shape == (8,)
    assert torch.equal(new_state, out)


def test_lrnn_forward_shape():
    d = LrnnDef(16, 2, input_size=8)
    module = d.build_module()
    out = module(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 16)


def test_lrnn_forward_matches_step():
    d = LrnnDef(8, 4, input_size=4)
    module = d.build_module()
    module.eval()
    module._deterministic_init = True

    inputs = torch.randn(1, 6, 4)
    with torch.no_grad():
        fwd_out = module(inputs)
        state = module.init_state()
        for t in range(inputs.shape[1]):
            state, step_out = module.step(state, inputs[0, t])
            assert torch.allclose(fwd_out[0, t], step_out, atol=1e-5)


def test_lrnn_state_dict_roundtrip():
    d = LrnnDef(8, 2, input_size=4)
    m1 = d.build_module()
    m2 = d.build_module(state_dict=m1.state_dict())
    m1._deterministic_init = True
    m2._deterministic_init = True
    inputs = torch.randn(2, 4, 4)
    with torch.no_grad():
        assert torch.equal(m1(inputs), m2(inputs))


def test_lrnn_neighbors():
    d = LrnnDef(16, 4, input_size=8)
    neighbors = list(d.neighbors())
    assert "lrnn.8.4" in neighbors
    assert "lrnn.32.4" in neighbors
    assert "lrnn.16.2" in neighbors
    assert "lrnn.16.8" in neighbors
    assert "latent.16.4" in neighbors


def test_lrnn_neighbors_reps2_collapse():
    d = LrnnDef(16, 2, input_size=8)
    neighbors = list(d.neighbors())
    assert "rnn.16.tanh" in neighbors


def test_rnn_tanh_has_lrnn_neighbor():
    from texmo.layers.rnn import RnnDef
    d = RnnDef(16, input_size=8, activation="tanh")
    neighbors = list(d.neighbors())
    assert "lrnn.16.2" in neighbors


def test_rnn_relu_no_lrnn_neighbor():
    from texmo.layers.rnn import RnnDef
    d = RnnDef(16, input_size=8, activation="relu")
    neighbors = list(d.neighbors())
    assert "lrnn.16.2" not in neighbors
