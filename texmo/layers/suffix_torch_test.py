import torch

from texmo.layers.suffix_torch import SuffixDef, SuffixModule


def test_def_properties():
    d = SuffixDef(4, input_size=8)
    assert d.length == 4
    assert d.output_size == 32
    assert d.num_weights == 0
    assert str(d) == 'suffix.4'
    assert d.is_valid()


def test_def_is_valid():
    assert SuffixDef(2, input_size=4).is_valid()
    assert SuffixDef(4, input_size=4).is_valid()
    assert not SuffixDef(1, input_size=4).is_valid()  # must be > 1
    assert not SuffixDef(3, input_size=4).is_valid()  # must be power of 2


def test_step():
    module = SuffixDef(2, input_size=4).build_module()
    state = module.init_state()
    assert state.shape == (1, 4)
    assert state.sum() == 0

    v1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    state, out = module.step(state, v1)
    # Window: [zeros, v1] → flattened
    assert out.shape == (8,)
    torch.testing.assert_close(out, torch.tensor([0, 0, 0, 0, 1, 2, 3, 4], dtype=torch.float32))

    v2 = torch.tensor([5, 6, 7, 8], dtype=torch.float32)
    state, out = module.step(state, v2)
    # Window: [v1, v2]
    torch.testing.assert_close(out, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32))

    v3 = torch.tensor([9, 10, 11, 12], dtype=torch.float32)
    state, out = module.step(state, v3)
    # Window: [v2, v3]
    torch.testing.assert_close(out, torch.tensor([5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.float32))


def test_forward():
    module = SuffixDef(2, input_size=4).build_module()
    # 1 batch, 4 positions (including padding), 4 features
    inputs = torch.tensor([[[0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]]], dtype=torch.float32)

    out = module(inputs)
    # suffix.2 consumes 1 position → output has 3 positions
    assert out.shape == (1, 3, 8)

    expected = torch.tensor([[[0, 0, 0, 0, 1, 2, 3, 4],
                              [1, 2, 3, 4, 5, 6, 7, 8],
                              [5, 6, 7, 8, 9, 10, 11, 12]]], dtype=torch.float32)
    torch.testing.assert_close(out, expected)


def test_forward_batch():
    module = SuffixDef(2, input_size=4).build_module()
    inputs = torch.tensor([[[0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]],
                           [[0, 0, 0, 0],
                            [10, 20, 30, 40],
                            [50, 60, 70, 80],
                            [90, 100, 110, 120]]], dtype=torch.float32)

    out = module(inputs)
    assert out.shape == (2, 3, 8)


def test_forward_matches_step():
    module = SuffixDef(2, input_size=4).build_module()
    inputs = torch.tensor([[[0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]]], dtype=torch.float32)

    fwd = module(inputs)

    # Step through the same inputs
    state = module.init_state()
    step_outputs = []
    for t in range(inputs.shape[1]):
        state, out = module.step(state, inputs[0, t])
        step_outputs.append(out)

    # forward output starts at position 1 (after consuming 1 padding position)
    # step outputs include position 0 too
    for i in range(fwd.shape[1]):
        assert torch.equal(fwd[0, i], step_outputs[i + 1])


def test_suffix4_forward():
    module = SuffixDef(4, input_size=2).build_module()
    # 7 positions, suffix.4 consumes 3 → 4 output positions
    inputs = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
    out = module(inputs)
    assert out.shape == (1, 4, 8)

    # First output: positions 0,1,2,3
    torch.testing.assert_close(out[0, 0], inputs[0, 0:4].reshape(-1))
    # Last output: positions 3,4,5,6
    torch.testing.assert_close(out[0, 3], inputs[0, 3:7].reshape(-1))


# -- model integration --

def test_model_with_suffix():
    from texmo.model_torch import ModelDef
    md = ModelDef("bytes|suffix.2-dense.64.gelu")
    assert md.total_padding == 2  # 1 base + (2-1)
    model = md.build_model()

    batch = torch.randint(0, 256, (4, 16))
    logits = model(batch)
    assert logits.shape == (4, 16, 256)


def test_model_suffix_step_matches_forward():
    from texmo.model_torch import ModelDef
    md = ModelDef("bytes|suffix.2-dense.32.tanh")
    model = md.build_model()
    model.eval()

    tokens = [10, 20, 30, 40, 50]
    batch = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        fwd_logits = model(batch)

        states, step_logits_0 = model.initial_step()
        step_logits = [step_logits_0]
        for t in tokens[:-1]:
            states, logits_t = model.step(states, t)
            step_logits.append(logits_t)

    for i in range(len(tokens)):
        assert torch.allclose(fwd_logits[0, i], step_logits[i], atol=1e-5)


def test_model_suffix_only():
    """Model with suffix and no other hidden layers."""
    from texmo.model_torch import ModelDef
    md = ModelDef("bytes|suffix.4")
    assert md.total_padding == 4  # 1 + 3
    model = md.build_model()

    batch = torch.randint(0, 256, (2, 32))
    logits = model(batch)
    assert logits.shape == (2, 32, 256)


def test_model_two_suffix_layers():
    from texmo.model_torch import ModelDef
    md = ModelDef("bytes|suffix.2-suffix.2")
    assert md.total_padding == 3  # 1 + 1 + 1
    model = md.build_model()

    batch = torch.randint(0, 256, (2, 16))
    logits = model(batch)
    assert logits.shape == (2, 16, 256)


def test_model_suffix_loss():
    from texmo.model_torch import ModelDef
    md = ModelDef("bytes|suffix.2-dense.64.gelu")
    model = md.build_model()

    batch = torch.randint(0, 256, (4, 16))
    loss = model.loss_batch(batch)
    assert loss.shape == ()
    assert loss.item() > 0


def test_model_bits_with_suffix():
    from texmo.model_torch import ModelDef
    md = ModelDef("bits.1+bp|suffix.4-dense.16.gelu")
    model = md.build_model()
    model.eval()

    tokens = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    batch = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        fwd_logits = model(batch)
        assert fwd_logits.shape == (1, 16, 2)

        states, step_logits_0 = model.initial_step()
        step_logits = [step_logits_0]
        for t in tokens[:-1]:
            states, logits_t = model.step(states, t)
            step_logits.append(logits_t)

    for i in range(len(tokens)):
        assert torch.allclose(fwd_logits[0, i], step_logits[i], atol=1e-5)
