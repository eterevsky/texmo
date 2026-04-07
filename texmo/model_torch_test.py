import math
import torch
import numpy as np

from texmo.model_torch import ModelDef, build_model_def

_1_BY_LOG2 = 1.0 / math.log(2.0)


def test_model_def_properties():
    md = ModelDef("bytes|dense.128.gelu")
    assert md.ntokens == 256
    assert str(md) == "bytes|dense.128.gelu"
    assert md.num_weights == (
        0  # input (no weights)
        + 256 * 128 + 128  # dense layer
        + 128 * 256 + 256  # output layer
    )


def test_build_model():
    md = ModelDef("bytes|dense.64.relu")
    model = md.build_model()

    # Check it's an nn.Module with parameters
    params = list(model.parameters())
    assert len(params) > 0


def test_forward_shape():
    md = ModelDef("bytes|dense.64.gelu")
    model = md.build_model()

    batch = torch.randint(0, 256, (4, 16))
    logits = model(batch)
    assert logits.shape == (4, 16, 256)


def test_loss_batch():
    md = ModelDef("bytes|dense.64.gelu")
    model = md.build_model()

    batch = torch.randint(0, 256, (4, 16))
    loss = model.loss_batch(batch)

    assert loss.shape == ()
    assert loss.item() > 0
    # Random model on 256 tokens should have loss near log2(256) = 8
    assert loss.item() < 12  # generous upper bound


def test_loss_decreases():
    """Verify that a few optimization steps reduce the loss."""
    md = ModelDef("bytes|dense.32.relu")
    model = md.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch = torch.randint(0, 256, (8, 32))

    initial_loss = model.loss_batch(batch).item()

    for _ in range(20):
        loss = model.loss_batch(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = model.loss_batch(batch).item()
    assert final_loss < initial_loss


def test_step_and_initial_step():
    md = ModelDef("bytes|dense.64.tanh")
    model = md.build_model()
    model.eval()

    states, logits0 = model.initial_step()
    assert logits0.shape == (256,)
    assert len(states) == 2  # input state + one dense layer

    states, logits1 = model.step(states, 65)  # feed 'A'
    assert logits1.shape == (256,)


def test_step_matches_forward():
    """step() should produce the same logits as forward() for the same input."""
    md = ModelDef("bytes|dense.64.tanh")
    model = md.build_model()
    model.eval()

    tokens = [10, 20, 30]
    batch = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        fwd_logits = model(batch)  # (1, 3, 256)

        # Run step-by-step
        states, step_logits_0 = model.initial_step()
        step_logits = [step_logits_0]
        for t in tokens[:-1]:
            states, logits_t = model.step(states, t)
            step_logits.append(logits_t)

    for i in range(len(tokens)):
        assert torch.allclose(fwd_logits[0, i], step_logits[i], atol=1e-5)


def test_step_prob():
    md = ModelDef("bytes|dense.32.gelu")
    model = md.build_model()
    model.eval()

    states, _ = model.initial_step()
    states, probs = model.step_prob(states, 0, temperature=1.0)

    assert probs.shape == (256,)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs >= 0).all()


def test_step_sample():
    md = ModelDef("bytes|dense.32.gelu")
    model = md.build_model()
    model.eval()

    states, _ = model.initial_step()
    states, token = model.step_sample(states, 0, temperature=1.0)

    assert isinstance(token, int)
    assert 0 <= token < 256


def test_state_dict_roundtrip():
    md = ModelDef("bytes|dense.64.gelu")
    model1 = md.build_model()

    saved = model1.state_dict()
    model2 = md.build_model(state_dict=saved)

    batch = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        assert torch.equal(model1(batch), model2(batch))


def test_dtype_fp64():
    md = ModelDef("bytes|dense.32.gelu", dtype=torch.float64)
    model = md.build_model()

    # Check parameters are fp64
    for p in model.parameters():
        assert p.dtype == torch.float64

    batch = torch.randint(0, 256, (2, 8))
    logits = model(batch)
    assert logits.dtype == torch.float64


def test_build_model_def_cache():
    md1 = build_model_def("bytes|dense.32.gelu")
    md2 = build_model_def("bytes|dense.32.gelu")
    assert md1 is md2

    md3 = build_model_def("bytes|dense.32.gelu", dtype=torch.float16)
    assert md1 is not md3


def test_multi_layer():
    md = ModelDef("bytes|dense.64.relu-dense.32.tanh")
    model = md.build_model()

    batch = torch.randint(0, 256, (2, 8))
    logits = model(batch)
    assert logits.shape == (2, 8, 256)

    # num_weights should include both layers + output
    assert md.num_weights == (
        0           # input
        + 256 * 64 + 64   # dense.64
        + 64 * 32 + 32    # dense.32
        + 32 * 256 + 256  # output
    )


def test_from_numpy():
    """Verify it works with numpy arrays from DataSet."""
    md = ModelDef("bytes|dense.32.gelu")
    model = md.build_model()

    data = np.random.randint(0, 256, (4, 16), dtype=np.int32)
    batch = torch.from_numpy(data).long()
    loss = model.loss_batch(batch)
    assert loss.shape == ()
