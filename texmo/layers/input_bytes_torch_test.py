import torch
import numpy as np

from texmo.layers.input_bytes_torch import InputBytesDef


def test_def_properties():
    d = InputBytesDef()
    assert d.ntokens == 256
    assert d.size == 256
    assert d.num_weights == 0
    assert str(d) == 'bytes'


def test_step():
    module = InputBytesDef().build_module()
    state, out = module.step(None, 65)  # ASCII 'A'
    assert state is None
    assert out.shape == (256,)
    assert out[65] == 1.0
    assert out.sum() == 1.0
    assert out.dtype == torch.float32


def test_step_zero():
    module = InputBytesDef().build_module()
    _, out = module.step(None, 0)
    assert out[0] == 1.0
    assert out.sum() == 1.0


def test_forward():
    module = InputBytesDef().build_module()
    tokens = torch.tensor([[0, 1, 255], [10, 20, 30]], dtype=torch.long)
    out = module(tokens)

    assert out.shape == (2, 3, 256)
    assert out.dtype == torch.float32

    # Check one-hot correctness
    assert out[0, 0, 0] == 1.0
    assert out[0, 1, 1] == 1.0
    assert out[0, 2, 255] == 1.0
    assert out[1, 0, 10] == 1.0

    # Each position sums to 1
    assert torch.equal(out.sum(dim=-1), torch.ones(2, 3))


def test_forward_matches_step():
    module = InputBytesDef().build_module()
    tokens = torch.tensor([[0, 65, 255]], dtype=torch.long)
    fwd_out = module(tokens)

    for t in range(tokens.shape[1]):
        _, step_out = module.step(None, tokens[0, t].item())
        assert torch.equal(fwd_out[0, t], step_out)


def test_dtype_fp16():
    module = InputBytesDef(dtype=torch.float16).build_module()
    _, out = module.step(None, 42)
    assert out.dtype == torch.float16

    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    out = module(tokens)
    assert out.dtype == torch.float16


def test_dtype_bf16():
    module = InputBytesDef(dtype=torch.bfloat16).build_module()
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    out = module(tokens)
    assert out.dtype == torch.bfloat16


def test_from_numpy():
    """Verify it works with numpy input converted to tensor, matching dataset usage."""
    module = InputBytesDef().build_module()
    data = np.array([[72, 101, 108, 108, 111]], dtype=np.int32)  # "Hello"
    tokens = torch.from_numpy(data).long()
    out = module(tokens)
    assert out.shape == (1, 5, 256)
    assert out[0, 0, 72] == 1.0  # 'H'
