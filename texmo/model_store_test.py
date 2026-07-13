"""Tests for the JSON model store (inline + safetensors descriptors)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.flax import save_file

from texmo.model_store import load_model, save_model
from texmo.pjson import pjson
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _init(spec, seed=1):
    md = parse_model2(spec, Precision.FP32)
    model = md.build_jax()
    return md, model, model.init_weights(jax.random.PRNGKey(seed))


def _assert_same_forward(md, w1, md2, w2, ntokens):
    batch = jax.random.randint(
        jax.random.PRNGKey(9), (2, 6), 0, ntokens).astype(jnp.int32)
    out1 = np.asarray(md.build_jax().forward(w1, batch))
    out2 = np.asarray(md2.build_jax().forward(w2, batch))
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_roundtrip_tied(tmp_path):
    md, _, w = _init("bytes.emb.4|dense.4.tanh")
    path = str(tmp_path / "m.json")
    save_model(path, md, w)
    md2, w2 = load_model(path, check_structure=True)
    assert md2.spec == md.spec
    assert md2.precision == Precision.FP32
    assert w2[2] is None  # tied head slot survives as null
    _assert_same_forward(md, w, md2, w2, 256)


def test_roundtrip_onehot_and_split(tmp_path):
    md, _, w = _init("bits.1+bp|split.add(rnn.4.tanh, pass)-dense.4.gelu")
    path = str(tmp_path / "m.json")
    save_model(path, md, w)
    md2, w2 = load_model(path, check_structure=True)
    _assert_same_forward(md, w, md2, w2, 2)


def test_descriptor_loading(tmp_path):
    # A hand-written manifest for the empty-chain bigram, with the
    # table living in a safetensors file (relative path resolved
    # against the manifest's directory) and the scalar inline.
    table = jax.random.normal(
        jax.random.PRNGKey(3), (256, 2), dtype=jnp.float32)
    save_file({'embedder.weight': table}, str(tmp_path / "w.safetensors"))
    manifest = {
        'spec': 'bytes.emb.2|',
        'precision': 'fp32',
        'weights': [
            {'emb': {'path': 'w.safetensors', 'id': 'embedder.weight'},
             'y': 0.25},
            [],
            None,
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(pjson(manifest), encoding='utf-8')
    md, w = load_model(str(path), check_structure=True)
    np.testing.assert_allclose(
        np.asarray(w[0]['emb']), np.asarray(table), atol=0)
    assert float(w[0]['y']) == 0.25
    # And it runs.
    logits = md.build_jax().forward(w, jnp.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, 256)


def test_precision_cast_on_load(tmp_path):
    # bf16 tensors in the file, fp32 model: cast at load.
    table = jnp.ones((256, 2), dtype=jnp.bfloat16) * 1.5
    save_file({'e': table}, str(tmp_path / "w.safetensors"))
    manifest = {
        'spec': 'bytes.emb.2|',
        'precision': 'fp32',
        'weights': [
            {'emb': {'path': 'w.safetensors', 'id': 'e'}, 'y': 0.0},
            [],
            None,
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(pjson(manifest), encoding='utf-8')
    _, w = load_model(str(path))
    assert w[0]['emb'].dtype == jnp.float32
    assert float(w[0]['emb'][0, 0]) == 1.5


def test_descriptor_transform(tmp_path):
    # Stored (D, 1, L) like RecurrentGemma's conv1d; loaded (L, D).
    t = jnp.arange(12, dtype=jnp.float32).reshape(3, 1, 4)
    save_file({'c': t}, str(tmp_path / "w.safetensors"))
    manifest = {
        'spec': 'bytes.emb.2|',
        'precision': 'fp32',
        'weights': [
            {'emb': {'path': 'w.safetensors', 'id': 'c',
                     'transform': [['reshape', [3, 4]],
                                   ['transpose', [1, 0]]]},
             'y': 0.0},
            [],
            None,
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(pjson(manifest), encoding='utf-8')
    _, w = load_model(str(path))
    expected = np.arange(12, dtype=np.float32).reshape(3, 4).T
    np.testing.assert_array_equal(np.asarray(w[0]['emb']), expected)


def test_structure_mismatch_raises(tmp_path):
    manifest = {
        'spec': 'bytes.emb.2|',
        'precision': 'fp32',
        'weights': [
            {'emb': [[0.0, 0.0]], 'y': 0.0},  # wrong table shape
            [],
            None,
        ],
    }
    path = tmp_path / "m.json"
    path.write_text(pjson(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='shape mismatch'):
        load_model(str(path), check_structure=True)
