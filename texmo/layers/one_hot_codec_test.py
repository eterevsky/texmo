"""Tests for OneHotCodec (fixed codebooks + head + optional cap)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.codec import _LOGIT_CAP, cap_logits
from texmo.layers.embedding_codec import EmbeddingCodecDef
from texmo.layers.one_hot_codec import OneHotCodecDef
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _tiny_model(spec="bytes|dense.8.gelu", cap=True):
    md = parse_model2(spec, Precision.FP32, cap=cap)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    return md, model, weights


def test_cap_logits_shape_and_bounds():
    x = jnp.array([-1e6, -10.0, -1.0, 0.0, 1.0, 10.0, 1e6])
    y = np.asarray(cap_logits(x))
    # <=: tanh saturates to exactly 1.0 in fp32 for huge inputs.
    assert np.all(np.abs(y) <= _LOGIT_CAP)
    # Near-identity for small logits, saturating for huge ones.
    assert abs(y[2] - (-1.0)) < 1e-3 and abs(y[4] - 1.0) < 1e-3
    assert y[0] < -29.99 and y[6] > 29.99
    # Monotone (greedy sampling unchanged).
    assert np.all(np.diff(y) > 0)


def test_logits_uncapped_when_disabled():
    _, model, weights = _tiny_model(cap=False)
    # Blown-up head weights: raw logits far beyond the cap must
    # survive when the cap is explicitly turned off.
    weights[2]['w'] = weights[2]['w'] * 1e5
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 8), 0, 256).astype(jnp.int32)
    logits = np.asarray(model.forward(weights, batch))
    assert np.max(np.abs(logits)) > _LOGIT_CAP


def test_forward_logits_capped_by_default():
    _, model, weights = _tiny_model()
    weights[2]['w'] = weights[2]['w'] * 1e5
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 8), 0, 256).astype(jnp.int32)
    logits = np.asarray(model.forward(weights, batch))
    assert np.all(np.abs(logits) <= _LOGIT_CAP)
    assert np.max(np.abs(logits)) > _LOGIT_CAP * 0.9  # cap actually bit
    # Loss stays finite even with a pathological head.
    assert np.isfinite(float(model.loss_batch(weights, batch)))


@pytest.mark.parametrize("cap", (False, True))
def test_step_matches_forward(cap):
    _, model, weights = _tiny_model("bits.1+bp|rnn.4.tanh", cap=cap)
    batch = jax.random.randint(
        jax.random.PRNGKey(2), (1, 10), 0, 2).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    assert np.allclose(np.asarray(logits0), fwd[0, 0], atol=1e-5)
    for t in range(9):
        states, logits = model.step(weights, states, batch[0, t])
        assert np.allclose(np.asarray(logits), fwd[0, t + 1], atol=1e-5), t


def test_binary_head_pads_after_cap():
    _, model, weights = _tiny_model("bits.1+bp|dense.4.tanh")
    batch = jnp.zeros((1, 6), dtype=jnp.int32)
    logits = np.asarray(model.forward(weights, batch))
    assert logits.shape[-1] == 2
    assert np.all(logits[..., 1] == 0.0)  # the padded reference logit
    assert np.all(np.abs(logits[..., 0]) <= _LOGIT_CAP)


def test_num_weights_unchanged_by_pairing():
    # The codec must not change any conf's weight count: the head is
    # the same tensor as the old separate output dense, the codebook
    # is parameter-free.
    md, _, weights = _tiny_model("bytes|dense.8.gelu")
    # dense.8.gelu: 256*8+8; head: 8*256+256; input: 0.
    assert md.num_weights == (256 * 8 + 8) + (8 * 256 + 256)
    total = sum(int(np.asarray(x).size) for x in jax.tree.leaves(weights))
    assert total == md.num_weights


def test_properties_delegate():
    md, _, _ = _tiny_model("bytes|dense.8.gelu")
    assert md.input is md.codec
    assert md.output is md.codec.head
    assert md.codec.tokens_name == "bytes"
    assert str(md.input) == "bytes"
    assert md.input.size == 256
    assert md.output.input_size == 8


def test_input_whitelist_validity():
    for good in ("bytes", "bits.1+bp", "bits.2.oh+bp", "bits.4.oh+bp"):
        md = parse_model2(f"{good}|dense.8.gelu", Precision.FP32)
        assert md.input.is_valid(), good
    # Off-whitelist variants parse and run but count as invalid --
    # including the retired bits.1+bm (2026-07: practically never
    # beat bits.1+bp in search).
    for bad in ("bits.2+bp", "bits.4+bp", "bits.1.oh+bp", "bits.2.oh",
                "bits.4", "bits.8", "bits.1+bm", "bits.2+bm",
                "bits.4+bm"):
        md = parse_model2(f"{bad}|dense.8.gelu", Precision.FP32)
        assert not md.input.is_valid(), bad
        assert not md.is_valid(), bad


def test_bits8_oh_canonicalizes_to_bytes():
    md = parse_model2("bits.8.oh|dense.8.gelu", Precision.FP32)
    assert md.spec.startswith("bytes|")
    assert md.codec.tokens_name == "bytes"
    assert md.is_valid()


def test_empty_input_spec_means_bytes():
    md = parse_model2("|dense.8.gelu", Precision.FP32)
    assert md.spec.startswith("bytes|")


def test_input_neighbors_ladder():
    # neighbors() takes the chain's final width; the last entry is
    # the mode swap to the tied codec at that width.
    md = parse_model2("bits.2.oh+bp|dense.8.gelu", Precision.FP32)
    assert md.input.neighbors(8) == (
        "bits.1+bp", "bits.4.oh+bp", "bits.2.emb.8")
    md = parse_model2("bytes|dense.8.gelu", Precision.FP32)
    assert md.input.neighbors(8) == ("bits.4.oh+bp", "bytes.emb.8")
    # bits.1+bm is retired: no incoming edges, but its own outgoing
    # edge remains as a migration bridge for the DB population.
    md = parse_model2("bits.1+bm|dense.8.gelu", Precision.FP32)
    assert md.input.neighbors(8) == ("bits.1+bp", "bits.1.emb.8")
    md = parse_model2("bits.1+bp|dense.8.gelu", Precision.FP32)
    assert md.input.neighbors(8) == ("bits.2.oh+bp", "bits.1.emb.8")


def test_bm_encoding():
    md = parse_model2("bits.1+bm|dense.4.tanh", Precision.FP32)
    assert md.input.size == 2  # value bit + marker bit
    assert md.input.tokens_name == "bits.1"
    codec = md.codec.build_jax()
    tokens = jnp.array([[1, 0, 1, 1, 0, 0, 0, 1, 1, 0]], dtype=jnp.int32)
    enc = np.asarray(codec.encode(None, tokens, padding=0))
    # Column 0: the value bits verbatim.
    assert np.array_equal(enc[0, :, 0], np.asarray(tokens[0], np.float32))
    # Column 1: marker fires on the first chunk of each byte (mod 8).
    assert np.array_equal(
        enc[0, :, 1], np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0], np.float32))
    # Padding continues the cycle in the negative direction:
    # position -1 -> (-1 % 8) == 7 -> no marker, max-entropy value.
    enc_p = np.asarray(codec.encode(None, tokens, padding=1))
    assert np.array_equal(enc_p[0, 0], np.array([0.5, 0.0], np.float32))
    assert np.array_equal(enc_p[0, 1:], enc[0])


def test_bm_step_matches_forward():
    _, model, weights = _tiny_model("bits.1+bm|rnn.4.tanh")
    batch = jax.random.randint(
        jax.random.PRNGKey(4), (1, 12), 0, 2).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    assert np.allclose(np.asarray(logits0), fwd[0, 0], atol=1e-5)
    for t in range(11):
        states, logits = model.step(weights, states, batch[0, t])
        assert np.allclose(np.asarray(logits), fwd[0, t + 1], atol=1e-5), t


def test_tokens_oh_parses_and_encodes():
    md = parse_model2("tokens.16.test.oh|dense.4.tanh", Precision.FP32)
    assert md.input.tokens_name == "tokens16_test"
    assert md.input.size == 16
    assert md.input.is_valid()
    assert md.input.neighbors(4) == ("tokens.16.test.emb.4",)
    codec = md.codec.build_jax()
    enc = np.asarray(codec.encode(None, jnp.array([[3, 5]]), padding=0))
    expected = np.zeros((1, 2, 16), dtype=np.float32)
    expected[0, 0, 3] = 1.0
    expected[0, 1, 5] = 1.0
    assert np.array_equal(enc, expected)
    # Step mode agrees and needs no state.
    assert codec.init_state() is None
    _, v = codec.encode_step(None, None, 3)
    assert np.array_equal(np.asarray(v), expected[0, 0])


def test_emb_specs_dispatch_to_embedding_codec():
    # *.emb.X belongs to EmbeddingCodec; the parser dispatches on it.
    md = parse_model2("tokens.16.test.emb.8|dense.8.tanh", Precision.FP32)
    assert isinstance(md.codec, EmbeddingCodecDef)
    md = parse_model2("bytes|dense.8.gelu", Precision.FP32)
    assert isinstance(md.codec, OneHotCodecDef)
    # Direct OneHotCodec parsing still rejects emb specs.
    with pytest.raises(ValueError, match="EmbeddingCodec"):
        OneHotCodecDef.from_spec("tokens.16.test.emb.8")
