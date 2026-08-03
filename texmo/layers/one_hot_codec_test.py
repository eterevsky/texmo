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
    assert md.input.neighbors(8) == (
        "bits.4.oh+bp", "tokens.128.fold.oh", "bytes.emb.8")
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


def test_fold_tokenset_counts_frequency_weights():
    md = parse_model2("tokens.32.fold.oh|dense.4.tanh", Precision.FP32)
    plain = parse_model2(
        "tokens.32.raw_bits4.oh|dense.4.tanh", Precision.FP32)
    # 64 = 32 head-token selections + 32 stored head frequencies
    # (raised from 32 on 2026-07-29 to also price the selection, the
    # same 1-per-choice charge hexbpe pays; DB weights migrated).
    assert md.codec.num_weights == plain.codec.num_weights + 64
    # The stored numbers cost no multiplies (they only price the
    # eval loss).
    assert md.codec.num_mults == plain.codec.num_mults
    # Regression on the absolute figure: num_weights is part of a
    # conf's fleet-wide identity. head (32*4 + 32) + 64 stored.
    assert md.codec.num_weights == 224


def test_hexbpe_counts_its_stored_corpus_knowledge():
    # stats.extra_weights: 17 at 32 tokens, 428 at 256.
    md = parse_model2("tokens.32.hexbpe.oh|dense.4.tanh", Precision.FP32)
    assert md.codec.num_weights == 32 * 4 + 32 + 17
    md = parse_model2("tokens.256.hexbpe.oh|dense.4.tanh", Precision.FP32)
    assert md.codec.num_weights == 256 * 4 + 256 + 428


def test_fold_bridges_the_bit_ladder():
    md = parse_model2("bits.4.oh+bp|rnn.4.tanh", Precision.FP32)
    assert "tokens.32.fold.oh" in md.input.neighbors(4)
    md = parse_model2("tokens.32.fold.oh|rnn.4.tanh", Precision.FP32)
    assert md.input.is_valid()
    assert md.input.neighbors(4) == (
        "bits.4.oh+bp", "tokens.32.hexbpe.oh", "tokens.32.shift.oh",
        "tokens.32.fold.emb.4")


def test_hexbpe_bridges_the_bit_ladder_and_fold():
    md = parse_model2("bits.4.oh+bp|rnn.4.tanh", Precision.FP32)
    assert "tokens.32.hexbpe.oh" in md.input.neighbors(4)
    md = parse_model2("tokens.32.hexbpe.oh|rnn.4.tanh", Precision.FP32)
    assert md.input.is_valid()
    assert md.input.neighbors(4) == (
        "bits.4.oh+bp", "tokens.32.fold.oh", "tokens.64.hexbpe.oh",
        "tokens.32.shift.oh", "tokens.32.hexbpe.emb.4")


def test_hexbpe_chain_is_symmetric():
    chain = (32, 64, 128, 256)
    for i, n in enumerate(chain):
        md = parse_model2(f"tokens.{n}.hexbpe.oh|rnn.4.tanh", Precision.FP32)
        nbs = md.input.neighbors(4)
        for j in (i - 1, i + 1):
            if 0 <= j < len(chain):
                assert f"tokens.{chain[j]}.hexbpe.oh" in nbs, (n, nbs)
        # Only the immediate rungs, never a two-step jump.
        for j, m in enumerate(chain):
            if abs(i - j) > 1:
                assert f"tokens.{m}.hexbpe.oh" not in nbs


def test_shift64_counts_selections_and_pairs():
    md = parse_model2("tokens.64.shift.oh|dense.4.tanh", Precision.FP32)
    plain = parse_model2(
        "tokens.64.raw_byteshuff.oh|dense.4.tanh", Precision.FP32)
    # 95 = 62 byte-token selections + 33 stored shift pairs (26
    # capitals, tab, six symbols) at 1 weight per choice -- the
    # tokens.32.shift convention, recharged 2026-07-31 from the flat
    # 64. Markers and hex-escape spellings are structural and free.
    assert md.codec.num_weights == plain.codec.num_weights + 95


def test_shift64_bridges_hexbpe64_and_shift32():
    """Re-enabled 2026-07-29 (shift beat hexbpe-64 on the frontier):
    it hangs off the hexbpe chain at its own size, bidirectionally,
    and since 2026-07-31 also borders its 32-token sibling."""
    md = parse_model2("tokens.64.shift.oh|rnn.4.tanh", Precision.FP32)
    assert md.input.neighbors(4) == (
        "tokens.64.hexbpe.oh", "tokens.32.shift.oh",
        "tokens.128.fold.oh", "tokens.64.shift.emb.4")
    md = parse_model2("tokens.64.hexbpe.oh|rnn.4.tanh", Precision.FP32)
    assert "tokens.64.shift.oh" in md.input.neighbors(4)
    # Only hexbpe-64, shift-32 and fold-128 reach the shift-64 rung.
    for spec in ("bytes", "bits.1+bp", "bits.2.oh+bp", "bits.4.oh+bp",
                 "tokens.32.fold.oh", "tokens.32.hexbpe.oh",
                 "tokens.128.hexbpe.oh", "tokens.256.hexbpe.oh"):
        md = parse_model2(f"{spec}|rnn.4.tanh", Precision.FP32)
        assert "tokens.64.shift.oh" not in md.input.neighbors(4), spec


def test_shift32_sits_between_the_32_token_incumbents():
    md = parse_model2("tokens.32.shift.oh|rnn.4.tanh", Precision.FP32)
    assert md.input.is_valid()
    assert md.input.neighbors(4) == (
        "tokens.32.fold.oh", "tokens.32.hexbpe.oh",
        "tokens.64.shift.oh", "tokens.32.shift.emb.4")
    # Every edge is reciprocal.
    for spec in ("tokens.32.fold.oh", "tokens.32.hexbpe.oh",
                 "tokens.64.shift.oh"):
        back = parse_model2(f"{spec}|rnn.4.tanh", Precision.FP32)
        assert "tokens.32.shift.oh" in back.input.neighbors(4), spec
    # No direct bits.4 edge: the bit ladder reaches shift-32 through
    # fold or hexbpe.
    md = parse_model2("bits.4.oh+bp|rnn.4.tanh", Precision.FP32)
    assert "tokens.32.shift.oh" not in md.input.neighbors(4)


def test_shift32_counts_its_selections_and_pairs():
    md = parse_model2("tokens.32.shift.oh|dense.4.tanh", Precision.FP32)
    plain = parse_model2(
        "tokens.32.raw_bits4.oh|dense.4.tanh", Precision.FP32)
    # 58 = 29 token selections + 29 stored shift pairs; the two
    # uniform buckets store nothing.
    assert md.codec.num_weights == plain.codec.num_weights + 58
    # The stored numbers cost no multiplies (they only price the
    # eval loss).
    assert md.codec.num_mults == plain.codec.num_mults
    # Regression on the absolute figure: num_weights is part of a
    # conf's fleet-wide identity. head (32*4 + 32) + 58 stored.
    assert md.codec.num_weights == 32 * 4 + 32 + 58
    assert md.codec.num_weights == 218


def test_fold128_bridges_shift64_bytes_and_hexbpe128():
    """fold-128 (2026-08-03): 127 top raw bytes + uniform catch-all.
    Borders shift-64 below, bytes above, hexbpe-128 across."""
    md = parse_model2("tokens.128.fold.oh|rnn.4.tanh", Precision.FP32)
    assert md.input.is_valid()
    assert md.input.neighbors(4) == (
        "tokens.64.shift.oh", "bytes", "tokens.128.hexbpe.oh",
        "tokens.128.fold.emb.4")
    md = parse_model2("tokens.128.hexbpe.oh|rnn.4.tanh", Precision.FP32)
    assert "tokens.128.fold.oh" in md.input.neighbors(4)
    # 127 selection weights; the uniform bucket stores nothing.
    md = parse_model2("tokens.128.fold.oh|dense.4.tanh", Precision.FP32)
    assert md.codec.num_weights == 128 * 4 + 128 + 127
