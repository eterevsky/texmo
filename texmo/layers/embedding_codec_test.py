"""Tests for EmbeddingCodec (tied table, exp(y) scale, always-direct)."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _build(spec):
    md = parse_model2(spec, Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(0))
    return md, model, weights


def test_parse_fields_and_roundtrip():
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    c = md.codec
    assert (c.ntokens, c.npositions, c.size, c.emb_size) == (16, 2, 8, 8)
    assert c.tokens_name == "bits.4"
    assert md.spec == "bits.4.emb.8|rnn.8.tanh"
    assert md.is_valid()
    # Canonical spec round-trips to an equal model.
    assert parse_model2(md.spec, Precision.FP32).spec == md.spec

    md = parse_model2("tokens.16.test.emb.8|dense.8.tanh", Precision.FP32)
    assert md.codec.tokens_name == "tokens.16.test"
    assert md.codec.npositions == 1


def test_stale_x_is_invalid_not_rewritten():
    # The parser is a faithful reader: a spec whose X disagrees with
    # the chain's final width parses as written and fails is_valid.
    # Keeping X in sync on structure mutations is the neighbor
    # generator's job (future search-facing pass), not the parser's.
    md = parse_model2("bytes.emb.8|dense.4.tanh", Precision.FP32)
    assert md.spec == "bytes.emb.8|dense.4.tanh"
    assert md.codec.emb_size == 8
    assert not md.is_valid()


def test_inconsistent_width_is_invalid():
    # suffix doubles the width, so no X satisfies X == final width.
    md = parse_model2("bits.2.emb.4|suffix.2", Precision.FP32)
    assert not md.is_valid()


def test_width_preserving_chain_keeps_spec_x():
    # The Gemma shape: the chain inherits its width FROM the input,
    # so X is load-bearing and consistent for any value.
    md = parse_model2("bytes.emb.16|rglru.1", Precision.FP32)
    assert md.codec.emb_size == 16
    assert md.is_valid()
    md = parse_model2("bytes.emb.32|rglru.1", Precision.FP32)
    assert md.codec.emb_size == 32


def test_empty_chain_is_the_symmetric_bigram():
    md, model, weights = _build("bytes.emb.4|")
    assert md.is_valid()
    assert md.num_weights == 256 * 4 + 1
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 6), 0, 256).astype(jnp.int32)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 6, 256)


def test_num_weights_smallest_byte_model():
    # The chain dense doubles as the adapter: table 256 + scale 1 +
    # dense.1.tanh (1*1+1) = 259.
    md, _, weights = _build("bytes.emb.1|dense.1.tanh")
    assert md.num_weights == 259
    total = sum(int(np.asarray(x).size) for x in jax.tree.leaves(weights))
    assert total == md.num_weights


def test_num_weights_with_positions():
    # bits.4.emb.8: table (16+2)*8 + scale 1; no head parameters.
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.num_weights == (16 + 2) * 8 + 1
    assert md.output.num_weights == 0


def test_scale_init_is_one():
    # y0 = 0 (scale 1) since 2026-08-16; the sqrt(X) init measured as
    # a no-op for final loss (see embedding_codec.py docstring).
    _, _, weights = _build("bytes.emb.16|dense.16.tanh")
    y = float(np.asarray(weights[0]['y']))
    assert abs(math.exp(y) - 1.0) < 1e-5


def test_head_is_tied_to_the_table():
    md, model, weights = _build("bytes.emb.4|dense.4.tanh")
    assert weights[2] is None  # the head has no parameters of its own
    codec = md.codec.build_jax()
    h = jnp.array([0.01, -0.02, 0.03, 0.005], dtype=jnp.float32)
    base = np.asarray(codec.logits(weights[0], None, h))
    assert base.shape == (256,)
    # Doubling the shared table doubles the logits computed from the
    # same activation.
    w2 = dict(weights[0])
    w2['emb'] = weights[0]['emb'] * 2.0
    doubled = np.asarray(codec.logits(w2, None, h))
    assert np.allclose(doubled, 2 * base, atol=1e-4)
    # ...and changes the input encoding too: one tensor, two roles.
    enc1 = np.asarray(codec.encode(weights[0], jnp.array([[7]]), padding=0))
    enc2 = np.asarray(codec.encode(w2, jnp.array([[7]]), padding=0))
    assert np.allclose(enc2, 2 * enc1, atol=1e-5)


def test_power2_rule_is_search_only():
    md = parse_model2(
        "tokens.256000.gemma.emb.2560|rmsnorm", Precision.FP32)
    assert not md.is_valid()  # 2560 off-grid: load-only
    # But it parses, sizes correctly, and counts weights: table + scale.
    assert md.codec.size == 2560
    assert md.codec.num_weights == 256000 * 2560 + 1


def test_binary_two_logits_no_padding():
    _, model, weights = _build("bits.1.emb.4|rnn.4.tanh")
    batch = jnp.zeros((1, 8), dtype=jnp.int32)
    logits = np.asarray(model.forward(weights, batch))
    assert logits.shape == (1, 8, 2)
    # Both logits are real scores (no hardwired zero reference).
    assert not np.all(logits[..., 1] == 0.0)


def test_step_matches_forward_with_positions():
    _, model, weights = _build("bits.1.emb.4|rnn.4.tanh")
    batch = jax.random.randint(
        jax.random.PRNGKey(2), (1, 12), 0, 2).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    assert np.allclose(np.asarray(logits0), fwd[0, 0], atol=1e-5)
    for t in range(11):
        states, logits = model.step(weights, states, batch[0, t])
        assert np.allclose(np.asarray(logits), fwd[0, t + 1], atol=1e-5), t


def test_padding_continues_positions_negatively():
    md, _, weights = _build("bits.2.emb.4|dense.4.tanh")
    codec = md.codec.build_jax()
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    enc = np.asarray(codec.encode(weights[0], tokens, padding=1))
    # Pad slot: zero value embedding + position (-1 % 4 == 3), scaled.
    s = math.exp(float(np.asarray(weights[0]['y'])))
    expected = s * np.asarray(weights[0]['pos'][3])
    assert np.allclose(enc[0, 0], expected, atol=1e-5)


def test_logits_are_the_raw_tied_scores():
    # No soft-cap since 2026-08-19 (see docs/io.md): the tied scoring
    # matmul is the whole output path, so it stays exactly linear in
    # the table however large the scores get.
    md, model, weights = _build("bytes.emb.4|dense.4.tanh")
    codec = md.codec.build_jax()
    h = jnp.array([300.0, -200.0, 100.0, 50.0], dtype=jnp.float32)
    base = np.asarray(codec.logits(weights[0], None, h))
    assert np.max(np.abs(base)) > 30.0
    big = dict(weights[0])
    big['emb'] = weights[0]['emb'] * 1e4
    assert np.allclose(
        np.asarray(codec.logits(big, None, h)), 1e4 * base, rtol=1e-4)
    # The whole model still produces finite logits and loss.
    batch = jnp.zeros((1, 4), dtype=jnp.int32)
    assert np.all(np.isfinite(np.asarray(model.forward(weights, batch))))
    assert np.isfinite(float(model.loss_batch(weights, batch)))


def test_gradients_reach_table_and_scale():
    _, model, weights = _build("bytes.emb.4|dense.4.tanh")
    batch = jax.random.randint(
        jax.random.PRNGKey(3), (2, 8), 0, 256).astype(jnp.int32)
    grads = jax.grad(model.loss_batch)(weights, batch)
    assert float(jnp.abs(grads[0]['emb']).sum()) > 0
    assert float(jnp.abs(grads[0]['y']).sum()) > 0
    assert float(jnp.abs(grads[1][0]['w']).sum()) > 0  # chain dense


def test_input_neighbors_mode_swaps_and_domain_ladder():
    md = parse_model2("bits.2.emb.4|dense.4.tanh", Precision.FP32)
    assert md.input.neighbors(4) == (
        "bits.2.oh+bp", "bits.1.emb.4", "bits.4.emb.4")
    md = parse_model2("bytes.emb.8|dense.8.tanh", Precision.FP32)
    assert md.input.neighbors(8) == (
        "bytes", "bits.4.emb.8", "tokens.128.fold.emb.8")
    md = parse_model2("bits.1.emb.4|rnn.4.tanh", Precision.FP32)
    assert md.input.neighbors(4) == ("bits.1+bp", "bits.2.emb.4")
    md = parse_model2("tokens.16.test.emb.8|dense.8.tanh", Precision.FP32)
    assert md.input.neighbors(8) == ("tokens.16.test.oh",)


def test_output_metadata_for_consumers():
    md = parse_model2("bytes.emb.8|dense.8.tanh", Precision.FP32)
    assert md.output.size == 256      # logit count
    assert md.output.input_size == 8  # last width == X
    md = parse_model2("bits.1.emb.4|rnn.4.tanh", Precision.FP32)
    assert md.output.size == 2


def test_bad_specs_rejected():
    for bad in ("bytes.emb|dense.4.tanh", "bits.3.emb.4|dense.4.tanh",
                "tokens.16.test.emb|dense.4.tanh",
                "bytes.emb.4.direct|dense.4.tanh"):
        with pytest.raises(ValueError):
            parse_model2(bad, Precision.FP32)


def test_fold_tokenset_counts_frequency_weights():
    md = parse_model2("tokens.32.fold.emb.8|rnn.8.tanh", Precision.FP32)
    # 64 extra = 32 head-token selections + 32 stored head
    # frequencies (raised from 32 on 2026-07-29 to also price the
    # selection, the same 1-per-choice charge hexbpe pays; DB
    # weights migrated). Absolute pin -- num_weights is part of a
    # conf's fleet-wide identity: table 32*8 + scale 1 + 64 stored.
    assert md.codec.num_weights == 32 * 8 + 1 + 64 == 321


def test_hexbpe_counts_its_stored_corpus_knowledge():
    # stats.extra_weights: 17 at 32 tokens, 428 at 256.
    md = parse_model2("tokens.32.hexbpe.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.num_weights == 32 * 8 + 1 + 17
    md = parse_model2("tokens.256.hexbpe.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.num_weights == 256 * 8 + 1 + 428


def test_missing_tokenset_raises_rather_than_guessing():
    # A silent 0 would give the same spec different weight counts on
    # machines with and without the tokenset -- forking conf identity.
    md = parse_model2("tokens.32.nosuchset.emb.8|rnn.8.tanh", Precision.FP32)
    with pytest.raises(RuntimeError, match=r"tokens\.32\.nosuchset"):
        md.codec.num_weights


def test_fold_emb_bridges_the_bit_ladder():
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    assert "tokens.32.fold.emb.8" in md.codec.neighbors(8)
    md = parse_model2("tokens.32.fold.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.is_valid()
    assert md.codec.neighbors(8) == (
        "tokens.32.fold.oh", "bits.4.emb.8", "tokens.32.hexbpe.emb.8",
        "tokens.32.shift.emb.8")


def test_hexbpe_emb_bridges_the_bit_ladder_and_fold():
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    assert "tokens.32.hexbpe.emb.8" in md.codec.neighbors(8)
    md = parse_model2("tokens.32.hexbpe.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.is_valid()
    assert md.codec.neighbors(8) == (
        "tokens.32.hexbpe.oh", "bits.4.emb.8", "tokens.32.fold.emb.8",
        "tokens.64.hexbpe.emb.8", "tokens.32.shift.emb.8")


def test_hexbpe_emb_chain_is_symmetric():
    chain = (32, 64, 128, 256)
    for i, n in enumerate(chain):
        md = parse_model2(
            f"tokens.{n}.hexbpe.emb.8|rnn.8.tanh", Precision.FP32)
        nbs = md.codec.neighbors(8)
        for j in (i - 1, i + 1):
            if 0 <= j < len(chain):
                assert f"tokens.{chain[j]}.hexbpe.emb.8" in nbs, (n, nbs)
        for j, m in enumerate(chain):
            if abs(i - j) > 1:
                assert f"tokens.{m}.hexbpe.emb.8" not in nbs


def test_shift64_emb_bridges_hexbpe64_and_shift32():
    """Re-enabled 2026-07-29 (shift beat hexbpe-64 on the frontier):
    it hangs off the hexbpe chain at its own size, bidirectionally,
    and since 2026-07-31 also borders its 32-token sibling."""
    md = parse_model2("tokens.64.shift.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.is_valid()
    assert md.codec.neighbors(8) == (
        "tokens.64.shift.oh", "tokens.64.hexbpe.emb.8",
        "tokens.32.shift.emb.8", "tokens.128.fold.emb.8")
    md = parse_model2("tokens.64.hexbpe.emb.8|rnn.8.tanh", Precision.FP32)
    assert "tokens.64.shift.emb.8" in md.codec.neighbors(8)
    # Only hexbpe-64, shift-32 and fold-128 reach the shift-64 rung.
    for spec in ("bytes.emb.8", "bits.1.emb.8", "bits.2.emb.8",
                 "bits.4.emb.8", "tokens.32.fold.emb.8",
                 "tokens.32.hexbpe.emb.8",
                 "tokens.128.hexbpe.emb.8", "tokens.256.hexbpe.emb.8"):
        md = parse_model2(f"{spec}|rnn.8.tanh", Precision.FP32)
        assert "tokens.64.shift.emb.8" not in md.codec.neighbors(8), spec


def test_shift32_emb_sits_between_the_32_token_incumbents():
    md = parse_model2("tokens.32.shift.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.is_valid()
    assert md.codec.neighbors(8) == (
        "tokens.32.shift.oh", "tokens.32.fold.emb.8",
        "tokens.32.hexbpe.emb.8", "tokens.64.shift.emb.8")
    # Every edge is reciprocal.
    for spec in ("tokens.32.fold.emb.8", "tokens.32.hexbpe.emb.8",
                 "tokens.64.shift.emb.8"):
        back = parse_model2(f"{spec}|rnn.8.tanh", Precision.FP32)
        assert "tokens.32.shift.emb.8" in back.codec.neighbors(8), spec
    # No direct bits.4 edge: the bit ladder reaches shift-32 through
    # fold or hexbpe.
    md = parse_model2("bits.4.emb.8|rnn.8.tanh", Precision.FP32)
    assert "tokens.32.shift.emb.8" not in md.codec.neighbors(8)


def test_shift32_counts_its_selections_and_pairs():
    # 58 = 29 token selections + 29 stored shift pairs; the two
    # uniform buckets store nothing. table 32*8 + scale 1 + 58.
    md = parse_model2("tokens.32.shift.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.num_weights == 32 * 8 + 1 + 58
    assert md.codec.num_weights == 315


def test_fold128_emb_bridges_shift64_bytes_and_hexbpe128():
    md = parse_model2("tokens.128.fold.emb.8|rnn.8.tanh", Precision.FP32)
    assert md.codec.is_valid()
    assert md.codec.neighbors(8) == (
        "tokens.128.fold.oh", "tokens.64.shift.emb.8", "bytes.emb.8",
        "tokens.128.hexbpe.emb.8")
    # 127 selection weights; the uniform bucket stores nothing.
    assert md.codec.num_weights == 128 * 8 + 1 + 127
