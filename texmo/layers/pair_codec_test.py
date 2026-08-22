"""Tests for PairCodec (`bits.4.pair.add`: the additive arm of the
hex-pair IO family, over whole bytes)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.layers.one_hot_codec import OneHotCodecDef
from texmo.layers.pair_codec import PairCodecDef
from texmo.precision import Precision
from texmo.spec_parser import parse_model2

SPEC = 'bits.4.pair.add|dense.8.gelu'


def _build(spec=SPEC, seed=0):
    md = parse_model2(spec, Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(seed))
    return md, model, weights


def _randomize_head(weights, seed=7):
    """Non-degenerate heads: xavier w1/w2 but zero biases and zero
    coupling out of the box, so tests that need a real D fill it in."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    w = list(weights)
    head = dict(w[2])
    head['b1'] = 0.5 * jax.random.normal(k1, head['b1'].shape)
    head['b2'] = 0.5 * jax.random.normal(k2, head['b2'].shape)
    head['d'] = jax.random.normal(k3, head['d'].shape)
    w[2] = head
    return w


# -- parsing / structure --------------------------------------------------


def test_parse_fields_and_roundtrip():
    md = parse_model2(SPEC, Precision.FP32)
    assert isinstance(md.codec, PairCodecDef)
    assert md.spec == SPEC
    assert parse_model2(md.spec, Precision.FP32).spec == SPEC
    assert md.codec.ntokens == 256
    assert md.codec.size == 32          # two 16-value one-hots
    assert md.codec.tokens_name == 'bytes'  # raw bytes, 1.0 B/token
    assert md.is_valid()
    assert md.input is md.codec
    assert md.output is md.codec.head
    assert md.output.size == 256
    assert md.output.input_size == 8


def test_bp_and_bm_and_emb_suffixes_rejected():
    # One position per byte: there is no phase to transmit, and there
    # is no tied variant.
    for bad in ('bits.4.pair.add+bp', 'bits.4.pair.add+bm',
                'bits.4.pair.add.emb.8', 'bits.8.pair.add',
                'bits.2.pair.add', 'bits.4.oh.pair.add'):
        with pytest.raises(ValueError):
            parse_model2(f'{bad}|dense.8.gelu', Precision.FP32)


def test_bare_family_name_is_rejected():
    """The arm is always explicit: `bits.4.pair` names the family,
    which the multiplicative sibling `bits.4.pair.K` also belongs to,
    so it must not silently default to the additive arm."""
    with pytest.raises(ValueError, match='FAMILY'):
        parse_model2('bits.4.pair|dense.8.gelu', Precision.FP32)
    with pytest.raises(ValueError, match='FAMILY'):
        PairCodecDef.from_spec('bits.4.pair')


def test_num_weights_formula():
    # 2*16*X (w1, w2) + 16*16 (coupling D) + 32 (b1, b2) = 32X + 288.
    for x in (1, 4, 8, 32):
        md = parse_model2(f'bits.4.pair.add|dense.{x}.gelu', Precision.FP32)
        assert md.codec.num_weights == 32 * x + 288
        assert md.codec.num_mults == 32 * x + 288
        assert md.output.num_weights == 32 * x + 288
        # Cheaper than the 256-way head it replaces at every width the
        # search can reach (crossover at X = 1/7).
        assert md.codec.num_weights < 256 * x + 256
    # The declared count matches the actual pytree.
    md, _, weights = _build('bits.4.pair.add|dense.8.gelu')
    total = sum(int(np.asarray(v).size) for v in jax.tree.leaves(weights))
    # chain dense.8.gelu from the 32-wide input: 32*8 + 8.
    assert md.num_weights == (32 * 8 + 8) + (32 * 8 + 288)
    assert total == md.num_weights
    assert weights[0] is None  # parameter-free input codebook


# -- input encoding -------------------------------------------------------


def test_encode_is_two_concatenated_hex_one_hots():
    md, _, _ = _build()
    codec = md.codec.build_jax()
    tokens = jnp.array([[0x00, 0x4A, 0xFF, 0x10]], dtype=jnp.int32)
    enc = np.asarray(codec.encode(None, tokens, padding=0))
    assert enc.shape == (1, 4, 32)
    for i, b in enumerate((0x00, 0x4A, 0xFF, 0x10)):
        expected = np.zeros(32, dtype=np.float32)
        expected[b >> 4] = 1.0
        expected[16 + (b & 15)] = 1.0
        assert np.array_equal(enc[0, i], expected), hex(b)


def test_initial_vector_is_max_entropy_and_stateless():
    md, _, _ = _build()
    codec = md.codec.build_jax()
    assert codec.init_state() is None
    v = np.asarray(codec.initial_vector(None))
    assert np.allclose(v, 1.0 / 16)
    # No position cycle: every padding slot is the same vector.
    enc = np.asarray(
        codec.encode(None, jnp.array([[65, 66]], dtype=jnp.int32), padding=3))
    assert enc.shape == (1, 5, 32)
    assert np.allclose(enc[0, :3], 1.0 / 16)


def test_encode_step_matches_encode():
    md, _, _ = _build()
    codec = md.codec.build_jax()
    tokens = jnp.array([[3, 200, 17]], dtype=jnp.int32)
    enc = np.asarray(codec.encode(None, tokens, padding=0))
    state = codec.init_state()
    for i in range(3):
        state, v = codec.encode_step(None, state, tokens[0, i])
        assert np.array_equal(np.asarray(v), enc[0, i])


# -- the composed 256-way contract ----------------------------------------


def test_composed_logits_are_normalized_log_probs():
    md, _, weights = _build('bits.4.pair.add|rnn.8.gelu')
    weights = _randomize_head(weights)
    codec = md.codec.build_jax()
    h = jnp.array([0.7, -1.3, 0.2, 2.0, -0.5, 0.1, 1.1, -0.9])
    logits = np.asarray(codec.logits(None, weights[2], h))
    assert logits.shape == (256,)
    # Already log-probabilities: logsumexp == 0.
    assert abs(float(jax.nn.logsumexp(jnp.asarray(logits)))) < 1e-5
    # Batched forward too, at every position.
    model = md.build_jax()
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 6), 0, 256).astype(jnp.int32)
    fwd = model.forward(weights, batch)
    assert fwd.shape == (2, 6, 256)
    lse = np.asarray(jax.nn.logsumexp(fwd, axis=-1))
    assert np.allclose(lse, 0.0, atol=1e-5)


def test_cross_entropy_is_ce_hi_plus_ce_lo_given_hi():
    """The chain rule, by hand from the two heads: teacher forcing is
    exactly what the composed 256-way CE computes."""
    md, _, weights = _build('bits.4.pair.add|rnn.8.gelu')
    weights = _randomize_head(weights)
    head = weights[2]
    codec = md.codec.build_jax()
    h = jnp.array([0.3, 1.4, -0.7, 0.05, -1.2, 0.9, 0.4, -0.3])
    logits = np.asarray(codec.logits(None, head, h))

    t = np.asarray(h @ head['w1'].T + head['b1'])
    s = np.asarray(h @ head['w2'].T + head['b2'])
    d = np.asarray(head['d'])

    def log_softmax(v):
        v = v - v.max()
        return v - np.log(np.exp(v).sum())

    for byte in (0, 1, 15, 16, 0x4A, 200, 255):
        hi, lo = byte >> 4, byte & 15
        ce_hi = -log_softmax(t)[hi]
        # d[:, hi] is the offset vector over lo for this hi.
        ce_lo = -log_softmax(s + d[:, hi])[lo]
        # Softmax CE against the composed logits (they normalize to 0,
        # but do it the general way anyway).
        ce = -(logits[byte] - float(jax.nn.logsumexp(jnp.asarray(logits))))
        assert abs(ce - (ce_hi + ce_lo)) < 1e-4, hex(byte)


def test_coupling_matrix_shifts_only_the_conditional():
    """D moves P(lo | hi) per hi and leaves P(hi) alone -- the whole
    point of the factorization."""
    md, _, weights = _build('bits.4.pair.add|rnn.8.gelu')
    weights = _randomize_head(weights)
    codec = md.codec.build_jax()
    h = jnp.array([0.2, -0.4, 1.0, 0.3, -0.8, 0.6, -0.1, 0.5])
    base = np.exp(np.asarray(codec.logits(None, weights[2], h)))

    bumped = dict(weights[2])
    d = np.asarray(bumped['d']).copy()
    d[:, 3] += np.arange(16) * 0.5   # only hi == 3's conditional
    bumped['d'] = jnp.asarray(d)
    other = np.exp(np.asarray(codec.logits(None, bumped, h)))

    p_hi_base = base.reshape(16, 16).sum(axis=1)
    p_hi_other = other.reshape(16, 16).sum(axis=1)
    assert np.allclose(p_hi_base, p_hi_other, atol=1e-5)
    # The hi == 3 conditional really moved; hi == 4's did not.
    cond = lambda p, i: p.reshape(16, 16)[i] / p.reshape(16, 16)[i].sum()
    assert not np.allclose(cond(base, 3), cond(other, 3), atol=1e-3)
    assert np.allclose(cond(base, 4), cond(other, 4), atol=1e-5)


def test_zero_coupling_factorizes_independently():
    # D starts at zero: P(lo | hi) is the same for every hi, so the
    # 16x16 probability grid is a rank-1 outer product.
    md, _, weights = _build('bits.4.pair.add|rnn.8.gelu')
    codec = md.codec.build_jax()
    assert np.array_equal(np.asarray(weights[2]['d']), np.zeros((16, 16)))
    h = jnp.array([0.9, -0.2, 0.4, 1.5, -1.0, 0.2, 0.7, -0.6])
    p = np.exp(np.asarray(codec.logits(None, weights[2], h))).reshape(16, 16)
    outer = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
    assert np.allclose(p, outer, atol=1e-6)


# -- step/forward parity and training -------------------------------------


def test_step_matches_forward():
    _, model, weights = _build('bits.4.pair.add|rnn.8.gelu')
    weights = _randomize_head(weights)
    batch = jax.random.randint(
        jax.random.PRNGKey(2), (1, 10), 0, 256).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    assert np.allclose(np.asarray(logits0), fwd[0, 0], atol=1e-5)
    for t in range(9):
        states, logits = model.step(weights, states, batch[0, t])
        assert np.allclose(np.asarray(logits), fwd[0, t + 1], atol=1e-5), t


def test_gradients_reach_every_head_block():
    _, model, weights = _build('bits.4.pair.add|rnn.8.gelu')
    batch = jax.random.randint(
        jax.random.PRNGKey(3), (2, 16), 0, 256).astype(jnp.int32)
    grads = jax.grad(model.loss_batch)(weights, batch)
    for key in ('w1', 'b1', 'w2', 'b2', 'd'):
        assert float(jnp.abs(grads[2][key]).sum()) > 0, key
    # ...and back through the chain.
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads[1]))


def test_end_to_end_trains_on_repetitive_data():
    import optax

    md, model, weights = _build('bits.4.pair.add|rnn.16.gelu')
    data = jnp.asarray(
        np.tile(np.frombuffer(b'abcabcabc ', dtype=np.uint8), (4, 8)),
        dtype=jnp.int32)
    loss0 = float(model.loss_batch(weights, data))
    assert np.isfinite(loss0)
    # An untrained model is near the 8 bits/byte uniform prior.
    assert 6.0 < loss0 < 9.0

    opt = optax.adam(0.05)
    state = opt.init(weights)
    for _ in range(40):
        grads = jax.grad(model.loss_batch)(weights, data)
        updates, state = opt.update(grads, state)
        weights = optax.apply_updates(weights, updates)
    loss1 = float(model.loss_batch(weights, data))
    assert loss1 < loss0 - 2.0, (loss0, loss1)


# -- search wiring --------------------------------------------------------


def test_toggle_bridge_with_bits4_oh_is_bidirectional():
    md = parse_model2('bits.4.pair.add|rnn.8.gelu', Precision.FP32)
    assert md.input.neighbors(8) == ('bits.4.oh+bp',)
    back = parse_model2('bits.4.oh+bp|rnn.8.gelu', Precision.FP32)
    assert 'bits.4.pair.add' in back.input.neighbors(8)
    # Nothing else reaches the pair codec in v1.
    for spec in ('bytes', 'bits.1+bp', 'bits.2.oh+bp',
                 'tokens.32.fold.oh', 'tokens.32.hexbpe.oh'):
        other = parse_model2(f'{spec}|rnn.8.gelu', Precision.FP32)
        assert 'bits.4.pair.add' not in other.input.neighbors(8), spec


def test_neighbor_models_parse_and_stay_valid():
    md = parse_model2('bits.4.pair.add|rnn.8.gelu', Precision.FP32)
    specs = [str(n) for n in md.neighbors()]
    assert 'bits.4.oh+bp|rnn.8.gelu' in specs
    # The chain mutations carry the pair input unchanged.
    assert any(s.startswith('bits.4.pair.add|') and s != md.spec for s in specs)
    md2 = parse_model2('bits.4.oh+bp|rnn.8.gelu', Precision.FP32)
    assert 'bits.4.pair.add|rnn.8.gelu' in [str(n) for n in md2.neighbors()]


def test_direct_from_spec_rejects_other_codecs_specs():
    with pytest.raises(ValueError):
        PairCodecDef.from_spec('bytes')
    with pytest.raises(ValueError):
        OneHotCodecDef.from_spec('bits.4.pair.add')
