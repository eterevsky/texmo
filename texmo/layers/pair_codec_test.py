"""Tests for PairCodec, the hex-pair IO family over whole bytes:
the additive arm `bits.4.pair.add` and the multiplicative arm
`bits.4.pair.K`."""
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
    # The additive arm's edges: out of the family to the two adjacent
    # 32-wide representations, and across to the multiplicative arm at
    # the weight-comparable k.
    assert md.input.neighbors(8) == (
        'bits.4.oh+bp', 'tokens.32.hexbpe.oh', 'bits.4.pair.16')
    back = parse_model2('bits.4.oh+bp|rnn.8.gelu', Precision.FP32)
    assert 'bits.4.pair.add' in back.input.neighbors(8)
    # Only bits.4.oh+bp and hexbpe-32 reach the pair family.
    for spec in ('bytes', 'bits.1+bp', 'bits.2.oh+bp',
                 'tokens.32.fold.oh', 'tokens.32.shift.oh',
                 'tokens.64.hexbpe.oh'):
        other = parse_model2(f'{spec}|rnn.8.gelu', Precision.FP32)
        nbs = other.input.neighbors(8)
        assert not any(n.startswith('bits.4.pair') for n in nbs), spec


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


# =========================================================================
# The multiplicative arm: bits.4.pair.K
# =========================================================================

MSPEC = 'bits.4.pair.16|dense.8.gelu'


def _randomize_mult_head(weights, seed=11):
    """Fill the zero-initialized biases so the static coupling and the
    lo prior are both live -- init leaves b_v and b_u at zero."""
    kv, ku = jax.random.split(jax.random.PRNGKey(seed))
    w = list(weights)
    head = dict(w[2])
    head['b_v'] = 0.5 * jax.random.normal(kv, head['b_v'].shape)
    head['b_u'] = 0.5 * jax.random.normal(ku, head['b_u'].shape)
    w[2] = head
    return w


def _nbs(spec):
    md = parse_model2(f'{spec}|rnn.8.gelu', Precision.FP32)
    return md.input.neighbors(8)


def test_mult_parse_fields_and_roundtrip():
    for k in (4, 8, 16, 64, 256):
        spec = f'bits.4.pair.{k}|dense.8.gelu'
        md = parse_model2(spec, Precision.FP32)
        assert isinstance(md.codec, PairCodecDef)
        assert md.codec.k == k
        assert str(md.codec) == f'bits.4.pair.{k}'
        assert md.spec == spec
        assert parse_model2(md.spec, Precision.FP32).spec == spec
        assert md.is_valid()
        # The input side is the additive arm's, verbatim.
        assert md.codec.size == 32
        assert md.codec.ntokens == 256
        assert md.codec.tokens_name == 'bytes'
        assert md.output.size == 256
        assert md.output.input_size == 8


def test_every_k_at_least_one_parses_and_is_valid():
    """Nothing about k is a validity rule: the search walks the powers
    of two because that is what the ladder edges generate, not because
    other k are rejected. Off-grid k stays available by hand."""
    for k in (1, 2, 3, 4, 5, 12, 64, 100):
        md = parse_model2(f'bits.4.pair.{k}|dense.8.gelu', Precision.FP32)
        assert md.codec.k == k
        assert md.spec == f'bits.4.pair.{k}|dense.8.gelu'
        assert md.is_valid(), k
        assert md.codec.num_weights == (16 + k) * 8 + 33 * k + 32


def test_k_zero_is_the_additive_arm_not_a_degenerate_k():
    with pytest.raises(ValueError, match='additive arm'):
        parse_model2('bits.4.pair.0|dense.8.gelu', Precision.FP32)


def test_k_one_actually_runs():
    """Parsing is not enough -- the smallest possible coupling has to
    build, forward, and produce a finite normalized loss."""
    md, model, weights = _build('bits.4.pair.1|rnn.8.gelu')
    assert md.codec.k == 1
    assert md.codec.num_weights == (16 + 1) * 8 + 33 + 32
    batch = jax.random.randint(
        jax.random.PRNGKey(31), (2, 12), 0, 256).astype(jnp.int32)
    logits = model.forward(weights, batch)
    assert logits.shape == (2, 12, 256)
    assert np.allclose(
        np.asarray(jax.nn.logsumexp(logits, axis=-1)), 0.0, atol=1e-5)
    loss = float(model.loss_batch(weights, batch))
    assert np.isfinite(loss) and loss > 0
    # A single channel still gradient-flows through every block.
    grads = jax.grad(model.loss_batch)(weights, batch)
    for key in ('w1', 'b1', 'v', 'b_v', 'a', 'u', 'b_u'):
        assert float(jnp.abs(grads[2][key]).sum()) > 0, key


def test_mult_bad_spellings_rejected():
    for bad in ('bits.4.pair.16+bp', 'bits.4.pair.16+bm',
                'bits.4.pair.16.emb.8', 'bits.4.pair.-16',
                'bits.4.pair.16.add', 'bits.4.pair.k',
                'bits.8.pair.16', 'bits.4.pair.16.32'):
        with pytest.raises(ValueError):
            parse_model2(f'{bad}|dense.8.gelu', Precision.FP32)


def test_mult_family_error_names_both_arms():
    with pytest.raises(ValueError, match='FAMILY') as exc:
        PairCodecDef.from_spec('bits.4.pair')
    msg = str(exc.value)
    assert 'bits.4.pair.add' in msg and 'bits.4.pair.16' in msg


def test_mult_num_weights_formula():
    # w1 (16X) + b1 (16) + v (kX) + b_v (k) + a (16k) + u (16k)
    # + b_u (16) = (16 + k)X + 33k + 32.
    for k in (4, 8, 16, 64):
        for x in (1, 8, 32):
            md = parse_model2(f'bits.4.pair.{k}|dense.{x}.gelu',
                              Precision.FP32)
            assert md.codec.num_weights == (16 + k) * x + 33 * k + 32
            assert md.codec.num_mults == md.codec.num_weights
    # k = 16 is the weight-comparable point: same X-slope as .add
    # (32X + 560 vs 32X + 288), which is why the toggle edge sits there.
    for x in (8, 32):
        mult = parse_model2(f'bits.4.pair.16|dense.{x}.gelu',
                            Precision.FP32).codec.num_weights
        add = parse_model2(f'bits.4.pair.add|dense.{x}.gelu',
                           Precision.FP32).codec.num_weights
        assert mult == 32 * x + 560
        assert add == 32 * x + 288
        assert mult - add == 272  # a constant: identical X-slope
    # The declared count matches the actual pytree.
    md, _, weights = _build(MSPEC)
    total = sum(int(np.asarray(v).size) for v in jax.tree.leaves(weights))
    assert md.num_weights == (32 * 8 + 8) + (32 * 8 + 560)
    assert total == md.num_weights
    assert weights[0] is None  # parameter-free input codebook


def test_mult_shares_the_additive_input_side():
    add = parse_model2('bits.4.pair.add|dense.8.gelu',
                       Precision.FP32).codec.build_jax()
    mult = parse_model2(MSPEC, Precision.FP32).codec.build_jax()
    tokens = jnp.array([[0x00, 0x4A, 0xFF, 0x10]], dtype=jnp.int32)
    assert np.array_equal(
        np.asarray(add.encode(None, tokens, padding=2)),
        np.asarray(mult.encode(None, tokens, padding=2)))
    assert mult.init_state() is None


# -- the composed contract, multiplicative arm ----------------------------


def test_mult_composed_logits_are_normalized_log_probs():
    md, model, weights = _build('bits.4.pair.16|rnn.8.gelu')
    weights = _randomize_mult_head(weights)
    codec = md.codec.build_jax()
    h = jnp.array([0.7, -1.3, 0.2, 2.0, -0.5, 0.1, 1.1, -0.9])
    logits = np.asarray(codec.logits(None, weights[2], h))
    assert logits.shape == (256,)
    assert abs(float(jax.nn.logsumexp(jnp.asarray(logits)))) < 1e-5
    batch = jax.random.randint(
        jax.random.PRNGKey(1), (2, 6), 0, 256).astype(jnp.int32)
    fwd = model.forward(weights, batch)
    assert fwd.shape == (2, 6, 256)
    assert np.allclose(
        np.asarray(jax.nn.logsumexp(fwd, axis=-1)), 0.0, atol=1e-5)


def test_mult_cross_entropy_is_ce_hi_plus_ce_lo_given_hi():
    """The chain rule, by hand from the gated channels."""
    md, _, weights = _build('bits.4.pair.8|rnn.8.gelu')
    weights = _randomize_mult_head(weights)
    head = weights[2]
    codec = md.codec.build_jax()
    h = jnp.array([0.3, 1.4, -0.7, 0.05, -1.2, 0.9, 0.4, -0.3])
    logits = np.asarray(codec.logits(None, head, h))

    t = np.asarray(h @ head['w1'].T + head['b1'])
    g = np.asarray(h @ head['v'].T + head['b_v'])
    a, u, b_u = (np.asarray(head[n]) for n in ('a', 'u', 'b_u'))

    def log_softmax(v):
        v = v - v.max()
        return v - np.log(np.exp(v).sum())

    for byte in (0, 1, 15, 16, 0x4A, 200, 255):
        hi, lo = byte >> 4, byte & 15
        ce_hi = -log_softmax(t)[hi]
        ce_lo = -log_softmax(u @ (g * a[:, hi]) + b_u)[lo]
        ce = -(logits[byte] - float(jax.nn.logsumexp(jnp.asarray(logits))))
        assert abs(ce - (ce_hi + ce_lo)) < 1e-4, hex(byte)


def test_mult_einsum_grid_matches_the_per_hi_loop():
    """The closed-form (16 hi, 16 lo) grid against the definition, one
    hi at a time -- this is what pins the einsum's axis order."""
    md, _, weights = _build('bits.4.pair.8|rnn.8.gelu')
    weights = _randomize_mult_head(weights)
    head = weights[2]
    codec = md.codec.build_jax()
    h = jax.random.normal(jax.random.PRNGKey(5), (8,))
    grid = np.asarray(codec._lo_logits(head, h))
    assert grid.shape == (16, 16)
    g = np.asarray(h @ head['v'].T + head['b_v'])
    a, u, b_u = (np.asarray(head[n]) for n in ('a', 'u', 'b_u'))
    for hi in range(16):
        assert np.allclose(grid[hi], u @ (g * a[:, hi]) + b_u, atol=1e-5)
    # Batched leading axes go through the same path, position by position.
    hb = jax.random.normal(jax.random.PRNGKey(6), (2, 3, 8))
    gridb = np.asarray(codec._lo_logits(head, hb))
    assert gridb.shape == (2, 3, 16, 16)
    for i in range(2):
        for j in range(3):
            one = np.asarray(codec._lo_logits(head, hb[i, j]))
            assert np.allclose(gridb[i, j], one, atol=1e-5)


def _grid(codec, head, h):
    p = np.exp(np.asarray(codec.logits(None, head, h)))
    return p.reshape(16, 16)


def _is_outer_product(p, atol):
    return np.allclose(p, p.sum(1)[:, None] * p.sum(0)[None, :], atol=atol)


def test_mult_static_coupling_lives_in_b_v():
    """With V = 0 the head is h-independent -- and `b_v * a[:, hi]`
    still generates a genuine rank-k static coupling, the additive
    arm's D-equivalent. Zero b_v as well and the grid collapses to
    rank 1."""
    md, _, weights = _build('bits.4.pair.4|rnn.8.gelu')
    codec = md.codec.build_jax()
    head = dict(weights[2])
    head['v'] = jnp.zeros_like(head['v'])
    h = jnp.array([0.9, -0.2, 0.4, 1.5, -1.0, 0.2, 0.7, -0.6])

    # b_v = 0 too: u(hi) = b_u for every hi -> identical conditionals
    # -> the joint grid is an outer product.
    assert _is_outer_product(_grid(codec, head, h), 1e-6)

    # Switch b_v on: still no dependence on h, but the conditionals now
    # differ per hi, so the grid is no longer rank 1.
    head['b_v'] = jax.random.normal(jax.random.PRNGKey(9),
                                    head['b_v'].shape)
    p = _grid(codec, head, h)
    assert not _is_outer_product(p, 1e-3)
    # ...and it really is h-independent: another h, same conditionals.
    h2 = jnp.array([-1.1, 0.8, 0.0, -0.3, 1.9, -0.7, 0.2, 1.2])
    q = _grid(codec, head, h2)
    cond = lambda m: m / m.sum(axis=1, keepdims=True)
    assert np.allclose(cond(p), cond(q), atol=1e-5)


def test_mult_coupling_is_context_dependent():
    """The whole point of the arm: with V live, P(lo | hi) reshapes
    with the hidden state -- which the additive arm's static D cannot
    do, since there every hi shares one h-dependent lo head."""
    md, _, weights = _build('bits.4.pair.16|rnn.8.gelu')
    weights = _randomize_mult_head(weights)
    codec = md.codec.build_jax()
    cond = lambda h: (lambda m: np.log(m / m.sum(axis=1, keepdims=True)))(
        _grid(codec, weights[2], h))
    deltas = cond(jax.random.normal(jax.random.PRNGKey(21), (8,))) - cond(
        jax.random.normal(jax.random.PRNGKey(22), (8,)))
    # Not one shared shift: the per-hi conditionals move differently.
    assert deltas.std(axis=0).max() > 1e-3


# -- parity, gradients, training ------------------------------------------


def test_mult_step_matches_forward():
    _, model, weights = _build('bits.4.pair.16|rnn.8.gelu')
    weights = _randomize_mult_head(weights)
    batch = jax.random.randint(
        jax.random.PRNGKey(2), (1, 10), 0, 256).astype(jnp.int32)
    fwd = np.asarray(model.forward(weights, batch))
    states, logits0 = model.initial_step(weights)
    assert np.allclose(np.asarray(logits0), fwd[0, 0], atol=1e-5)
    for t in range(9):
        states, logits = model.step(weights, states, batch[0, t])
        assert np.allclose(np.asarray(logits), fwd[0, t + 1], atol=1e-5), t


def test_mult_gradients_reach_every_head_block():
    _, model, weights = _build('bits.4.pair.8|rnn.8.gelu')
    batch = jax.random.randint(
        jax.random.PRNGKey(3), (2, 16), 0, 256).astype(jnp.int32)
    grads = jax.grad(model.loss_batch)(weights, batch)
    for key in ('w1', 'b1', 'v', 'b_v', 'a', 'u', 'b_u'):
        assert float(jnp.abs(grads[2][key]).sum()) > 0, key
    assert any(float(jnp.abs(g).sum()) > 0 for g in jax.tree.leaves(grads[1]))


def test_mult_end_to_end_trains_on_repetitive_data():
    import optax

    _, model, weights = _build('bits.4.pair.16|rnn.16.gelu')
    data = jnp.asarray(
        np.tile(np.frombuffer(b'abcabcabc ', dtype=np.uint8), (4, 8)),
        dtype=jnp.int32)
    loss0 = float(model.loss_batch(weights, data))
    assert np.isfinite(loss0)
    opt = optax.adam(0.05)
    state = opt.init(weights)
    for _ in range(40):
        grads = jax.grad(model.loss_batch)(weights, data)
        updates, state = opt.update(grads, state)
        weights = optax.apply_updates(weights, updates)
    loss1 = float(model.loss_batch(weights, data))
    assert loss1 < loss0 - 2.0, (loss0, loss1)


# -- search wiring, multiplicative arm ------------------------------------


def test_mult_k_ladder_and_toggle_edges():
    # k = 16 carries the cross-family edges plus both k rungs.
    assert _nbs('bits.4.pair.16') == (
        'bits.4.oh+bp', 'tokens.32.hexbpe.oh', 'bits.4.pair.add',
        'bits.4.pair.8', 'bits.4.pair.32')
    # Other k values are pure k-ladder rungs.
    assert _nbs('bits.4.pair.8') == ('bits.4.pair.4', 'bits.4.pair.16')
    assert _nbs('bits.4.pair.32') == ('bits.4.pair.16', 'bits.4.pair.64')
    # The ladder runs all the way down to 1 and has no cap.
    assert _nbs('bits.4.pair.4') == ('bits.4.pair.2', 'bits.4.pair.8')
    assert _nbs('bits.4.pair.2') == ('bits.4.pair.1', 'bits.4.pair.4')
    assert _nbs('bits.4.pair.1') == ('bits.4.pair.2',)  # no half-edge
    assert _nbs('bits.4.pair.256') == ('bits.4.pair.128', 'bits.4.pair.512')


def test_off_grid_k_gets_outgoing_migration_edges_only():
    """No edge ever emits an off-grid k, so one can only be built by
    hand -- and when it is, it gets outgoing edges to the nearest
    rungs below and above so the population can walk onto the grid.
    The bits.1+bm migration-bridge precedent."""
    assert _nbs('bits.4.pair.3') == ('bits.4.pair.2', 'bits.4.pair.4')
    assert _nbs('bits.4.pair.5') == ('bits.4.pair.4', 'bits.4.pair.8')
    assert _nbs('bits.4.pair.12') == ('bits.4.pair.8', 'bits.4.pair.16')
    assert _nbs('bits.4.pair.100') == ('bits.4.pair.64', 'bits.4.pair.128')
    # Off-grid k carries no cross-family or arm-toggle edges...
    for k in (3, 5, 12, 100):
        nbs = _nbs(f'bits.4.pair.{k}')
        assert 'bits.4.oh+bp' not in nbs
        assert 'tokens.32.hexbpe.oh' not in nbs
        assert 'bits.4.pair.add' not in nbs
    # ...and nothing anywhere points back at one.
    for spec in ('bits.4.pair.add', 'bits.4.pair.1', 'bits.4.pair.2',
                 'bits.4.pair.4', 'bits.4.pair.8', 'bits.4.pair.16',
                 'bits.4.pair.64', 'bits.4.oh+bp', 'tokens.32.hexbpe.oh'):
        for nb in _nbs(spec):
            if nb.startswith('bits.4.pair.') and nb != 'bits.4.pair.add':
                k = int(nb.rsplit('.', 1)[1])
                assert k & (k - 1) == 0, (spec, nb)
    # Migration targets really are reachable, valid models.
    for nb in _nbs('bits.4.pair.5'):
        assert parse_model2(f'{nb}|rnn.8.gelu', Precision.FP32).is_valid()


def test_mult_every_edge_is_reciprocal():
    # On-grid k only: the off-grid migration bridges are deliberately
    # one-way (see test_off_grid_k_gets_outgoing_migration_edges_only).
    for spec in ('bits.4.pair.add', 'bits.4.pair.1', 'bits.4.pair.2',
                 'bits.4.pair.4', 'bits.4.pair.8',
                 'bits.4.pair.16', 'bits.4.pair.32', 'bits.4.oh+bp',
                 'tokens.32.hexbpe.oh'):
        for nb in _nbs(spec):
            in_family = (nb.startswith('bits.4.pair')
                         or spec.startswith('bits.4.pair'))
            if not in_family:
                continue
            assert spec in _nbs(nb), (spec, nb)


def test_mult_bridges_the_one_hot_ladder_both_ways():
    back = _nbs('bits.4.oh+bp')
    assert 'bits.4.pair.16' in back
    # Only k = 16 hangs off the ladder; other rungs go through it.
    for k in (4, 8, 32, 64):
        assert f'bits.4.pair.{k}' not in back


def test_hexbpe32_borders_both_arms():
    """32-wide IO on both sides, and hexbpe's encoding is itself a mix
    of whole-byte tokens and hex-digit fallbacks -- adjacent enough to
    cross in one step instead of routing through the bit ladder."""
    hexbpe = _nbs('tokens.32.hexbpe.oh')
    assert 'bits.4.pair.add' in hexbpe
    assert 'bits.4.pair.16' in hexbpe
    # Both directions, and only at the weight-comparable k.
    assert 'tokens.32.hexbpe.oh' in _nbs('bits.4.pair.add')
    assert 'tokens.32.hexbpe.oh' in _nbs('bits.4.pair.16')
    for k in (4, 8, 32, 64):
        assert f'bits.4.pair.{k}' not in hexbpe
        assert 'tokens.32.hexbpe.oh' not in _nbs(f'bits.4.pair.{k}')
    # The rest of hexbpe-32's own ladder is untouched.
    for keep in ('bits.4.oh+bp', 'tokens.32.fold.oh',
                 'tokens.64.hexbpe.oh', 'tokens.32.shift.oh'):
        assert keep in hexbpe, keep


def test_mult_neighbor_models_parse_and_stay_valid():
    md = parse_model2('bits.4.pair.16|rnn.8.gelu', Precision.FP32)
    specs = [str(n) for n in md.neighbors()]
    for expected in ('bits.4.oh+bp|rnn.8.gelu', 'bits.4.pair.add|rnn.8.gelu',
                     'bits.4.pair.8|rnn.8.gelu', 'bits.4.pair.32|rnn.8.gelu'):
        assert expected in specs, expected
    assert all(parse_model2(s, Precision.FP32).is_valid() for s in specs)
