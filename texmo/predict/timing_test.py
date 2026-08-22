"""Tests for the training-run time prediction model."""

import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.precision import Precision
from texmo.predict.timing import (
    CHUNK_SIZE,
    TrainTimingModel,
    Weights,
    _dot,
    _fit,
    chunk_structure,
    featurize,
    predict_total_time,
)
from texmo.run import Run
from texmo.spec_parser import parse_model2


def _conf(
    spec: str = "bits.1+bp|dense.8.gelu",
    lr: float = 0.01,
    length: int = 64,
    batch: int = 16,
    steps: int = 100,
    decay: float = 1.0,
    precision: Precision = Precision.FP32,
) -> Configuration:
    return Configuration(
        parse_model2(spec, precision=precision),
        lr=lr,
        length=length,
        batch=batch,
        steps=steps,
        decay=decay,
    )


def test_chunk_structure():
    assert chunk_structure(2) == (0, 2)
    assert chunk_structure(CHUNK_SIZE) == (1, 0)
    assert chunk_structure(CHUNK_SIZE - 1) == (0, CHUNK_SIZE - 1)
    assert chunk_structure(CHUNK_SIZE + 5) == (1, 5)
    assert chunk_structure(2 * CHUNK_SIZE) == (2, 0)
    assert chunk_structure(2 * CHUNK_SIZE + 1) == (2, 1)


def test_featurize_types():
    conf = _conf("bits.1+bp|dense.8.gelu", batch=16, length=32)
    comps = featurize(conf)
    assert len(comps) == 3
    assert comps[0].type_id == "bits.1+bp"
    assert comps[1].type_id == "dense.gelu"
    assert comps[2].type_id == "output"


def test_featurize_feature_sizes():
    """Different types have different feature vector lengths."""
    conf = _conf(
        "bits.1+bp|dense.8.gelu-suffix.2-split.add(norm, pass)",
        batch=4, length=10,
    )
    comps = featurize(conf)
    types = {c.type_id: c.features.shape[0] for c in comps}
    assert types["bits.1+bp"] == 3   # input
    assert types["dense.gelu"] == 12  # dense_matmul
    assert types["suffix"] == 10      # plain + suffix_length
    assert types["split.add"] == 4    # minimal merge component
    assert types["norm"] == 9         # plain
    assert types["output"] == 12      # dense_matmul


def test_featurize_base_values():
    """Base features: [1, L, B*L, IS, IS*L, IS*B*L, OS, OS*L, OS*B*L]."""
    conf = _conf("bits.1+bp|dense.8.gelu", batch=4, length=10)
    comps = featurize(conf)
    dense = comps[1]
    bl = 40
    expected_base = [1, 10, bl, 4, 40, 160, 8, 80, 320]
    expected_matmul = [32, 320, 4 * 8 * bl]
    np.testing.assert_array_equal(
        dense.features, expected_base + expected_matmul)


def test_featurize_msr_l2_features():
    """msr features extend dense_matmul with [L^2, L^2*OS, L^2*OS*B]."""
    conf = _conf("bits.1+bp|msr.2.4-dense.8.gelu", batch=4, length=10)
    comps = featurize(conf)
    # Order: input, msr, dense, output.
    msr = comps[1]
    assert msr.type_id == "msr"
    # msr OS = heads * dim = 4 * 2 = 8.  IS comes from the bits.1+bp
    # input layer's output (one-hot encoding width).
    os_, b, l = 8, 4, 10
    is_ = int(msr.features[3])  # the IS feature, validated by the base block
    bl = b * l
    expected_base = [1, l, bl, is_, is_ * l, is_ * bl,
                     os_, os_ * l, os_ * bl]
    expected_matmul = [is_ * os_, is_ * os_ * l, is_ * os_ * bl]
    expected_l2 = [l * l, l * l * os_, l * l * os_ * b]
    np.testing.assert_array_equal(
        msr.features, expected_base + expected_matmul + expected_l2)
    # Sanity: there should be 15 features (9 base + 3 matmul + 3 L^2).
    assert msr.features.shape == (15,)


# -- Model2 / split featurization -------------------------------------


def test_featurize_split_components_recurse_into_branches():
    """A split contributes a merge component plus the components of
    every layer in its branches (incl. the bare gate dense)."""
    conf = _conf(
        "bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)-dense.4.tanh",
        batch=8, length=16)
    types = [c.type_id for c in featurize(conf)]
    assert types == [
        "bits.1+bp",   # input
        "dense.gelu",  # pre-split layer
        "split.mul",   # merge
        "dense.gelu",  # main branch
        "dense",       # gate branch (bare dense -- clean key, not dense.None)
        "dense.tanh",  # post-split layer
        "output",
    ]


def test_featurize_split_merge_feature_values():
    """split merge features are [1, OS, OS*L, OS*B*L] (4, like skip)."""
    conf = _conf("bits.1+bp|split.add(dense.4.gelu, pass)", batch=4, length=10)
    split = next(c for c in featurize(conf) if c.type_id == "split.add")
    os_ = int(split.features[1])  # merged output size
    assert split.features.shape == (4,)
    np.testing.assert_array_equal(
        split.features, [1, os_, os_ * 10, os_ * 4 * 10])


def test_fit_and_predict_model2_split():
    """End-to-end fit + predict on a split-containing Model2 spec."""
    rng = np.random.default_rng(7)
    spec = "bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, dense.4)"
    samples = []
    for _ in range(60):
        batch = int(rng.choice([8, 16, 32]))
        length = int(rng.choice([32, 64]))
        steps = int(rng.choice([64, 128, 256]))
        conf = _conf(spec, batch=batch, length=length, steps=steps)
        comps = featurize(conf)
        total = 0.2 * len(comps) + steps * 1e-4 * len(comps)
        samples.append((conf, _fake_run("m2sys", total)))

    model = TrainTimingModel()
    model.fit(samples)
    assert ("m2sys", Precision.FP32) in model.keys()
    pred = model.predict("m2sys", _conf(spec, batch=12, length=48, steps=128))
    assert pred is not None and pred > 0


def test_predict_total_time_with_known_weights():
    """If only T_step has nonzero weights, total = steps * step_per_layer_sum."""
    conf = _conf("bits.1+bp|dense.8.gelu", batch=4, length=10, steps=100)
    comps = featurize(conf)
    c_step = 0.001  # constant per-component step coef
    weights = Weights(
        init={c.type_id: np.zeros(c.features.shape[0]) for c in comps},
        step={
            c.type_id: np.array(
                [c_step] + [0.0] * (c.features.shape[0] - 1)
            )
            for c in comps
        },
        scan_full={comps[0].type_id: np.zeros(3)},
        scan_short={comps[0].type_id: np.zeros(4)},
    )
    pred = predict_total_time(weights, conf)
    # Each component contributes c_step (its constant) per step.
    assert pred == pytest.approx(100 * c_step * len(comps), rel=1e-9)


def test_predict_total_time_includes_init_and_scan():
    conf = _conf("bits.1+bp|dense.8.gelu", batch=4, length=10, steps=8)
    comps = featurize(conf)
    in_t = comps[0].type_id
    weights = Weights(
        init={c.type_id: np.zeros(c.features.shape[0]) for c in comps},
        step={c.type_id: np.zeros(c.features.shape[0]) for c in comps},
        scan_full={in_t: np.zeros(3)},
        scan_short={in_t: np.zeros(4)},
    )
    # T_init for the input layer = 5.0 (constant feature only).
    weights.init[in_t] = np.array([5.0, 0.0, 0.0])
    # 8 steps -> num_full=0, short_steps=8; scan_short fires.
    weights.scan_short[in_t] = np.array([2.0, 0.0, 0.0, 0.0])
    pred = predict_total_time(weights, conf)
    assert pred == pytest.approx(5.0 + 2.0)


def test_fit_recovers_synthetic_components():
    """Synthetic data with known T_init / T_step / T_scan_full coefficients
    fits to small residuals."""
    rng = np.random.default_rng(0)
    spec = "bits.1+bp|dense.8.gelu"

    # Ground-truth coefficients (per-component, only on the constant feature).
    init_per_comp = 0.5
    step_per_comp = 1e-4
    scan_full_const = 0.05

    samples = []
    for _ in range(120):
        batch = int(rng.choice([4, 8, 16]))
        length = int(rng.choice([16, 32, 64]))
        steps = int(rng.choice([64, 128, 256, 512, 1024]))
        conf = _conf(spec, batch=batch, length=length, steps=steps)
        comps = featurize(conf)
        n_full, n_short = chunk_structure(steps)
        t_init = init_per_comp * len(comps)
        t_step = step_per_comp * len(comps)
        t_scan = n_full * scan_full_const
        # No short scan effect on synth data — keep total a clean linear combo.
        if n_short == 0:
            total = t_init + t_scan + steps * t_step
            samples.append((conf, total))
        else:
            # For short-only runs, fold the short-scan cost into init so the
            # fit still succeeds.
            total = t_init + scan_full_const + steps * t_step
            samples.append((conf, total))

    weights, mse = _fit(samples)
    rmse = np.sqrt(mse)
    # Synthetic data is exact; fit should be near-perfect modulo NNLS slop.
    assert rmse < 0.1, f"RMSE too high: {rmse}"


def _fake_run(system: str, train_time: float) -> Run:
    return Run(train_time=train_time, system=system)


def test_model_fit_and_predict_total():
    """End-to-end: synthetic total-time samples, fit, predict total time."""
    rng = np.random.default_rng(1)
    spec = "bits.1+bp|dense.8.gelu"
    step_per_comp = 1e-4
    init_per_comp = 0.2

    samples = []
    for _ in range(120):
        batch = int(rng.choice([8, 16, 32]))
        length = int(rng.choice([32, 64, 128]))
        steps = int(rng.choice([64, 128, 256, 512]))
        conf = _conf(spec, batch=batch, length=length, steps=steps)
        comps = featurize(conf)
        # Synthetic total: constant init + steps * constant per layer.
        # Add modest noise.
        total = init_per_comp * len(comps) + steps * step_per_comp * len(comps)
        total *= 1 + 0.02 * rng.standard_normal()
        samples.append((conf, _fake_run("testbench", total)))

    model = TrainTimingModel()
    model.fit(samples)
    assert ("testbench", Precision.FP32) in model.keys()

    # Predict on a fresh shape; should land within ~30% of the truth.
    conf = _conf(spec, batch=24, length=48, steps=128)
    pred = model.predict("testbench", conf)
    expected = init_per_comp * 3 + 128 * step_per_comp * 3
    assert pytest.approx(pred, rel=0.3) == expected


def test_model_unknown_system_returns_none():
    model = TrainTimingModel()
    conf = _conf()
    assert model.predict("nope", conf) is None
    assert model.predict_batch("nope", [conf]) is None
    assert model.predict_step_time("nope", conf) is None
    assert model.predict_max_steps("nope", conf, 10.0) is None


def test_model_skips_diverged_runs():
    """Runs missing train_time or below MIN_TIME are filtered out."""
    rng = np.random.default_rng(2)
    samples = []
    for _ in range(80):
        batch = int(rng.choice([4, 8, 16]))
        length = int(rng.choice([32, 64]))
        steps = int(rng.choice([64, 128, 256]))
        conf = _conf(batch=batch, length=length, steps=steps)
        if rng.random() < 0.5:
            run = _fake_run("test", None)
        else:
            run = _fake_run("test", 0.05 * steps)
        samples.append((conf, run))

    model = TrainTimingModel()
    model.fit(samples)
    assert ("test", Precision.FP32) in model.keys()


def test_predict_max_steps_rounds_to_pow2():
    """Given a budget, predict_max_steps returns a power-of-two."""
    conf = _conf("bits.1+bp|dense.8.gelu", batch=8, length=32, steps=128)
    comps = featurize(conf)
    in_t = comps[0].type_id

    # Only T_step fires: each component costs 0.001 / step.
    weights = Weights(
        init={c.type_id: np.zeros(c.features.shape[0]) for c in comps},
        step={
            c.type_id: np.array(
                [0.001] + [0.0] * (c.features.shape[0] - 1)
            )
            for c in comps
        },
        scan_full={in_t: np.zeros(3)},
        scan_short={in_t: np.zeros(4)},
    )
    model = TrainTimingModel()
    model._weights[("sysX", Precision.FP32)] = weights

    # Per-step cost across 3 components = 0.003. Budget 1.0 -> 333 steps,
    # rounded to 256 (largest power of 2 ≤ 333).
    assert model.predict_max_steps("sysX", conf, 1.0) == 256
    # Budget too tight for steps=2: 0.006 > 0.005.
    assert model.predict_max_steps("sysX", conf, 0.005) == 0


def test_predict_batch_matches_predict():
    """Vectorized predict_batch agrees with per-conf predict()."""
    rng = np.random.default_rng(3)
    samples = []
    for _ in range(40):
        batch = int(rng.choice([8, 16, 32]))
        length = int(rng.choice([32, 64, 128]))
        steps = int(rng.choice([64, 128, 256, 512]))
        conf = _conf(batch=batch, length=length, steps=steps)
        total = 0.5 + 0.001 * steps
        samples.append((conf, _fake_run("test", total)))

    model = TrainTimingModel()
    model.fit(samples)

    confs = [_conf(batch=b, length=l, steps=s)
             for b in (4, 8) for l in (16, 32) for s in (64, 256, 1024)]
    batch_pred = model.predict_batch("test", confs)
    assert batch_pred is not None
    for i, c in enumerate(confs):
        assert batch_pred[i] == pytest.approx(
            model.predict("test", c), rel=1e-9, abs=1e-9)


def test_emb_input_component_type_id():
    """Embedded inputs get one timing key per token domain: the table
    width doesn't change the lookup cost enough for per-width keys."""
    comps = featurize(_conf("bytes.emb.4|dense.4.tanh"))
    assert comps[0].type_id == "bytes.emb"
    assert comps[-1].type_id == "output"
    comps8 = featurize(_conf("bytes.emb.8|dense.8.tanh"))
    assert comps8[0].type_id == "bytes.emb"
    # The tied head's scoring matmul reuses the shared 'output' key
    # with the dense-shaped features (last width -> ntokens).
    comps_oh = featurize(_conf("bits.4.emb.8|rnn.8.tanh"))
    assert comps_oh[0].type_id == "bits.4.emb"


# -- Registry coverage --------------------------------------------------


def test_featurize_covers_every_layer_type():
    """Every layer type in the spec registry featurizes without an
    unknown-layer error, under its expected timing key. Guards against
    a new layer (or codec) silently missing from the dispatch."""
    specs = [
        "bytes|dense.8.gelu-norm-gru.8-mgru.8-mingru.8-lstm.8",
        "bytes|rnn.8.tanh-rmsnorm-slstm.8-mullstm.8-matlstm.8",
        "bits.1+bp|suffix.2-dense.8.gelu-conv.2-msr.8.1-rglru.2",
        "bits.2.oh+bp|latent.8.2-lrnn.8.2-lmgu.8.2",
        "bytes.emb.8|dense.8.tanh",
        "bits.1+bp|attn.8.2.16-split.add(dense.8.gelu, pass)"
        "-split.mul(dense.8.gelu, dense.8)-split.cat(dense.8.gelu, pass)",
    ]
    seen = set()
    for spec in specs:
        for c in featurize(_conf(spec, batch=4, length=16)):
            seen.add(c.type_id)
    expected = {
        # inputs
        "bytes", "bits.1+bp", "bits.2.oh+bp", "bytes.emb",
        # matmul family
        "dense.gelu", "dense.tanh", "dense", "rnn.tanh",
        "gru", "mgru", "mingru", "lstm", "slstm", "mullstm",
        # dedicated feature shapes
        "matlstm", "msr", "rglru", "attn", "suffix", "conv",
        "latent", "lrnn", "lmgu",
        # elementwise + markers + head
        "norm", "rmsnorm",
        "split.add", "split.cat", "split.mul",
        "output",
    }
    assert expected <= seen, f"missing: {sorted(expected - seen)}"


def test_reps_layers_scale_with_reps():
    """latent/lrnn/lmgu compute scales with `reps` (their inner
    recurrence fires reps times per token) -- the features must
    distinguish latent.8.2 from latent.8.8."""
    for name in ("latent", "lrnn", "lmgu"):
        comps2 = featurize(_conf(f"bits.1+bp|{name}.8.2", batch=4, length=16))
        comps8 = featurize(_conf(f"bits.1+bp|{name}.8.8", batch=4, length=16))
        f2 = next(c for c in comps2 if c.type_id == name).features
        f8 = next(c for c in comps8 if c.type_id == name).features
        # base15 (dense + OS^2 triple) + the same block reps-scaled.
        assert f2.shape == (30,)
        # One-shot base is reps-independent; the scaled copy grows 4x
        # (reps 2 -> 8); and the scaled half is reps * the base half.
        np.testing.assert_array_equal(f2[:15], f8[:15])
        np.testing.assert_array_equal(f8[15:], 4 * f2[15:])
        np.testing.assert_array_equal(f2[15:], 2 * f2[:15])


def test_dot_ignores_stale_weight_lengths():
    """Weights fitted before a feature-schema change have the wrong
    length for that type; they must contribute zero, not crash."""
    assert _dot({"latent": np.ones(12)}, "latent", [1.0] * 15) == 0.0
    assert _dot({"latent": np.ones(15)}, "latent", [1.0] * 15) == 15.0


# -- Output-head mult accounting ----------------------------------------


def _output_features(spec, batch=4, length=16):
    comps = featurize(_conf(spec, batch=batch, length=length))
    out = comps[-1]
    assert out.type_id == "output"
    return out.features


def test_output_head_features_unchanged_for_existing_families():
    """REGRESSION GUARD. The 'output' key is shared by every codec and
    its fitted weights depend on the features meaning one fixed thing.
    Any change here silently invalidates the persisted timing model,
    so these vectors are pinned to literals rather than recomputed.

    Shape: [1, L, B*L, IS, IS*L, IS*B*L, OS, OS*L, OS*B*L,
            M, M*L, M*B*L] with M the head's mult count -- which for
    every family below is exactly IS*OS.
    """
    b, l_ = 4, 16
    bl = b * l_

    # One-hot dense head: bytes, X = 8 -> 256 logits.
    is_, os_ = 8, 256
    m = is_ * os_  # 2048
    np.testing.assert_array_equal(
        _output_features("bytes|dense.8.gelu", b, l_),
        [1, l_, bl, is_, is_ * l_, is_ * bl, os_, os_ * l_, os_ * bl,
         m, m * l_, m * bl])

    # Sub-byte one-hot head: bits.4.oh+bp, X = 8 -> 16 logits.
    is_, os_ = 8, 16
    m = is_ * os_  # 128
    np.testing.assert_array_equal(
        _output_features("bits.4.oh+bp|dense.8.gelu", b, l_),
        [1, l_, bl, is_, is_ * l_, is_ * bl, os_, os_ * l_, os_ * bl,
         m, m * l_, m * bl])

    # Tied head: bytes.emb.8 -> the scoring matmul, X = 8, 256 rows.
    is_, os_ = 8, 256
    m = is_ * os_  # 2048
    np.testing.assert_array_equal(
        _output_features("bytes.emb.8|dense.8.tanh", b, l_),
        [1, l_, bl, is_, is_ * l_, is_ * bl, os_, os_ * l_, os_ * bl,
         m, m * l_, m * bl])

    # Tokenized one-hot head: 32 tokens, X = 4.
    is_, os_ = 4, 32
    m = is_ * os_  # 128
    np.testing.assert_array_equal(
        _output_features("tokens.32.hexbpe.oh|dense.4.tanh", b, l_),
        [1, l_, bl, is_, is_ * l_, is_ * bl, os_, os_ * l_, os_ * bl,
         m, m * l_, m * bl])


def test_pair_head_is_charged_its_real_mults_not_a_256_way_dense():
    """The hex-pair heads reach 256 logits through two 16-way heads, so
    the dense figure (256*X) over-charges them 5-6x. The base block
    still carries the true 256 logit width (softmax/reshape work)."""
    b, l_ = 4, 16
    bl = b * l_
    x = 32
    os_ = 256

    for spec, mults in (
        (f"bits.4.pair.add|dense.{x}.gelu", 32 * x + 288),
        (f"bits.4.pair.16|dense.{x}.gelu", (16 + 16) * x + 33 * 16 + 32),
        (f"bits.4.pair.4|dense.{x}.gelu", (16 + 4) * x + 33 * 4 + 32),
    ):
        feats = _output_features(spec, b, l_)
        np.testing.assert_array_equal(
            feats,
            [1, l_, bl, x, x * l_, x * bl, os_, os_ * l_, os_ * bl,
             mults, mults * l_, mults * bl])
        # Well below what the old ntokens-shaped assumption charged.
        assert mults < x * os_ / 4
        # ...and it is exactly the codec's own published figure.
        md = parse_model2(spec, precision=Precision.FP32)
        assert md.codec.num_mults == mults


def test_pair_head_mults_track_k_and_width():
    """k and X both move the head cost -- the old features saw neither
    (256*X regardless of k)."""
    def m(spec):
        return int(_output_features(spec)[9])

    assert m("bits.4.pair.4|dense.8.gelu") < m("bits.4.pair.64|dense.8.gelu")
    assert m("bits.4.pair.16|dense.8.gelu") < m("bits.4.pair.16|dense.32.gelu")
    # The additive arm sits just under .16 at equal width (32X+288 vs
    # 32X+560) -- the weight-comparable pairing the toggle edge uses.
    assert (m("bits.4.pair.16|dense.32.gelu")
            - m("bits.4.pair.add|dense.32.gelu")) == 272


def test_pair_conf_featurizes_and_predicts_end_to_end():
    """The whole pipeline runs for both arms: keys, feature widths,
    and a finite total-time prediction."""
    for spec in ("bits.4.pair.add|rnn.8.gelu", "bits.4.pair.16|rnn.8.gelu"):
        comps = featurize(_conf(spec, batch=4, length=16))
        # Both arms share ONE input key (identical 32-wide gather).
        assert comps[0].type_id == "bits.4.pair"
        assert comps[-1].type_id == "output"
        assert comps[-1].features.shape == (12,)
        per_key = {c.type_id: np.ones(c.features.shape) for c in comps}
        w = Weights(
            init=per_key, step=per_key,
            scan_full={"bits.4.pair": np.ones(3)},
            scan_short={"bits.4.pair": np.ones(4)},
        )
        t = predict_total_time(w, _conf(spec, batch=4, length=16))
        assert np.isfinite(t) and t > 0
