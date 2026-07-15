import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.db import DbReader, DbWriter
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.predict import build_loss_trend
from texmo.run import Run


class _RW:
    """Test helper: a `DbWriter` plus a separate `DbReader` over the
    same file-backed DB, exposed as a single object so the existing
    `db.add_run(...)` plus `db.get_systems()` style of test calls
    keeps working.

    Writer methods are checked first; only names exclusive to the
    read side (`get_*`, `iter_*`, `top_*`, ...) fall through to the
    reader. Tests poking the raw connection via `db._db` get the
    writer's connection.
    """

    def __init__(self, path: str):
        # Open writer first so it creates the schema; reader then opens
        # the same file in mode=ro.
        self.writer = DbWriter(path)
        self.reader = DbReader(path)

    def close(self):
        self.reader.close()
        self.writer.close()

    @property
    def _db(self):
        return self.writer._db

    def __getattr__(self, name):
        attr = getattr(self.writer, name, None)
        if attr is not None:
            return attr
        return getattr(self.reader, name)


def _make_conf_run(
    spec="bytes|dense.64.gelu", batch=32, loss=3.123, system="test",
):
    model = parse_model2(spec, precision=Precision.FP32)
    conf = Configuration(
        model=model,
        lr=0.25,
        length=128,
        batch=batch,
        steps=256,
        decay=1,
    )
    loss_trend = build_loss_trend(None, 1, [1, 2, 3])
    run = Run(
        system=system,
        step_loss=None,
        loss=loss,
        loss_trend=loss_trend,
        train_time=12.345,
    )
    return conf, run


@pytest.fixture
def db(tmp_path):
    bundle = _RW(str(tmp_path / "test.db"))
    yield bundle
    bundle.close()


def test_empty_db(db):
    cur = db._db.execute("SELECT * FROM conf")
    assert cur.fetchone() is None


def test_add_two_runs_same_conf(db):
    conf1, run1 = _make_conf_run()
    conf2, run2 = _make_conf_run(loss=3.321)
    db.add_run(conf1, run1)
    db.add_run(conf2, run2)

    cur = db._db.execute("SELECT COUNT(*) AS count FROM conf")
    assert cur.fetchone()["count"] == 1

    cur = db._db.execute("SELECT COUNT(*) AS count FROM run")
    assert cur.fetchone()["count"] == 2


def test_add_two_runs_different_conf(db):
    conf1, run1 = _make_conf_run()
    db.add_run(conf1, run1)
    conf2, run2 = _make_conf_run(spec="bytes|rnn.64.tanh", loss=3.321)
    db.add_run(conf2, run2)

    cur = db._db.execute("SELECT COUNT(*) AS count FROM conf")
    assert cur.fetchone()["count"] == 2

    cur = db._db.execute("SELECT COUNT(*) AS count FROM run")
    assert cur.fetchone()["count"] == 2


def test_get_confs_runs(db):
    conf1, run1 = _make_conf_run(loss=1)
    db.add_run(conf1, run1)
    conf2, run2 = _make_conf_run(loss=2)
    db.add_run(conf2, run2)
    conf3, run3 = _make_conf_run(spec="bytes|rnn.64.tanh", loss=3)
    db.add_run(conf3, run3)
    conf4, run4 = _make_conf_run(spec="bytes|rnn.64.tanh", loss=4)
    db.add_run(conf4, run4)

    conf_runs = set((conf, run.loss) for _, conf, run in db.get_confs_runs())

    assert conf_runs == {
        (conf1, 1),
        (conf2, 2),
        (conf3, 3),
        (conf4, 4),
    }


def test_step_loss_not_stored(db):
    """The per-step training-loss history is deliberately NOT
    persisted (it dominated the DB size and went unused); only the
    final loss and the fitted trend params survive the round trip."""
    conf1, run1 = _make_conf_run(loss=1)
    run1.add_step(1)
    run1.add_step(2)
    run1.add_step(3)
    db.add_run(conf1, run1)

    conf_runs = list(db.get_confs_runs())
    assert len(conf_runs) == 1
    stored = conf_runs[0][2]
    assert stored.loss == 1
    assert stored.step_loss is None or len(stored.step_loss) == 0


def test_total_runs(db):
    assert db.total_runs() == 0

    conf1, run1 = _make_conf_run(loss=1)
    db.add_run(conf1, run1)
    assert db.total_runs() == 1

    conf2, run2 = _make_conf_run(loss=2)
    db.add_run(conf2, run2)
    assert db.total_runs() == 2


def test_find_or_add_conf(db):
    conf, _ = _make_conf_run()
    id1 = db.find_or_add_conf(conf)
    id2 = db.find_or_add_conf(conf)
    assert id1 == id2


def test_conf_weights_stored(db):
    conf, run = _make_conf_run()
    db.add_run(conf, run)

    cur = db._db.execute("SELECT weights FROM conf")
    row = cur.fetchone()
    assert row["weights"] == conf.num_weights


def test_different_batch_different_conf(db):
    conf1, run1 = _make_conf_run(batch=32, loss=1)
    conf2, run2 = _make_conf_run(batch=64, loss=2)
    db.add_run(conf1, run1)
    db.add_run(conf2, run2)

    cur = db._db.execute("SELECT COUNT(*) AS count FROM conf")
    assert cur.fetchone()["count"] == 2


def test_median_score_updated(db):
    conf, run1 = _make_conf_run(loss=2.0)
    conf, run2 = _make_conf_run(loss=4.0)
    db.add_run(conf, run1)
    db.add_run(conf, run2)

    cur = db._db.execute("SELECT median_score FROM conf")
    row = cur.fetchone()
    assert row["median_score"] == pytest.approx(3.0)


def test_get_conf_id(db):
    conf, run = _make_conf_run()
    assert db.get_conf_id(conf) is None

    db.add_run(conf, run)
    conf_id = db.get_conf_id(conf)
    assert conf_id is not None
    assert db.get_conf_id(conf) == conf_id  # cached


def test_get_conf_id_cache(db):
    conf1, run1 = _make_conf_run(loss=1)
    conf2, run2 = _make_conf_run(spec="bytes|rnn.64.tanh", loss=2)
    db.add_run(conf1, run1)
    db.add_run(conf2, run2)

    id1 = db.get_conf_id(conf1)
    id2 = db.get_conf_id(conf2)
    assert id1 != id2


def test_get_run_counts(db):
    conf1, run1 = _make_conf_run(loss=1)
    conf2, run2 = _make_conf_run(loss=2)
    conf3, run3 = _make_conf_run(spec="bytes|rnn.64.tanh", loss=3)
    db.add_run(conf1, run1)
    db.add_run(conf2, run2)
    db.add_run(conf3, run3)

    id1 = db.get_conf_id(conf1)
    id3 = db.get_conf_id(conf3)
    counts = db.get_run_counts([id1, id3])
    # Without a system arg: system_runs is 0 for both.
    assert counts[id1] == (2, 0)
    assert counts[id3] == (1, 0)


def test_get_run_counts_with_system(db):
    conf1, _ = _make_conf_run()
    conf2, _ = _make_conf_run(spec="bytes|rnn.64.tanh")

    _, r1_rpi1 = _make_conf_run(loss=1.0, system="rpi")
    _, r1_rpi2 = _make_conf_run(loss=1.2, system="rpi")
    _, r1_wb = _make_conf_run(loss=1.4, system="whitebox")
    _, r2_wb = _make_conf_run(spec="bytes|rnn.64.tanh", loss=2.0, system="whitebox")

    db.add_run(conf1, r1_rpi1)
    db.add_run(conf1, r1_rpi2)
    db.add_run(conf1, r1_wb)
    db.add_run(conf2, r2_wb)

    id1 = db.get_conf_id(conf1)
    id2 = db.get_conf_id(conf2)

    counts = db.get_run_counts([id1, id2], system="rpi")
    assert counts[id1] == (3, 2)   # 3 total, 2 on rpi
    assert counts[id2] == (1, 0)   # 1 total, 0 on rpi

    counts = db.get_run_counts([id1, id2], system="whitebox")
    assert counts[id1] == (3, 1)
    assert counts[id2] == (1, 1)


def test_get_systems_empty(db):
    assert db.get_systems() == []


def test_get_systems(db):
    conf, run1 = _make_conf_run(loss=1, system="whitebox")
    conf, run2 = _make_conf_run(loss=2, system="rpi")
    conf, run3 = _make_conf_run(loss=3, system="whitebox")
    db.add_run(conf, run1)
    db.add_run(conf, run2)
    db.add_run(conf, run3)
    assert db.get_systems() == ["rpi", "whitebox"]


def _make_template():
    from texmo.common import INF
    from texmo.configuration import Template
    return Template(
        spec=None, lr=None, length=None, batch=None,
        steps=None, max_weights=(2, INF),
        precision=list(Precision),
    )


def test_clear_system_deletes_runs_and_estimates(db):
    conf, _ = _make_conf_run()
    _, r1 = _make_conf_run(loss=1.0, system="rpi")
    _, r2 = _make_conf_run(loss=2.0, system="rpi")
    _, r3 = _make_conf_run(loss=3.0, system="whitebox")
    db.add_run(conf, r1)
    db.add_run(conf, r2)
    db.add_run(conf, r3)

    # Both systems are present.
    assert set(db.get_systems()) == {"rpi", "whitebox"}

    deleted = db.clear_system("rpi")
    assert deleted == 2

    # Only whitebox remains.
    assert db.get_systems() == ["whitebox"]

    # run table has only whitebox runs.
    cur = db._db.execute("SELECT COUNT(*) FROM run WHERE system = 'rpi'")
    assert cur.fetchone()[0] == 0
    cur = db._db.execute("SELECT COUNT(*) FROM run WHERE system = 'whitebox'")
    assert cur.fetchone()[0] == 1

    # Time estimates for rpi are gone.
    cur = db._db.execute(
        "SELECT COUNT(*) FROM conf_time_estimate WHERE system = 'rpi'")
    assert cur.fetchone()[0] == 0


def test_clear_system_recomputes_median_score(db):
    conf, _ = _make_conf_run()
    _, r1 = _make_conf_run(loss=1.0, system="rpi")
    _, r2 = _make_conf_run(loss=3.0, system="rpi")
    _, r3 = _make_conf_run(loss=5.0, system="whitebox")
    db.add_run(conf, r1)
    db.add_run(conf, r2)
    db.add_run(conf, r3)

    # Before clearing: median over all runs = median(1, 3, 5) = 3.
    cur = db._db.execute("SELECT median_score FROM conf")
    assert cur.fetchone()[0] == pytest.approx(3.0)

    db.clear_system("rpi")

    # After clearing: median over remaining runs = median(5) = 5.
    cur = db._db.execute("SELECT median_score FROM conf")
    assert cur.fetchone()[0] == pytest.approx(5.0)


def test_clear_system_unknown_system(db):
    conf, run = _make_conf_run()
    db.add_run(conf, run)
    # Clearing an unknown system should be a no-op.
    deleted = db.clear_system("nonexistent")
    assert deleted == 0
    assert db.total_runs() == 1


def test_add_run_writes_median_estimate(db):
    conf, run = _make_conf_run(system="rpi")
    run.train_time = 5.5
    db.add_run(conf, run)
    conf_id = db.get_conf_id(conf)
    est = db.get_time_estimate(conf_id, "rpi")
    assert est is not None
    time_s, source = est
    assert time_s == pytest.approx(5.5)
    assert source == "median"


def test_predicted_estimate_replaced_by_median_on_add_run(db):
    conf, _ = _make_conf_run()
    conf_id = db.find_or_add_conf(conf)
    # Insert a prediction first.
    db.upsert_time_estimates([(conf_id, "rpi", 2.0, "predicted")])
    assert db.get_time_estimate(conf_id, "rpi") == (2.0, "predicted")

    # Adding a run upgrades the estimate to median.
    _, run = _make_conf_run(system="rpi")
    run.train_time = 7.0
    db.add_run(conf, run, conf_id=conf_id)
    time_s, source = db.get_time_estimate(conf_id, "rpi")
    assert time_s == pytest.approx(7.0)
    assert source == "median"


def test_upsert_predicted_does_not_clobber_median(db):
    """Once a 'median' row exists, a later 'predicted' bulk write must
    leave it alone -- the timing thread's stale-snapshot upsert would
    otherwise erase the truth a concurrent _add_run just wrote."""
    conf, run = _make_conf_run(system="rpi")
    run.train_time = 7.0
    db.add_run(conf, run)
    conf_id = db.get_conf_id(conf)
    assert db.get_time_estimate(conf_id, "rpi") == (pytest.approx(7.0), "median")

    db.upsert_predicted_time_estimates([(conf_id, "rpi", 2.0)])
    # Median row preserved, prediction discarded.
    assert db.get_time_estimate(conf_id, "rpi") == (pytest.approx(7.0), "median")


def test_upsert_predicted_overwrites_predicted(db):
    """A second 'predicted' write should refresh the value."""
    conf, _ = _make_conf_run()
    conf_id = db.find_or_add_conf(conf)
    db.upsert_predicted_time_estimates([(conf_id, "rpi", 2.0)])
    db.upsert_predicted_time_estimates([(conf_id, "rpi", 3.5)])
    assert db.get_time_estimate(conf_id, "rpi") == (pytest.approx(3.5), "predicted")


def test_upsert_predicted_inserts_when_missing(db):
    conf, _ = _make_conf_run()
    conf_id = db.find_or_add_conf(conf)
    db.upsert_predicted_time_estimates([(conf_id, "rpi", 4.2)])
    assert db.get_time_estimate(conf_id, "rpi") == (pytest.approx(4.2), "predicted")


def test_iter_confs_by_precision(db):
    conf_fp32, run_fp32 = _make_conf_run(system="rpi")
    model_fp16 = parse_model2("bytes|dense.64.gelu", precision=Precision.FP16)
    conf_fp16 = Configuration(
        model=model_fp16, lr=0.25, length=128, batch=32, steps=256, decay=1)
    db.add_run(conf_fp32, run_fp32)
    db.find_or_add_conf(conf_fp16)

    fp32_confs = list(db.iter_confs_by_precision(Precision.FP32))
    fp16_confs = list(db.iter_confs_by_precision(Precision.FP16))
    assert len(fp32_confs) == 1
    assert len(fp16_confs) == 1
    assert fp32_confs[0][1] == conf_fp32
    assert fp16_confs[0][1] == conf_fp16


def test_diverged_run_stored_and_read_as_none(db):
    """train_time=None on the way in (diverged training) becomes the
    schema's 0 sentinel in storage, then translates back to None on
    every read so consumers can rely on `is None` checks."""
    conf, run = _make_conf_run(system="rpi")
    run.train_time = None
    run.loss = float('inf')
    db.add_run(conf, run)

    # No 'median' row should exist (no usable timing signal).
    conf_id = db.get_conf_id(conf)
    assert db.get_time_estimate(conf_id, "rpi") is None

    # get_runs_for_timing translates the 0 sentinel back to None.
    runs = list(db.get_runs_for_timing("rpi", Precision.FP32))
    assert len(runs) == 1
    assert runs[0][1].train_time is None

    # get_confs_runs does too.
    all_runs = list(db.get_confs_runs())
    assert len(all_runs) == 1
    assert all_runs[0][2].train_time is None


def test_zero_train_time_rejected(db):
    """0 is reserved as the 'no usable time' sentinel in storage; a
    caller passing literal 0 should be caught by the assert rather
    than silently producing an instantaneous-run row."""
    conf, run = _make_conf_run(system="rpi")
    run.train_time = 0
    with pytest.raises(AssertionError):
        db.add_run(conf, run)


def test_get_runs_for_timing_filters_by_system_and_precision(db):
    conf, run_rpi = _make_conf_run(system="rpi")
    run_rpi.train_time = 3.0
    _, run_whitebox = _make_conf_run(system="whitebox")
    run_whitebox.train_time = 4.0
    db.add_run(conf, run_rpi)
    db.add_run(conf, run_whitebox)

    rpi = list(db.get_runs_for_timing("rpi", Precision.FP32))
    assert len(rpi) == 1
    assert rpi[0][0] == conf
    assert rpi[0][1].system == "rpi"
    assert rpi[0][1].train_time == pytest.approx(3.0)

    fp16 = list(db.get_runs_for_timing("rpi", Precision.FP16))
    assert fp16 == []




def _add_runs(db, conf, losses, system, train_time):
    """Helper: add multiple runs of the same conf with given losses."""
    import datetime
    base = datetime.datetime(2026, 4, 11, 12, 0, 0)
    for i, loss in enumerate(losses):
        _, run = _make_conf_run(
            spec=str(conf.model), loss=loss, system=system)
        run.train_time = train_time
        db.add_run(conf, run, timestamp=base + datetime.timedelta(seconds=i))


def test_fastest_near_best_segments_empty(db):
    template = _make_template()
    segments = db.fastest_near_best_segments(
        template=template, system="whitebox", pareto=[])
    assert segments == []


def test_fastest_near_best_segments(db):
    """Test building the scaling curve segments.

    Setup (all on whitebox, three dense layers of increasing size):
    - dense.8.gelu:  W=4360,  score 5.00,  time 8s
    - dense.16.gelu: W=8464,  score 4.98,  time 2s
    - dense.32.gelu: W=16672, score 4.00,  time 10s

    All three end up on the Pareto frontier because each has a strictly
    better score than the previous one (5.00 > 4.98 > 4.00).

    Expected segments: for each Pareto interval, the algorithm walks the
    confs sorted by time ASC and assigns them to sub-ranges. dense.16 is
    faster than dense.8 (2s vs 8s) and fits the threshold 5.00*1.01, so
    it should cover [8464, 16672). For the last interval [16672, inf),
    the threshold 4.04 excludes dense.16 (score 4.98), so dense.32
    covers itself.
    """
    template = _make_template()

    conf_a, _ = _make_conf_run(spec="bytes|dense.8.gelu")
    conf_b, _ = _make_conf_run(spec="bytes|dense.16.gelu")
    conf_c, _ = _make_conf_run(spec="bytes|dense.32.gelu")

    wa = conf_a.model.num_weights
    wb = conf_b.model.num_weights
    wc = conf_c.model.num_weights
    assert wa < wb < wc

    _add_runs(db, conf_a, [5.00, 5.00], "whitebox", 8.0)
    _add_runs(db, conf_b, [4.98, 4.98], "whitebox", 2.0)
    _add_runs(db, conf_c, [4.00, 4.00], "whitebox", 10.0)

    pareto = list(db.top_confs_global(template, system="whitebox"))
    pareto_specs = [str(cs.conf.model) for cs in pareto]
    assert pareto_specs == [
        "bytes|dense.8.gelu",
        "bytes|dense.16.gelu",
        "bytes|dense.32.gelu",
    ]

    segments = db.fastest_near_best_segments(
        template=template, system="whitebox", pareto=pareto)

    # Map by spec for easier assertions.
    by_spec = {
        str(cs.conf.model): (lo, hi, cs.median_time)
        for lo, hi, cs in segments
    }

    # dense.8 covers [wa, wb): it's the fastest conf reaching within 1%
    # of its own score (5.05) within [wa, wb).
    assert "bytes|dense.8.gelu" in by_spec
    lo, hi, t = by_spec["bytes|dense.8.gelu"]
    assert lo == wa
    assert hi == wb
    assert t == pytest.approx(8.0)

    # dense.16 covers [wb, wc): threshold from the left Pareto point
    # (dense.16 itself, 4.98*1.01=5.0298). dense.16 is the fastest conf
    # qualifying at <= wc weights.
    assert "bytes|dense.16.gelu" in by_spec
    lo, hi, t = by_spec["bytes|dense.16.gelu"]
    assert lo == wb
    assert hi == wc

    # dense.32 covers [wc, inf): threshold 4.04, only dense.32 qualifies.
    assert "bytes|dense.32.gelu" in by_spec
    lo, hi, t = by_spec["bytes|dense.32.gelu"]
    assert lo == wc
    assert hi is None  # unbounded right
    assert t == pytest.approx(10.0)


def test_fastest_near_best_segments_faster_alternative(db):
    """Verify that a non-Pareto faster conf is picked when available.

    Setup:
    - dense.8.gelu:  score 5.00, time 20s (Pareto)
    - dense.16.gelu: score 5.04, time 2s  (NOT Pareto — worse score)
    - dense.32.gelu: score 4.00, time 10s (Pareto)

    Since dense.16's score (5.04) is *worse* than dense.8's (5.00),
    it's not on the Pareto frontier. But it's within 1% of dense.8's
    score (threshold 5.05) and is much faster, so in the [wa, wc)
    interval, dense.16 should replace dense.8 for weights >= wb.
    """
    template = _make_template()

    conf_a, _ = _make_conf_run(spec="bytes|dense.8.gelu")
    conf_b, _ = _make_conf_run(spec="bytes|dense.16.gelu")
    conf_c, _ = _make_conf_run(spec="bytes|dense.32.gelu")
    wa, wb, wc = (
        conf_a.model.num_weights,
        conf_b.model.num_weights,
        conf_c.model.num_weights,
    )

    _add_runs(db, conf_a, [5.00, 5.00], "whitebox", 20.0)
    _add_runs(db, conf_b, [5.04, 5.04], "whitebox", 2.0)
    _add_runs(db, conf_c, [4.00, 4.00], "whitebox", 10.0)

    pareto = list(db.top_confs_global(template, system="whitebox"))
    # dense.16 (score 5.04) is worse than dense.8 (5.00), so it should NOT
    # be on the Pareto frontier.
    pareto_specs = [str(cs.conf.model) for cs in pareto]
    assert pareto_specs == [
        "bytes|dense.8.gelu",
        "bytes|dense.32.gelu",
    ]

    segments = db.fastest_near_best_segments(
        template=template, system="whitebox", pareto=pareto)

    by_spec = {
        str(cs.conf.model): (lo, hi, cs.median_time)
        for lo, hi, cs in segments
    }

    # dense.16 should appear as an alternative in the [wa, wc) interval,
    # covering [wb, wc).
    assert "bytes|dense.16.gelu" in by_spec
    lo, hi, t = by_spec["bytes|dense.16.gelu"]
    assert lo == wb
    assert hi == wc
    assert t == pytest.approx(2.0)

    # dense.8 still covers [wa, wb) (anchor of the interval).
    assert "bytes|dense.8.gelu" in by_spec
    lo, hi, _ = by_spec["bytes|dense.8.gelu"]
    assert lo == wa
    assert hi == wb

    # dense.32 covers [wc, inf).
    assert "bytes|dense.32.gelu" in by_spec


def test_top_confs_global_system_filter(db):
    # top_confs_global yields a Pareto frontier: only configs whose score
    # improves as weight count increases. So we need the larger config to
    # have the better (lower) score.
    #
    # conf_small: dense.32.gelu, runs on rpi only, worse score
    # conf_big: dense.64.gelu, runs on both rpi AND whitebox, better score
    conf_small, _ = _make_conf_run(spec="bytes|dense.32.gelu")
    conf_big, _ = _make_conf_run(spec="bytes|dense.64.gelu")

    _, rs_rpi1 = _make_conf_run(spec="bytes|dense.32.gelu", loss=3.0, system="rpi")
    _, rs_rpi2 = _make_conf_run(spec="bytes|dense.32.gelu", loss=3.2, system="rpi")
    _, rb_rpi1 = _make_conf_run(spec="bytes|dense.64.gelu", loss=1.0, system="rpi")
    _, rb_rpi2 = _make_conf_run(spec="bytes|dense.64.gelu", loss=1.2, system="rpi")
    _, rb_wb1 = _make_conf_run(spec="bytes|dense.64.gelu", loss=1.4, system="whitebox")
    _, rb_wb2 = _make_conf_run(spec="bytes|dense.64.gelu", loss=1.6, system="whitebox")

    db.add_run(conf_small, rs_rpi1)
    db.add_run(conf_small, rs_rpi2)
    db.add_run(conf_big, rb_rpi1)
    db.add_run(conf_big, rb_rpi2)
    db.add_run(conf_big, rb_wb1)
    db.add_run(conf_big, rb_wb2)

    template = _make_template()

    # Without filter: both confs on the frontier.
    all_specs = {
        str(cs.conf.model) for cs in db.top_confs_global(template)
    }
    assert all_specs == {"bytes|dense.32.gelu", "bytes|dense.64.gelu"}

    # Filter by "whitebox": only conf_big has runs there.
    wb_specs = {
        str(cs.conf.model)
        for cs in db.top_confs_global(template, system="whitebox")
    }
    assert wb_specs == {"bytes|dense.64.gelu"}

    # Filter by "rpi": both have runs there.
    rpi_specs = {
        str(cs.conf.model)
        for cs in db.top_confs_global(template, system="rpi")
    }
    assert rpi_specs == {"bytes|dense.32.gelu", "bytes|dense.64.gelu"}


def _conf_with_steps(steps, batch=32):
    model = parse_model2("bytes|dense.64.gelu", precision=Precision.FP32)
    return Configuration(
        model=model, lr=0.25, length=128, batch=batch, steps=steps, decay=1,
    )


def test_has_covering_run_exact_match(db):
    conf = _conf_with_steps(256)
    db.add_run(conf, Run(system="rpi", step_loss=[0.1], loss=1.0, train_time=2.0))
    assert db.has_covering_run(conf, system="rpi") is True


def test_has_covering_run_higher_steps_covers(db):
    """A run at 1024 steps covers a 256-steps query — the higher-steps
    run is strictly more informative."""
    higher = _conf_with_steps(1024)
    db.add_run(higher, Run(system="rpi", step_loss=[0.1], loss=1.0, train_time=2.0))
    query = _conf_with_steps(256)
    assert db.has_covering_run(query, system="rpi") is True


def test_has_covering_run_lower_steps_does_not_cover(db):
    lower = _conf_with_steps(128)
    db.add_run(lower, Run(system="rpi", step_loss=[0.1], loss=1.0, train_time=2.0))
    query = _conf_with_steps(256)
    assert db.has_covering_run(query, system="rpi") is False


def test_has_covering_run_wrong_system(db):
    conf = _conf_with_steps(256)
    db.add_run(conf, Run(system="rpi", step_loss=[0.1], loss=1.0, train_time=2.0))
    assert db.has_covering_run(conf, system="whitebox") is False


def test_has_covering_run_different_batch_does_not_cover(db):
    """A run at the same steps but different batch is a different
    configuration in the timing sense — should not count as covered."""
    db.add_run(
        _conf_with_steps(256, batch=64),
        Run(system="rpi", step_loss=[0.1], loss=1.0, train_time=2.0),
    )
    query = _conf_with_steps(256, batch=32)
    assert db.has_covering_run(query, system="rpi") is False


def test_fastest_near_best_segments_any_system_empty(db):
    template = _make_template()
    assert db.fastest_near_best_segments_any_system(
        template=template, pareto=[]) == []


def test_fastest_near_best_segments_any_system_picks_fastest_system(db):
    """A single conf with runs on two systems — the returned segment's
    `system` field is whichever system had the lower median_time."""
    template = _make_template()
    conf, _ = _make_conf_run(spec="bytes|dense.8.gelu")
    # Same conf on two systems: rpi slow, whitebox fast.
    _add_runs(db, conf, [5.00, 5.00], "rpi", 10.0)
    _add_runs(db, conf, [5.00, 5.00], "whitebox", 2.0)

    pareto = list(db.top_confs_global(template))
    segments = db.fastest_near_best_segments_any_system(
        template=template, pareto=pareto)

    assert len(segments) == 1
    lo, hi, cs = segments[0]
    assert str(cs.conf.model) == "bytes|dense.8.gelu"
    assert cs.system == "whitebox"
    assert cs.median_time == pytest.approx(2.0)


def test_fastest_near_best_segments_any_system_cross_system(db):
    """Two confs with different best-system winners; both segments
    should reflect their respective winning systems."""
    template = _make_template()
    conf_a, _ = _make_conf_run(spec="bytes|dense.8.gelu")
    conf_b, _ = _make_conf_run(spec="bytes|dense.32.gelu")
    wa = conf_a.model.num_weights
    wb = conf_b.model.num_weights
    assert wa < wb

    # dense.8 fastest on rpi; dense.32 fastest on whitebox.
    _add_runs(db, conf_a, [5.00, 5.00], "rpi", 3.0)
    _add_runs(db, conf_a, [5.00, 5.00], "whitebox", 8.0)
    _add_runs(db, conf_b, [4.00, 4.00], "rpi", 30.0)
    _add_runs(db, conf_b, [4.00, 4.00], "whitebox", 5.0)

    pareto = list(db.top_confs_global(template))
    segments = db.fastest_near_best_segments_any_system(
        template=template, pareto=pareto)

    by_spec = {
        str(cs.conf.model): (lo, hi, cs.system, cs.median_time)
        for lo, hi, cs in segments
    }
    assert by_spec["bytes|dense.8.gelu"][2] == "rpi"
    assert by_spec["bytes|dense.8.gelu"][3] == pytest.approx(3.0)
    assert by_spec["bytes|dense.32.gelu"][2] == "whitebox"
    assert by_spec["bytes|dense.32.gelu"][3] == pytest.approx(5.0)


# --- pick_me / invalid-conf filter -----------------------------------------


def _pick_me_conf(spec="bytes|dense.32.gelu"):
    model = parse_model2(spec, precision=Precision.FP32)
    return Configuration(
        model=model, lr=0.1, length=128, batch=32, steps=256, decay=1.0,
    )


def test_add_pick_me_conf_inserts_with_flag_set(db):
    conf = _pick_me_conf()
    cid, inserted = db.writer.add_pick_me_conf(conf)
    assert inserted is True
    flag = db._db.execute(
        'SELECT pick_me FROM conf WHERE id = ?', (cid,)).fetchone()[0]
    assert flag == 1


def test_add_pick_me_conf_does_not_reflag_existing(db):
    """If the conf already exists with pick_me=0, the call must not
    silently flip its flag — that conf already has measurement
    history and we shouldn't re-prioritize it."""
    conf = _pick_me_conf()
    # Insert via normal find_or_add (pick_me defaults to 0).
    first_id = db.writer.find_or_add_conf(conf)
    cid, inserted = db.writer.add_pick_me_conf(conf)
    assert inserted is False
    assert cid == first_id
    flag = db._db.execute(
        'SELECT pick_me FROM conf WHERE id = ?', (cid,)).fetchone()[0]
    assert flag == 0


def test_pick_me_conf_returns_flagged_untrained(db):
    """A pick_me=1 conf with no runs is returned."""
    conf = _pick_me_conf()
    db.writer.add_pick_me_conf(conf)
    template = _make_template()
    got = db.pick_me_conf(template)
    assert got == conf


def test_pick_me_conf_returns_none_after_min_runs(db):
    """Once a pick_me conf has >= min_runs, it's no longer returned."""
    conf = _pick_me_conf()
    db.writer.add_pick_me_conf(conf)
    # Add two runs (default min_runs = 2).
    db.add_run(conf, Run(
        system="rpi", step_loss=None, loss=1.0, train_time=1.0))
    db.add_run(conf, Run(
        system="rpi", step_loss=None, loss=2.0, train_time=1.0))
    template = _make_template()
    assert db.pick_me_conf(template) is None


def test_pick_me_conf_skips_invalid(db):
    """A pick_me conf that no longer passes is_valid (norm as the first
    layer here) is skipped; a valid one is returned instead. Guards the
    Model2 transition, where the validity rules change."""
    invalid = _pick_me_conf("bytes|norm-dense.32.gelu")
    assert not invalid.model.is_valid()
    valid = _pick_me_conf("bytes|dense.32.gelu")
    db.writer.add_pick_me_conf(invalid)
    db.writer.add_pick_me_conf(valid)
    assert db.pick_me_conf(_make_template()) == valid


def test_pick_me_conf_none_when_all_invalid(db):
    invalid = _pick_me_conf("bytes|norm-dense.32.gelu")
    db.writer.add_pick_me_conf(invalid)
    assert db.pick_me_conf(_make_template()) is None


def test_pick_me_conf_ignores_unflagged(db):
    """Confs without pick_me=1 are never returned."""
    conf, run = _make_conf_run()
    db.add_run(conf, run)  # adds the conf with pick_me=0
    template = _make_template()
    assert db.pick_me_conf(template) is None


def test_top_confs_global_filters_invalid_by_default(db):
    """A conf where norm is the first stack layer is invalid; it must
    not appear in the Pareto by default, but does with
    `include_invalid=True`."""
    invalid_conf, _ = _make_conf_run(spec="bits.1+bp|norm-rnn.2.tanh")
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.5, train_time=1.0))
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.6, train_time=1.0))

    template = _make_template()
    assert list(db.top_confs_global(template)) == []

    with_invalid = list(
        db.top_confs_global(template, include_invalid=True))
    assert len(with_invalid) == 1
    assert str(with_invalid[0].conf.model) == "bits.1+bp|norm-rnn.2.tanh"


def test_confs_under_time_skips_invalid(db):
    """`time_budget` calls this; invalid confs must not surface."""
    invalid_conf, _ = _make_conf_run(spec="bits.1+bp|norm-rnn.2.tanh")
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.5, train_time=1.0))
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.6, train_time=1.0))

    from texmo.common import INF
    results = list(db.confs_under_time(
        template=_make_template(), system="rpi",
        max_weights=INF, max_time=1e9,
    ))
    assert results == []


def test_top_confs_for_system_skips_invalid(db):
    """`_select_top_neighbor`, `_select_max_weights`,
    `_select_predicted_best_impl` all call this. Invalid confs would
    otherwise become 'top' seeds."""
    invalid_conf, _ = _make_conf_run(spec="bits.1+bp|norm-rnn.2.tanh")
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.5, train_time=1.0))
    db.add_run(invalid_conf, Run(
        system="rpi", step_loss=None, loss=0.6, train_time=1.0))

    results = list(db.top_confs_for_system(
        system="rpi", template=_make_template(),
    ))
    assert results == []


def test_top_confs_global_invalid_does_not_block_heavier_valid(db):
    """When the best conf at weight bucket W1 is invalid, a valid
    conf at heavier weight W2 with worse-but-still-better-than-best-so-
    far score must still surface — the invalid one shouldn't bump
    `best_score` for the monotonic-improvement filter."""
    # bits.1+bp|norm-rnn.2.tanh is small (a few dozen weights);
    # bytes|dense.32.gelu is heavier. Give the invalid one a *worse*
    # score so the valid one is naturally Pareto-better than nothing
    # (best_score starts at INF). Without the "don't bump best_score
    # on skipped invalid" guard, the invalid skip would still raise
    # best_score and hide the valid heavier conf.
    invalid_conf, _ = _make_conf_run(spec="bits.1+bp|norm-rnn.2.tanh")
    valid_conf, _ = _make_conf_run(spec="bytes|dense.32.gelu")

    # Same score so the only thing that distinguishes them is weights;
    # the invalid one comes first by weight ascending. If the filter
    # bumped best_score on skip, the valid one (same score) would be
    # blocked by the `< best_score` check.
    for c, loss in [(invalid_conf, 5.0), (invalid_conf, 5.1),
                    (valid_conf, 5.0), (valid_conf, 5.1)]:
        db.add_run(c, Run(
            system="rpi", step_loss=None, loss=loss, train_time=1.0))

    specs = {
        str(cs.conf.model)
        for cs in db.top_confs_global(_make_template())
    }
    assert specs == {"bytes|dense.32.gelu"}
