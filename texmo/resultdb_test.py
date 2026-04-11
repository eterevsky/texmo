import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.model import build_model_def
from texmo.precision import Precision
from texmo.predict import build_loss_trend
from texmo.resultdb import ResultDB
from texmo.run import Run


def _make_conf_run(
    spec="bytes|dense.64.gelu", batch=32, loss=3.123, system="test",
):
    model = build_model_def(spec, precision=Precision.FP32)
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
def db():
    return ResultDB()


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


def test_step_loss(db):
    conf1, run1 = _make_conf_run(loss=1)
    run1.add_step(1)
    run1.add_step(2)
    run1.add_step(3)
    db.add_run(conf1, run1)

    conf_runs = list(db.get_confs_runs())
    assert len(conf_runs) == 1
    np.testing.assert_array_equal(conf_runs[0][2].step_loss, [1, 2, 3])


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
        precision=list(Precision), decay=None,
    )


def test_clear_system_deletes_runs_and_conf_time(db):
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

    # conf_time for rpi is gone.
    cur = db._db.execute(
        "SELECT COUNT(*) FROM conf_time WHERE system = 'rpi'")
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
