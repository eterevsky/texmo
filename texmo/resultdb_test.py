import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.model_torch import build_model_def
from texmo.precision import Precision
from texmo.predict import build_loss_trend
from texmo.resultdb import ResultDB
from texmo.run import Run


def _make_conf_run(spec="bytes|dense.64.gelu", batch=32, loss=3.123):
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
        system="test",
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
