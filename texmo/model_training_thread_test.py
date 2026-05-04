"""Integration tests for ModelTrainingThread.

Uses a file-backed ResultDB because the thread opens its own connection
and an in-memory DB would give it a fresh, empty database. The thread
is told to stop after the work messages are enqueued; ordering of a
single-consumer Queue guarantees the work messages are processed before
"stop".
"""

import datetime
from queue import Queue

import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.model import build_model_def
from texmo.model_training_thread import (
    ModelTrainingThread,
    bootstrap,
)
from texmo.precision import Precision
from texmo.predict.timing import TrainTimingModel
from texmo.resultdb import ResultDB
from texmo.run import Run


def _conf(
    spec="bytes|dense.8.gelu",
    batch=8,
    length=32,
    steps=100,
    precision=Precision.FP32,
):
    model = build_model_def(spec, precision=precision)
    return Configuration(
        model=model, lr=0.01, length=length, batch=batch, steps=steps,
        decay=1,
    )


def _seed_runs(db: ResultDB, system: str, n: int, rng: np.random.Generator):
    """Insert `n` runs on `system` with varied shape so fit succeeds.

    Vary `steps` as well: the new timing model decomposes total time
    into init + scan + per-step, and needs samples at multiple step
    counts to disambiguate the components.
    """
    base = datetime.datetime(2026, 4, 11, 12, 0, 0)
    for i in range(n):
        batch = int(rng.choice([8, 16, 32]))
        length = int(rng.choice([32, 64, 128]))
        steps = int(rng.choice([64, 128, 256, 512]))
        conf = _conf(batch=batch, length=length, steps=steps)
        per_step = 0.01 + 1e-8 * batch * length
        run = Run(
            system=system,
            step_loss=[0.1],
            loss=3.0,
            train_time=per_step * conf.steps,
        )
        db.add_run(conf, run, timestamp=base + datetime.timedelta(seconds=i))


def _run_thread(db_path: str, messages: list):
    """Run ModelTrainingThread with messages + stop; block until done."""
    q = Queue()
    thread = ModelTrainingThread(db_path, q)
    thread.start()
    for m in messages:
        q.put(m)
    q.put(("stop", None))
    thread.join(timeout=60)
    assert not thread.is_alive(), "model-training thread did not stop"


@pytest.fixture(autouse=True)
def _lower_min_samples(monkeypatch):
    # Fit is expensive with MIN_SAMPLES=20; lower for fast tests.
    monkeypatch.setattr(TrainTimingModel, "MIN_SAMPLES", 5)


def test_refit_writes_predicted_estimates(tmp_path):
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    with ResultDB(db_path) as db:
        _seed_runs(db, "rpi", 10, rng)
        unrun_conf = _conf(batch=4, length=64)
        unrun_id = db.find_or_add_conf(unrun_conf)

    _run_thread(db_path, [("refit", ("rpi", Precision.FP32))])

    with ResultDB(db_path, readonly=True) as db:
        est = db.get_time_estimate(unrun_id, "rpi")
        assert est is not None
        time_s, source = est
        assert source == "predicted"
        # Predicted estimate stores total train_time (≈ (steps-1) *
        # per_step), not per-step. For our seeded runs (steps=100,
        # per_step ≈ 0.01) that's ≈ 1 s; a per-step value would be ≈
        # 0.01 s. Guard against the units regression.
        assert time_s > 0.1, f"time_s={time_s} looks like per-step, not total"


def test_refit_keeps_median_estimates(tmp_path):
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    with ResultDB(db_path) as db:
        _seed_runs(db, "rpi", 10, rng)
        # The first-seeded conf has an observed run, so its estimate is
        # 'median'. It should stay 'median' after refit.
        run_conf = next(iter(db.iter_confs_by_precision(Precision.FP32)))[1]
        run_id = db.get_conf_id(run_conf)
        before = db.get_time_estimate(run_id, "rpi")
        assert before[1] == "median"

    _run_thread(db_path, [("refit", ("rpi", Precision.FP32))])

    with ResultDB(db_path, readonly=True) as db:
        after = db.get_time_estimate(run_id, "rpi")
        assert after[1] == "median"
        assert after[0] == pytest.approx(before[0])


def test_refit_skips_when_below_min_samples(tmp_path, monkeypatch):
    # Force a higher threshold than we're feeding.
    monkeypatch.setattr(TrainTimingModel, "MIN_SAMPLES", 100)
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    with ResultDB(db_path) as db:
        _seed_runs(db, "rpi", 10, rng)
        unrun_conf = _conf(batch=4, length=64)
        unrun_id = db.find_or_add_conf(unrun_conf)

    _run_thread(db_path, [("refit", ("rpi", Precision.FP32))])

    with ResultDB(db_path, readonly=True) as db:
        # No model was fit, so no predicted estimates should appear.
        assert db.get_time_estimate(unrun_id, "rpi") is None


def test_bootstrap_fits_all_pairs():
    rng = np.random.default_rng(0)
    db = ResultDB()
    _seed_runs(db, "rpi", 8, rng)
    _seed_runs(db, "whitebox", 8, rng)
    unrun_conf = _conf(batch=4, length=64)
    unrun_id = db.find_or_add_conf(unrun_conf)

    bootstrap(db)

    for system in ("rpi", "whitebox"):
        est = db.get_time_estimate(unrun_id, system)
        assert est is not None, f"missing estimate for {system}"
        assert est[1] == "predicted"
        assert est[0] > 0.1, (
            f"time_s={est[0]} on {system} looks like per-step, not total"
        )


def test_bootstrap_backfills_medians():
    """Bootstrap should populate medians for confs that have runs but
    no conf_time_estimate row (i.e. just after a schema migration)."""
    rng = np.random.default_rng(0)
    db = ResultDB()
    _seed_runs(db, "rpi", 8, rng)

    # Simulate the post-migration state: wipe the estimate table so the
    # 'median' rows are absent even though runs exist.
    db._db.execute("DELETE FROM conf_time_estimate")
    db._db.commit()

    bootstrap(db)

    # Every conf with a run on rpi should now have a median estimate.
    run_conf = next(iter(db.iter_confs_by_precision(Precision.FP32)))[1]
    run_id = db.get_conf_id(run_conf)
    time_s, source = db.get_time_estimate(run_id, "rpi")
    assert source == "median"
    assert time_s > 0
