"""Integration tests for ModelThread.

Uses a file-backed DB because the thread opens its own connections
and an in-memory DB would give it a fresh, empty database. The thread
is told to stop after the work messages are enqueued; ordering of a
single-consumer Queue guarantees the work messages are processed before
"stop".
"""

import dataclasses
import datetime
import os
import pickle
from queue import Queue

import numpy as np
import pytest

from texmo.configuration import Configuration
from texmo.db import DbReader, DbWriter
from texmo.db.writer import Stop as WriterStop, WriterThread
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.predict import loss_tree, model_thread
from texmo.predict.loss_rnn import LossModelHolder
from texmo.predict.model_thread import (
    LossRefit,
    ModelThread,
    RunAdded,
    Stop,
    bootstrap,
)
from texmo.predict.persist import predictor_path
from texmo.predict.timing import TrainTimingModel
from texmo.run import Run


@dataclasses.dataclass
class _StubLossModel:
    """Stands in for a fitted TreeLossModel; module-level so the
    publish path can pickle it."""
    tag: int
    simple_types: tuple = ()
    per_type_fwd: bool = True


def _conf(
    spec="bytes|dense.8.gelu",
    batch=8,
    length=32,
    steps=100,
    precision=Precision.FP32,
):
    model = parse_model2(spec, precision=precision)
    return Configuration(
        model=model, lr=0.01, length=length, batch=batch, steps=steps,
        decay=1,
    )


def _seed_runs(
    writer: DbWriter, system: str, n: int, rng: np.random.Generator,
):
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
        writer.add_run(
            conf, run, timestamp=base + datetime.timedelta(seconds=i))


def _run_thread(
    db_path: str, messages: list, loss_model: LossModelHolder | None = None,
    loss_refit_local: bool = False,
):
    """Run ModelThread with messages + Stop; block until done.

    Spins up a real `WriterThread` so the writes ModelThread posts
    actually land in the DB before assertions run.
    """
    write_queue = Queue()
    writer_thread = WriterThread(db_path, write_queue)
    writer_thread.start()

    q = Queue()
    thread = ModelThread(
        db_path, q,
        write_queue=write_queue,
        timing_model=TrainTimingModel(),
        loss_model=loss_model or LossModelHolder(),
        loss_refit_local=loss_refit_local,
    )
    thread.start()
    for m in messages:
        q.put(m)
    q.put(Stop())
    thread.join(timeout=60)
    assert not thread.is_alive(), "model thread did not stop"

    write_queue.put(WriterStop())
    writer_thread.join(timeout=60)
    assert not writer_thread.is_alive(), "writer thread did not stop"


@pytest.fixture(autouse=True)
def _lower_min_samples(monkeypatch):
    # Fit is expensive with MIN_SAMPLES=20; lower for fast tests.
    monkeypatch.setattr(TrainTimingModel, "MIN_SAMPLES", 5)
    # A single RunAdded should drive a refit in tests (both the
    # established-pair and the first-fit cadence).
    monkeypatch.setattr(model_thread, "_REFIT_EVERY", 1)
    monkeypatch.setattr(model_thread, "_FIRST_FIT_EVERY", 1)


def test_refit_writes_predicted_estimates(tmp_path):
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    with DbWriter(db_path) as writer:
        _seed_runs(writer, "rpi", 10, rng)
        unrun_conf = _conf(batch=4, length=64)
        unrun_id = writer.find_or_add_conf(unrun_conf)

    _run_thread(db_path, [RunAdded(system="rpi", precision=Precision.FP32)])

    with DbReader(db_path) as reader:
        est = reader.get_time_estimate(unrun_id, "rpi")
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
    with DbWriter(db_path) as writer, DbReader(db_path) as reader:
        _seed_runs(writer, "rpi", 10, rng)
        # The first-seeded conf has an observed run, so its estimate is
        # 'median'. It should stay 'median' after refit.
        run_conf = next(iter(reader.iter_confs_by_precision(Precision.FP32)))[1]
        run_id = reader.get_conf_id(run_conf)
        before = reader.get_time_estimate(run_id, "rpi")
        assert before[1] == "median"

    _run_thread(db_path, [RunAdded(system="rpi", precision=Precision.FP32)])

    with DbReader(db_path) as reader:
        after = reader.get_time_estimate(run_id, "rpi")
    assert after[1] == "median"
    assert after[0] == pytest.approx(before[0])


def test_refit_skips_when_below_min_samples(tmp_path, monkeypatch):
    # Force a higher threshold than we're feeding.
    monkeypatch.setattr(TrainTimingModel, "MIN_SAMPLES", 100)
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    with DbWriter(db_path) as writer:
        _seed_runs(writer, "rpi", 10, rng)
        unrun_conf = _conf(batch=4, length=64)
        unrun_id = writer.find_or_add_conf(unrun_conf)

    _run_thread(db_path, [RunAdded(system="rpi", precision=Precision.FP32)])

    with DbReader(db_path) as reader:
        # No model was fit, so no predicted estimates should appear.
        assert reader.get_time_estimate(unrun_id, "rpi") is None


def _stub_loss_fit(monkeypatch) -> list:
    """Patch the loss fit to a stub; returns the list it appends to.

    The real fit is minutes long, so what these tests exercise is the
    cadence and the publish plumbing, not the model."""
    fitted = []

    def _fake_train(reader):
        fitted.append(reader)
        return _StubLossModel(len(fitted))

    monkeypatch.setattr(loss_tree, "train_loss_model", _fake_train)
    return fitted


def test_loss_refit_fires_once_per_cadence(tmp_path, monkeypatch):
    """With the local trigger on, labeled runs drive the loss refit:
    one fit per `LOSS_REFIT_EVERY` runs, published to the holder and
    to disk."""
    db_path = str(tmp_path / "test.db")
    with DbWriter(db_path):
        pass
    monkeypatch.setattr(model_thread, "LOSS_REFIT_EVERY", 3)
    fitted = _stub_loss_fit(monkeypatch)

    holder = LossModelHolder()
    _run_thread(
        db_path,
        [RunAdded(system="rpi", precision=Precision.FP32)] * 5,
        loss_model=holder,
        loss_refit_local=True,
    )

    assert len(fitted) == 1, "expected exactly one refit at 5 runs, cadence 3"
    assert holder.is_ready()
    with open(predictor_path(db_path, 'loss'), 'rb') as f:
        assert pickle.load(f).tag == 1


def test_no_local_loss_refit_when_workers_own_it(tmp_path, monkeypatch):
    """With the local trigger off (the server grants the fit to a
    worker instead) the same runs produce no fit here."""
    db_path = str(tmp_path / "test.db")
    with DbWriter(db_path):
        pass
    monkeypatch.setattr(model_thread, "LOSS_REFIT_EVERY", 3)
    fitted = _stub_loss_fit(monkeypatch)

    holder = LossModelHolder()
    _run_thread(
        db_path,
        [RunAdded(system="rpi", precision=Precision.FP32)] * 5,
        loss_model=holder,
        loss_refit_local=False,
    )

    assert fitted == []
    assert not holder.is_ready()
    assert not os.path.exists(predictor_path(db_path, 'loss'))


def test_startup_loss_refit_message_works_in_both_modes(
    tmp_path, monkeypatch,
):
    """`LossRefit` is the startup fallback and predates the
    distributed flow: it must fit even when workers own the cadence."""
    db_path = str(tmp_path / "test.db")
    with DbWriter(db_path):
        pass
    fitted = _stub_loss_fit(monkeypatch)

    holder = LossModelHolder()
    _run_thread(
        db_path, [LossRefit()], loss_model=holder,
        loss_refit_local=False,
    )

    assert len(fitted) == 1
    assert holder.is_ready()


def test_bootstrap_fits_all_pairs(tmp_path):
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    writer = DbWriter(db_path)
    reader = DbReader(db_path)
    _seed_runs(writer, "rpi", 8, rng)
    _seed_runs(writer, "whitebox", 8, rng)
    unrun_conf = _conf(batch=4, length=64)
    unrun_id = writer.find_or_add_conf(unrun_conf)

    bootstrap(reader, writer, TrainTimingModel())

    for system in ("rpi", "whitebox"):
        est = reader.get_time_estimate(unrun_id, system)
        assert est is not None, f"missing estimate for {system}"
        assert est[1] == "predicted"
        assert est[0] > 0.1, (
            f"time_s={est[0]} on {system} looks like per-step, not total"
        )
    reader.close()
    writer.close()


def test_bootstrap_backfills_medians(tmp_path):
    """Bootstrap should populate medians for confs that have runs but
    no conf_time_estimate row (i.e. just after a schema migration)."""
    db_path = str(tmp_path / "test.db")
    rng = np.random.default_rng(0)
    writer = DbWriter(db_path)
    reader = DbReader(db_path)
    _seed_runs(writer, "rpi", 8, rng)

    # Simulate the post-migration state: wipe the estimate table so the
    # 'median' rows are absent even though runs exist.
    writer._db.execute("DELETE FROM conf_time_estimate")
    writer._db.commit()

    bootstrap(reader, writer, TrainTimingModel())

    # Every conf with a run on rpi should now have a median estimate.
    run_conf = next(iter(reader.iter_confs_by_precision(Precision.FP32)))[1]
    run_id = reader.get_conf_id(run_conf)
    time_s, source = reader.get_time_estimate(run_id, "rpi")
    assert source == "median"
    assert time_s > 0

    reader.close()
    writer.close()
