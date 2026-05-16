from copy import deepcopy

from .common import INF
from .configuration import Configuration, Precision, Template
from .db import DbReader, DbWriter
from .model import build_model_def
from .predict.loss_rnn import LossModelHolder
from .predict.timing import TrainTimingModel
from .search import Search, _generate_limits, _run_limit_sequences


def _make_conf(steps=1024):
    model = build_model_def("bytes|dense.8.gelu", precision=Precision.FP32)
    return Configuration(
        model=model, lr=0.01, length=128, batch=32, steps=steps, decay=1,
    )


def _make_search(tmp_path):
    """Build a Search with a real DbReader on a fresh DB. `_predicted_time`
    is monkeypatched in individual tests to script the prediction source."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()  # apply schema
    return Search(
        reader=DbReader(path),
        template=Template(
            spec=None, precision=list(Precision),
            lr=(0, INF), length=(1, INF), batch=(1, INF),
            steps=(4, INF), max_weights=(2, INF),
        ),
        init_conf=_make_conf(),
        train_time=(1.0, 16.0),
        timing_model=TrainTimingModel(),
        loss_model=LossModelHolder(),
    )


def test_cap_steps_under_budget_unchanged(tmp_path, monkeypatch):
    search = _make_search(tmp_path)
    monkeypatch.setattr(search, '_predicted_time', lambda c, s: 4.0)
    assert search._cap_steps(_make_conf(steps=1024), 'test').steps == 1024


def test_cap_steps_no_prediction_unchanged(tmp_path, monkeypatch):
    """Cold start: no timing model, no median — leave the conf alone so
    the first run on this (system, precision) seeds the predictor."""
    search = _make_search(tmp_path)
    monkeypatch.setattr(search, '_predicted_time', lambda c, s: None)
    assert search._cap_steps(_make_conf(steps=1024), 'test').steps == 1024


def test_cap_steps_halves_until_under_budget(tmp_path, monkeypatch):
    """64s at 1024 steps, scales linearly. maxt=16. Expect: 256 steps
    (predicted 16s, exactly at budget)."""
    search = _make_search(tmp_path)
    monkeypatch.setattr(
        search, '_predicted_time',
        lambda c, s: 64.0 * (c.steps / 1024.0))
    assert search._cap_steps(_make_conf(steps=1024), 'test').steps == 256


def test_cap_steps_floors_at_min_steps(tmp_path, monkeypatch):
    """Always over budget — should hit the template's steps.min and
    return the conf at that floor (bounded overrun)."""
    search = _make_search(tmp_path)
    monkeypatch.setattr(search, '_predicted_time', lambda c, s: 1000.0)
    capped = search._cap_steps(_make_conf(steps=1024), 'test')
    assert capped.steps == 4  # template.steps.min


def test_run_limit_sequences():
    seqs = []
    for i, s in enumerate(_run_limit_sequences()):
        seqs.append(list(s))
        if i == 3:
            break
    assert seqs == [
        [1],
        [2, 1, 1],
        [3, 2, 2, 1, 1, 1, 1, 1, 1],
        [4, 3, 3] + [2] * 6 + [1] * 18,
    ]


def test_generate_limits():
    limits = []
    for seq in _generate_limits():
        limits.append(deepcopy(seq))
        if len(limits) == 16: break

    assert limits == [
        [[1, 0]],
        [[2, 0]],
        [[2, 1]],
        [[2, 1], [1, 0]],
        [[3, 1], [1, 0]],
        [[3, 2], [1, 0]],
        [[3, 2], [2, 0]],
        [[3, 2], [2, 1]],
        [[3, 2], [2, 1], [1, 0]],
        [[4, 2], [2, 1], [1, 0]],
        [[4, 3], [2, 1], [1, 0]],
        [[4, 3], [3, 1], [1, 0]],
        [[4, 3], [3, 2], [1, 0]],
        [[4, 3], [3, 2], [2, 0]],
        [[4, 3], [3, 2], [2, 1]],
        [[4, 3], [3, 2], [2, 1], [1, 0]],
    ]