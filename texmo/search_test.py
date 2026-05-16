from copy import deepcopy

from .common import INF
from .configuration import Configuration, Precision, Template
from .db import DbReader, DbWriter
from .model import build_model_def
from .predict.loss_rnn import LossModelHolder
from .predict.timing import TrainTimingModel
from .run import Run
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


def test_select_neighbor_fewest_runs_drops_seed_when_cap_collides(
    tmp_path, monkeypatch,
):
    """Step-neighbors include `conf.steps * 2` and `conf.steps / 2`. If
    the *2 variant is capped back down to `conf.steps`, it'd land on
    the seed — `_select_neighbor_fewest_runs` must not return the seed
    as its own neighbor."""
    search = _make_search(tmp_path)
    conf = _make_conf(steps=512)
    # Pretend everything overflows budget proportionally to steps, so
    # the steps=1024 neighbor caps to 512 (== seed.steps).
    monkeypatch.setattr(
        search, '_predicted_time',
        lambda c, s: 64.0 * (c.steps / 1024.0))
    result = search._select_neighbor_fewest_runs(conf, 'test')
    if result is not None:
        neighbor, _, _ = result
        assert neighbor != conf, (
            "neighbor selection returned the seed conf — cap should "
            "have dropped it after collapsing the 2x-steps variant")


def _make_search_at(path):
    """Like `_make_search` but reuses an existing DB at `path` so tests
    can seed runs through a separate `DbWriter` first."""
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


def _seed_runs(path, conf, system, n):
    """Add `n` runs of `conf` on `system` so it qualifies as a top
    conf (`num_runs > 1` + non-null `median_score`)."""
    writer = DbWriter(path)
    try:
        for i in range(n):
            writer.add_run(conf, Run(
                system=system, step_loss=[0.1],
                loss=0.5 + 0.01 * i, train_time=2.0))
    finally:
        writer.close()


def test_select_global_top_empty_db(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    search = _make_search_at(path)
    assert search._select_global_top(max_weights=10_000, system='b') is None


def test_select_global_top_returns_uncovered_conf(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _make_conf(steps=256)
    _seed_runs(path, conf, system='a', n=2)
    search = _make_search_at(path)
    result = search._select_global_top(max_weights=10_000, system='b')
    assert result is not None
    # No timing-model fit → `_cap_steps` is a no-op, so we get the conf
    # back unchanged.
    assert result == conf


def test_select_global_top_skips_when_covered(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _make_conf(steps=256)
    _seed_runs(path, conf, system='a', n=2)
    _seed_runs(path, conf, system='b', n=1)
    search = _make_search_at(path)
    assert search._select_global_top(
        max_weights=10_000, system='b') is None


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