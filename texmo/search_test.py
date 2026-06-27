from copy import deepcopy

from .common import INF
from .configuration import Configuration, Precision, Template
from .db import DbReader, DbWriter
from .model import build_model_def
from .predict.loss_rnn import LossModelHolder
from .predict.timing import TrainTimingModel
from .run import Run
from .search import Search, _generate_limits, _run_limit_sequences


def _make_conf(steps=1024, batch=32, spec="bytes|dense.8.gelu"):
    model = build_model_def(spec, precision=Precision.FP32)
    return Configuration(
        model=model, lr=0.01, length=128, batch=batch, steps=steps, decay=1,
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


def _seed_runs(path, conf, system, n, loss_base=0.5):
    """Add `n` runs of `conf` on `system` so it qualifies as a top
    conf (`num_runs > 1` + non-null `median_score`)."""
    writer = DbWriter(path)
    try:
        for i in range(n):
            writer.add_run(conf, Run(
                system=system, step_loss=[0.1],
                loss=loss_base + 0.01 * i, train_time=2.0))
    finally:
        writer.close()


def test_select_uncovered_top_empty_db(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    search = _make_search_at(path)
    assert search._select_uncovered_top('b') is None
    assert search._coverage_flag.get('b', False) is False


def test_select_uncovered_top_one_uncovered_no_flag(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _make_conf(steps=256)
    _seed_runs(path, conf, system='a', n=2)  # qualifies as top
    search = _make_search_at(path)
    result = search._select_uncovered_top('b')
    assert result == conf
    # Only one uncovered conf — flag stays False.
    assert search._coverage_flag.get('b', False) is False


def test_select_uncovered_top_below_threshold_no_flag(tmp_path):
    """Two uncovered confs isn't enough to set the sticky flag with
    `_COVERAGE_STICKY_THRESHOLD=5`."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    # Distinct specs (different weight counts) and a strictly better
    # loss for the larger one so both surface in `top_confs_global`'s
    # Pareto-improving stream.
    confA = _make_conf(steps=256, spec="bytes|dense.8.gelu")
    confB = _make_conf(steps=256, spec="bytes|dense.16.gelu")
    _seed_runs(path, confA, system='a', n=2, loss_base=0.5)
    _seed_runs(path, confB, system='a', n=2, loss_base=0.3)
    search = _make_search_at(path)
    result = search._select_uncovered_top('b')
    assert result in (confA, confB)
    assert search._coverage_flag.get('b', False) is False


def test_select_uncovered_top_at_threshold_sets_flag(
    tmp_path, monkeypatch,
):
    """With `_COVERAGE_STICKY_THRESHOLD` uncovered confs the next
    select on this system fires unconditionally."""
    from texmo.db.reader import ConfScore
    from texmo import search as search_mod

    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    search = _make_search_at(path)

    # Build a fake Pareto front big enough to clear the threshold.
    n = search_mod._COVERAGE_STICKY_THRESHOLD
    fake = [
        ConfScore(
            conf_id=i,
            conf=_make_conf(steps=256, batch=8 + i),
            median_score=0.5 - 0.01 * i,
            system='a',
            median_time=1.0,
            num_runs=2,
        )
        for i in range(n)
    ]
    monkeypatch.setattr(
        search._db, 'top_confs_global',
        lambda template, **kwargs: iter(fake))
    monkeypatch.setattr(
        search._db, 'has_covering_run',
        lambda conf, system: False)  # all uncovered

    result = search._select_uncovered_top('b')
    assert result is not None
    assert search._coverage_flag['b'] is True


def test_select_uncovered_top_all_covered(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _make_conf(steps=256)
    _seed_runs(path, conf, system='a', n=2)
    _seed_runs(path, conf, system='b', n=1)
    search = _make_search_at(path)
    assert search._select_uncovered_top('b') is None
    assert search._coverage_flag['b'] is False


def test_select_uncovered_top_higher_steps_covers(tmp_path):
    """A run with more steps than the capped variant counts as covered
    — no need to re-run the same architecture at fewer steps."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf_query = _make_conf(steps=256)
    conf_higher = _make_conf(steps=1024)
    _seed_runs(path, conf_query, system='a', n=2)   # makes it a top conf
    _seed_runs(path, conf_higher, system='b', n=1)  # covers conf_query on 'b'
    search = _make_search_at(path)
    # conf_query is still uncovered on 'b' since steps=1024 >= 256 means
    # the higher-steps run covers it. No selection.
    assert search._select_uncovered_top('b') is None


def test_select_predicted_best_no_seed_returns_none(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    search = _make_search_at(path)
    # No runs at all -> no top conf for 'a' -> no seed -> None.
    assert search._select_predicted_best_impl(
        t=4.0, max_weights=INF, system='a', bfs_depth=2,
    ) is None


def test_select_predicted_best_no_timing_returns_none(
    tmp_path, monkeypatch,
):
    """Seed exists, but the timing model isn't fit — every _norm()
    returns None, adjusted is empty, function returns None."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    seed_conf = _make_conf(steps=256, spec="bytes|dense.8.gelu")
    _seed_runs(path, seed_conf, system='a', n=3)
    search = _make_search_at(path)
    # predict returns None for every conf (no model fit).
    monkeypatch.setattr(
        search.timing_model, 'predict', lambda system, conf: None)
    assert search._select_predicted_best_impl(
        t=4.0, max_weights=INF, system='a', bfs_depth=2,
    ) is None


def test_select_predicted_best_happy_path(tmp_path, monkeypatch):
    """End-to-end: seed exists, timing+loss models stubbed, depth-2 BFS
    runs, a Configuration comes back. Time scales linearly with steps
    so a band of step values is feasible — locks in that the function
    actually exercises (W, T) variety from the neighbor walk and that
    over-budget step counts get capped rather than dropped."""
    import numpy as np
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    seed_conf = _make_conf(steps=256, spec="bytes|dense.8.gelu")
    _seed_runs(path, seed_conf, system='a', n=3)
    search = _make_search_at(path)

    # max_t = train_time[1] = 16.0. Time = 16.0 * (steps / 1024), so
    # any steps <= 1024 fits; > 1024 overflows. Returning 1024 from
    # predict_max_steps covers every conf with the same cap.
    monkeypatch.setattr(
        search.timing_model, 'predict',
        lambda system, conf: 16.0 * conf.steps / 1024.0)
    monkeypatch.setattr(
        search.timing_model, 'predict_max_steps',
        lambda system, conf, t: 1024)

    # Record what gets scored — lets us assert on the candidate pool.
    scored_confs: list[Configuration] = []

    class FakeLossModel:
        def is_ready(self): return True
        def predict(self, confs):
            scored_confs.extend(confs)
            return np.full(len(confs), -1.0, dtype=np.float32)

    search.loss_model = FakeLossModel()

    picked = search._select_predicted_best_impl(
        t=4.0, max_weights=INF, system='a', bfs_depth=2)
    assert picked is not None
    assert isinstance(picked, Configuration)

    # Sanity on what was scored:
    assert scored_confs, "loss model never invoked"
    # Every scored conf must fit max_t -- either under-budget natively
    # or capped to 1024.
    assert all(c.steps <= 1024 for c in scored_confs)
    # And (W, T) variety survives the post-BFS normalization: multiple
    # distinct steps values appear in the candidate pool, not just the
    # cap value. (Step neighbors include c.steps/2 and c.steps*2 from
    # the seed at 256; depth 2 reaches 64 and 1024.)
    distinct_steps = {c.steps for c in scored_confs}
    assert len(distinct_steps) >= 3, (
        f"expected multiple step values in candidates; got {distinct_steps}")


def test_select_conf_returns_pick_me_with_priority(tmp_path):
    """A pick_me=1 conf with no runs must win over every other strategy."""
    path = str(tmp_path / "test.db")
    writer = DbWriter(path)
    pm_conf = _make_conf(steps=256, spec="bytes|dense.16.gelu")
    _, inserted = writer.add_pick_me_conf(pm_conf)
    assert inserted
    writer.close()
    search = _make_search_at(path)
    result = search.select_conf('rpi')
    assert result is not None
    assert result.strategy == 'pick_me'
    assert result.conf == pm_conf


def test_select_conf_skips_pick_me_after_min_runs(tmp_path):
    """Once a pick_me conf has accumulated its min_runs threshold, the
    strategy yields and lower-priority strategies fire instead."""
    path = str(tmp_path / "test.db")
    writer = DbWriter(path)
    pm_conf = _make_conf(steps=256, spec="bytes|dense.16.gelu")
    writer.add_pick_me_conf(pm_conf)
    # Two runs (PICK_ME_MIN_RUNS) retires the flag's effect.
    writer.add_run(pm_conf, Run(
        system="rpi", step_loss=[0.1], loss=0.5, train_time=2.0))
    writer.add_run(pm_conf, Run(
        system="rpi", step_loss=[0.1], loss=0.6, train_time=2.0))
    writer.close()
    search = _make_search_at(path)
    result = search.select_conf('rpi')
    # Whatever fires must not be the pick_me strategy.
    assert result is None or result.strategy != 'pick_me'


def test_timing_ready_reflects_fitted_pairs(tmp_path):
    from .predict.timing import Weights
    search = _make_search(tmp_path)
    assert search._timing_ready('sys') is False
    search.timing_model._weights[('sys', Precision.FP32)] = Weights(
        {}, {}, {}, {})
    assert search._timing_ready('sys') is True
    assert search._timing_ready('other') is False


def test_select_conf_holds_coverage_walk_until_timing_ready(
    tmp_path, monkeypatch,
):
    """The coverage walk must not fire on a system with no fitted timing
    model (it can't cap the global top confs); once one is fit, it
    fires."""
    from .predict.timing import Weights
    search = _make_search(tmp_path)
    sentinel = _make_conf(spec="bytes|dense.16.gelu")
    monkeypatch.setattr(
        search, '_select_uncovered_top', lambda system: sentinel)
    # Sticky so the walk would fire unconditionally if not gated.
    search._coverage_flag['sys'] = True

    # No timing model for 'sys' -> walk held back (some other strategy
    # or the default fires, but never coverage_walk).
    res = search.select_conf('sys')
    assert res is None or res.strategy != 'coverage_walk'

    # Fit a timing model for the pair -> the walk fires.
    search.timing_model._weights[('sys', Precision.FP32)] = Weights(
        {}, {}, {}, {})
    res = search.select_conf('sys')
    assert res is not None and res.strategy == 'coverage_walk'
    assert res.conf == sentinel


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