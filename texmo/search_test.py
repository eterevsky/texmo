from copy import deepcopy

from .common import INF
from .configuration import Configuration, Precision, Template
from .db import DbReader, DbWriter
from .spec_parser import parse_model2
from .predict.loss_rnn import LossModelHolder
from .predict.timing import TrainTimingModel
from .run import Run
from .configuration import conf_neighbors
from . import search as search_mod
from .search import (
    _WARMUP_LADDER,
    _WARMUP_SELECTS_PER_RUNG,
    Search,
    _generate_limits,
    _is_retired,
    _layer_capped_template,
    _run_limit_sequences,
)


def _make_conf(steps=1024, batch=32, spec="bytes|dense.8.gelu"):
    model = parse_model2(spec, precision=Precision.FP32)
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
    result = search._select_neighbor_fewest_runs(
        conf, 'test', search.template)
    if result is not None:
        neighbor, _, _ = result
        assert neighbor != conf, (
            "neighbor selection returned the seed conf — cap should "
            "have dropped it after collapsing the 2x-steps variant")


def _make_template(num_layers=None):
    return Template(
        spec=None, precision=list(Precision),
        lr=(0, INF), length=(1, INF), batch=(1, INF),
        steps=(4, INF), max_weights=(2, INF),
        num_layers=num_layers,
    )


def test_layer_capped_template_derivation():
    base = _make_template()
    # No cap -> base template object, unchanged.
    t, cap = _layer_capped_template(base, None)
    assert t is base and cap is None
    # A cap intersects to (0, cap) on a copy; base is untouched.
    t, cap = _layer_capped_template(base, 2)
    assert cap == 2 and t is not base
    assert t.num_layers.min == 0 and t.num_layers.max == 2
    assert base.num_layers.max == INF
    # Cap below the base floor -> unrestricted.
    floored = _make_template(num_layers=(2, INF))
    t, cap = _layer_capped_template(floored, 1)
    assert t is floored and cap is None
    # Cap at or above the base max -> already implied, unrestricted.
    tight = _make_template(num_layers=(0, 2))
    for c in (2, 4):
        t, cap = _layer_capped_template(tight, c)
        assert t is tight and cap is None
    # Cap inside the base range -> applies, keeps the base floor.
    mid = _make_template(num_layers=(1, 8))
    t, cap = _layer_capped_template(mid, 3)
    assert cap == 3 and t.num_layers.min == 1 and t.num_layers.max == 3


def test_layer_capped_template_prunes_neighbors():
    base = _make_template()
    conf = _make_conf(spec="bytes|dense.8.gelu")  # 1 hidden layer
    capped, cap = _layer_capped_template(base, 1)
    assert cap == 1
    # Under the base template some mutations grow the chain...
    assert any(
        n.model.num_layers > 1 for n in conf_neighbors(conf, base))
    # ...under the capped template none do (and some survive).
    capped_neighbors = conf_neighbors(conf, capped)
    assert capped_neighbors
    assert all(n.model.num_layers <= 1 for n in capped_neighbors)


def _make_search_at(path, warmup_systems=None):
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
        warmup_systems=warmup_systems,
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
    # Stats recorded for the client log: 0 covered of 1 total.
    assert search._coverage_stats['b'] == (0, 1)


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
        template=search.template,
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
        template=search.template,
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
        t=4.0, max_weights=INF, system='a', bfs_depth=2,
        template=search.template)
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

    def fake_uncovered(system):
        search._coverage_stats[system] = (3, 10)
        return sentinel
    monkeypatch.setattr(search, '_select_uncovered_top', fake_uncovered)
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
    # Coverage progress is attached for the client log.
    assert res.covered == 3 and res.total == 10
    d = res.to_dict()
    assert d['covered'] == 3 and d['total'] == 10


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

# --- warmup ladder ---------------------------------------------------


def _big_conf():
    """A conf whose knobs are all far above the first warmup rung."""
    return Configuration(
        model=parse_model2("bytes|dense.8.gelu", precision=Precision.FP32),
        lr=0.01, length=1024, batch=256, steps=8192, decay=1,
    )


def _small_conf(spec="bytes|dense.4.gelu"):
    """A conf the first rung admits: steps at the template floor (4,
    which outranks the rung's cap of 2), batch and length at the caps."""
    return Configuration(
        model=parse_model2(spec, precision=Precision.FP32),
        lr=0.01, length=64, batch=1, steps=4, decay=1,
    )


def test_warmup_picks_an_existing_conf_unchanged(tmp_path):
    """The rung FILTERS the fleet's top confs rather than rewriting
    them, so the pick is a conf other systems have already run -- it
    keeps its knobs and its existing runs."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    small = _small_conf()
    _seed_runs(path, small, system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])

    conf = search._select_warmup('b')
    assert conf == small          # identical, not a shrunken variant
    assert search.template.match(conf)


def test_warmup_skips_confs_above_the_rung(tmp_path):
    """A conf whose knobs exceed rung 0 is not selected there. The
    ladder climbs until a rung admits it."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    big = _big_conf()
    _seed_runs(path, big, system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])

    conf = search._select_warmup('b')
    # Either it climbed to a rung that admits the conf, or the ladder
    # ran out -- never a rewritten variant of it.
    if conf is not None:
        assert conf == big
        steps, batch, length = _WARMUP_LADDER[search._warmup_rung['b']]
        assert big.steps <= max(steps, search.template.steps.min)
    assert search._warmup_rung['b'] > 0


def test_warmup_prefers_the_tight_rung_before_climbing(tmp_path):
    """With both a small and a big conf seeded, rung 0 yields the small
    one; the big one only becomes reachable further up the ladder."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    small, big = _small_conf(), _big_conf()
    _seed_runs(path, small, system='a', n=2)
    _seed_runs(path, big, system='a', n=2, loss_base=0.2)
    search = _make_search_at(path, warmup_systems=['b'])

    assert search._select_warmup('b') == small
    assert search._warmup_rung.get('b', 0) == 0


def test_warmup_advances_when_rung_is_covered(tmp_path):
    """With the rung's only conf already run on this system, the
    ladder steps up instead of returning nothing."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    small = _small_conf()
    _seed_runs(path, small, system='a', n=2)
    _seed_runs(path, small, system='b', n=1)   # covered on 'b'
    search = _make_search_at(path, warmup_systems=['b'])

    search._select_warmup('b')
    assert search._warmup_rung['b'] > 0


def test_warmup_advances_after_picks_spent(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    _seed_runs(path, _small_conf(), system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])

    for _ in range(_WARMUP_SELECTS_PER_RUNG):
        assert search._select_warmup('b') is not None
    assert search._warmup_rung['b'] == 1
    assert search._warmup_selects['b'] == 0


def test_warmup_ladder_is_monotone():
    """Every rung raises exactly one knob and never lowers any."""
    for (s0, b0, l0), (s1, b1, l1) in zip(
        _WARMUP_LADDER, _WARMUP_LADDER[1:]
    ):
        assert (s1 >= s0 and b1 >= b0 and l1 >= l0)
        assert sum([s1 > s0, b1 > b0, l1 > l0]) == 1


def test_select_conf_warmup_takes_priority_then_falls_through(tmp_path):
    """Warmup outranks the ordinary strategies while the ladder runs,
    and the system leaves warmup once it is climbed."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    _seed_runs(path, _small_conf(), system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])

    result = search.select_conf('b')
    assert result.strategy == 'warmup'

    # Ladder exhausted -> ordinary selection, and the flag clears.
    search._warmup_rung['b'] = len(_WARMUP_LADDER)
    result = search.select_conf('b')
    assert result is None or result.strategy != 'warmup'
    assert 'b' not in search.warmup_systems


def test_select_conf_ignores_warmup_for_other_systems(tmp_path):
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    _seed_runs(path, _small_conf(), system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])
    result = search.select_conf('c')
    assert result is None or result.strategy != 'warmup'


def test_warmup_caches_top_confs_per_rung(tmp_path, monkeypatch):
    """`top_confs_global` is one of the priciest queries on the live DB,
    and warmup fires on every select for its system -- it must be
    fetched once per rung, not once per pick."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    _seed_runs(path, _small_conf(), system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])

    calls = []
    real = search._db.top_confs_global
    monkeypatch.setattr(
        search._db, 'top_confs_global',
        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    for _ in range(_WARMUP_SELECTS_PER_RUNG):
        search._select_warmup('b')
    assert len(calls) == 1
    # Advancing the rung refreshes the candidate list.
    search._select_warmup('b')
    assert len(calls) == 2


# --- retired inputs --------------------------------------------------
#
# RETIRED_INPUTS is empty right now (tokens.64.shift was retired
# 2026-07-28 and re-enabled 2026-07-29), so these tests retire shift
# synthetically to keep the mechanism covered for the next retirement.

_SHIFT_SPEC = "tokens.64.shift.emb.4|rnn.4.tanh"


def _shift_conf(steps=256):
    return _make_conf(steps=steps, spec=_SHIFT_SPEC)


def _retire_shift(monkeypatch):
    monkeypatch.setattr(
        search_mod, 'RETIRED_INPUTS', ('tokens.64.shift',))


def test_is_retired_covers_both_codec_spellings(monkeypatch):
    _retire_shift(monkeypatch)
    assert _is_retired("tokens.64.shift.emb.4|rnn.4.tanh")
    assert _is_retired("tokens.64.shift.oh|rnn.4.tanh")
    assert _is_retired("tokens.64.shift.emb.16|split.add(dense.16.tanh)")
    # The hexbpe bridge it points at is very much alive.
    assert not _is_retired("tokens.64.hexbpe.emb.4|rnn.4.tanh")
    assert not _is_retired("tokens.32.fold.oh|rnn.4.tanh")
    assert not _is_retired("bytes|dense.8.gelu")


def test_nothing_is_retired_by_default():
    for spec in (_SHIFT_SPEC, "tokens.64.hexbpe.emb.4|rnn.4.tanh",
                 "bytes|dense.8.gelu"):
        assert not _is_retired(spec)


def test_retired_conf_is_never_re_run(tmp_path, monkeypatch):
    _retire_shift(monkeypatch)
    """A shift conf already in the DB is a top conf on every ordinary
    re-run path -- and none of them may hand it back."""
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _shift_conf()
    _seed_runs(path, conf, system='a', n=2)
    search = _make_search_at(path)

    # It IS in the DB's top confs -- the filter is what excludes it,
    # not an empty query.
    top = list(search._db.top_confs_for_system(
        max_time=16.0, max_weights=INF, system='a', limit=10,
        template=search.template))
    assert [c.conf for c in top] == [conf]

    # Neighbour walk: with mutation sources yielding nothing, the walk
    # would fall back to re-running the top conf itself.
    monkeypatch.setattr(
        search, '_select_neighbor_fewest_runs', lambda c, s, t: None)
    assert search._select_top_neighbor(
        16.0, INF, 'a', search.template) is None
    # Time-budget scan and coverage walk: pure re-run pickers.
    assert search._select_time_budget(
        16.0, INF, 'a', search.template) is None
    assert search._select_uncovered_top('b') is None
    # ...and the retired conf doesn't latch the sticky coverage flag on.
    assert search._coverage_flag['b'] is False


def test_retired_conf_is_still_a_mutation_source(tmp_path, monkeypatch):
    """Its neighbours must keep being generated -- the hexbpe bridge is
    how the shift population migrates."""
    _retire_shift(monkeypatch)
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _shift_conf()
    _seed_runs(path, conf, system='a', n=2)
    search = _make_search_at(path)

    specs = {str(n.model) for n in conf_neighbors(conf, search.template)}
    assert any(s.startswith('tokens.64.hexbpe') for s in specs), specs

    result = search._select_neighbor_fewest_runs(
        conf, 'a', search.template)
    assert result is not None
    neighbor, _, _ = result
    assert not _is_retired(str(neighbor.model))

    # End to end: the walk reaches a live conf THROUGH the retired one.
    picked = search._select_top_neighbor(16.0, INF, 'a', search.template)
    assert picked is not None
    assert not _is_retired(str(picked.model))


def test_warmup_skips_retired_confs(tmp_path, monkeypatch):
    _retire_shift(monkeypatch)
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    _seed_runs(path, _small_conf(spec=_SHIFT_SPEC), system='a', n=2)
    search = _make_search_at(path, warmup_systems=['b'])
    # Nothing schedulable at any rung -> the ladder runs out.
    assert search._select_warmup('b') is None
    assert search._warmup_rung['b'] == len(_WARMUP_LADDER)


def test_predicted_best_expands_retired_seed_but_never_picks_it(
    tmp_path, monkeypatch,
):
    import numpy as np
    _retire_shift(monkeypatch)
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    seed = _shift_conf()
    _seed_runs(path, seed, system='a', n=3)
    search = _make_search_at(path)

    monkeypatch.setattr(
        search.timing_model, 'predict',
        lambda system, conf: 16.0 * conf.steps / 1024.0)
    monkeypatch.setattr(
        search.timing_model, 'predict_max_steps',
        lambda system, conf, t: 1024)

    scored: list[Configuration] = []

    class FakeLossModel:
        def is_ready(self): return True
        def predict(self, confs):
            scored.extend(confs)
            return np.full(len(confs), -1.0, dtype=np.float32)

    search.loss_model = FakeLossModel()

    picked = search._select_predicted_best_impl(
        t=16.0, max_weights=INF, system='a', bfs_depth=2,
        template=search.template)
    assert picked is not None
    assert not _is_retired(str(picked.model))
    # The retired seed was expanded (its neighbours are in the pool)
    # but no retired conf was ever scored as a candidate.
    assert not any(_is_retired(str(c.model)) for c in scored)
    assert scored


def test_select_conf_never_returns_a_retired_conf(tmp_path, monkeypatch):
    """Whole-select smoke: with only a shift conf in the DB, no
    strategy -- including the default fallback -- may schedule it."""
    _retire_shift(monkeypatch)
    path = str(tmp_path / "test.db")
    DbWriter(path).close()
    conf = _shift_conf()
    _seed_runs(path, conf, system='a', n=2)
    writer = DbWriter(path)
    writer.add_pick_me_conf(conf)   # even an explicit pick is refused
    writer.close()
    search = _make_search_at(path)
    search.init_conf = conf
    for _ in range(20):
        result = search.select_conf('b')
        if result is not None:
            assert not _is_retired(str(result.conf.model)), result.strategy
