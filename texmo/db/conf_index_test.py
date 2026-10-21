"""Tests for `ConfIndex` and the index-backed query paths.

The core is `test_index_matches_sql_under_random_mutations`: a DB fed
through every real write path (plus raw SQL), with index-backed
results compared against the SQL reference implementations -- and the
incrementally synced index compared against a fresh build -- after
every round. The rest pin down the individual consistency mechanisms:
trigger coverage, rebuild triggers (prune gap, threshold, rollback,
epoch, schema), snapshot pinning, and the shadow / drift self-checks.
"""

import random
import sqlite3

from texmo.common import INF
from texmo.configuration import Configuration, DecayType, Template
from texmo.db import DbReader, DbWriter
from texmo.db.conf_index import ConfIndex
from texmo.precision import Precision
from texmo.run import Run
from texmo.spec_parser import parse_model2


_SPECS = [
    'bytes|dense.64.gelu',
    'bytes|dense.32.gelu',
    'bytes|dense.16.tanh',
    'bytes|dense.16.tanh-dense.16.tanh',
    'bytes|rnn.8.tanh',
    'bits.1+bp|',
    'bits.1+bp|rnn.2.tanh',
    'bits.1+bp|rnn.4.tanh-dense.8.tanh',
    'bits.1+bp|rnn.1.tanh-msr.1.1.nope-dense.1.tanh',
    'bits.2.emb.1|msr.1.1.nope',
    'bits.1+bp|norm-rnn.2.tanh',  # parses, but is_valid() is False
]
_REGEXES = [
    'bytes.*', 'bits.*', r'.*rnn.*', r'.*dense\.(16|32).*', 'nomatch',
    r'.*\|(rnn\.\d+\.\w+-)?(msr|attn).*',
]
_SYSTEMS = ['sys-a', 'sys-b', 'sys-c', 'sys-d']
# Few distinct losses and times, so medians and scores tie often.
_LOSSES = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
_TIMES = [1.0, 2.0, 5.0, 10.0]


def _random_conf(rng: random.Random) -> Configuration:
    precision = rng.choice(
        [Precision.FP32, Precision.FP32, Precision.FP16, Precision.BF16])
    model = parse_model2(rng.choice(_SPECS), precision)
    kind = rng.choice(['none', 'exp', 'cosine'])
    return Configuration(
        model=model,
        lr=rng.choice([0.01, 0.05, 0.1]),
        length=rng.choice([16, 32, 64]),
        batch=rng.choice([4, 8, 16]),
        steps=rng.choice([2, 8, 32, 128]),
        decay=rng.choice([0.9, 0.99]) if kind == 'exp' else 1,
        cosine=kind == 'cosine',
    )


def _random_template(rng: random.Random) -> Template:
    spec = rng.choice([
        None, None, None,
        rng.choice(_SPECS),  # literal (parses -> canonical spec)
        rng.choice(_REGEXES), rng.choice(_REGEXES),
    ])

    def maybe(lo, hi):
        return None if rng.random() < 0.4 else (rng.choice(lo), rng.choice(hi))

    precisions = list(Precision)
    if rng.random() < 0.6:
        precisions = rng.sample(precisions, rng.randint(0, len(precisions)))
    return Template(
        spec=spec,
        lr=maybe([0, 0.01, 0.05], [INF, 0.05, 0.1]),
        length=maybe([1, 16, 32], [INF, 32, 64]),
        batch=maybe([1, 4, 8], [INF, 8, 16]),
        steps=maybe([1, 2, 8, 9], [INF, 32, 128]),
        max_weights=maybe([16, 100], [INF, 100, 9000, 20000]),
        precision=precisions,
        decay_types=rng.sample(list(DecayType), rng.randint(0, 3)),
        num_layers=maybe([0, 1, 2], [INF, 1, 2]),
    )


def _conf_ids(writer: DbWriter) -> list[int]:
    return [r[0] for r in writer._db.execute('SELECT id FROM conf')]


def _top_key(cs) -> tuple:
    return (cs.conf_id, cs.median_score, cs.median_time, cs.num_runs,
            str(cs.conf.model), cs.conf.lr, cs.conf.steps)


def _cut_key(c) -> tuple:
    return (c.conf_id, c.median_score, c.time_estimate, c.total_runs,
            c.system_runs, str(c.conf.model))


def _mutate(writer: DbWriter, rng: random.Random, pool: list) -> str:
    """Apply one random write through a real write path (or raw SQL).
    Returns its name, for failure messages."""
    db = writer._db
    ids = _conf_ids(writer)
    op = rng.choice([
        'add_run', 'add_run', 'add_run', 'add_run', 'predicted',
        'upsert', 'score', 'pick_me', 'num_layers',
        'delete_conf', 'replace_conf', 'delete_estimate',
        'update_all_scores',
    ])
    if rng.random() < 0.01:
        op = 'clear_system'  # wipes a quarter of everything; keep rare
    if op == 'add_run':
        if not pool or rng.random() < 0.6:
            pool.append(_random_conf(rng))
        conf = rng.choice(pool)
        time_s = None if rng.random() < 0.1 else rng.choice(_TIMES)
        writer.add_run(
            conf,
            Run(system=rng.choice(_SYSTEMS), loss=rng.choice(_LOSSES),
                train_time=time_s),
            track_winner_change=rng.random() < 0.7)
    elif op == 'predicted' and ids:
        writer.upsert_predicted_time_estimates(
            (cid, rng.choice(_SYSTEMS), rng.choice(_TIMES))
            for cid in rng.sample(ids, min(len(ids), 15)))
    elif op == 'upsert' and ids:
        writer.upsert_time_estimates(
            (cid, rng.choice(_SYSTEMS), rng.choice(_TIMES),
             rng.choice(['median', 'predicted']))
            for cid in rng.sample(ids, min(len(ids), 8)))
    elif op == 'clear_system':
        writer.clear_system(rng.choice(_SYSTEMS))
    elif op == 'score' and ids:
        db.execute(
            'UPDATE conf SET median_score = ? WHERE id = ?',
            (rng.choice(_LOSSES + [None]), rng.choice(ids)))
        db.commit()
    elif op == 'pick_me':
        conf = _random_conf(rng)
        pool.append(conf)
        writer.add_pick_me_conf(conf, runs=rng.randint(1, 3))
    elif op == 'num_layers' and ids:
        db.execute(
            'UPDATE conf SET num_layers = ? WHERE id = ?',
            (rng.choice([None, 0, 1, 2, 3]), rng.choice(ids)))
        db.commit()
    elif op == 'delete_conf' and ids:
        # Raw delete, foreign keys off: leaves orphan runs/estimates,
        # which the SQL never joins to and the index must ignore too.
        db.execute('DELETE FROM conf WHERE id = ?', (rng.choice(ids),))
        db.commit()
    elif op == 'replace_conf' and ids:
        # INSERT OR REPLACE onto an existing conf's unique key: SQLite
        # deletes the old row without firing DELETE triggers; the
        # BEFORE INSERT trigger is what logs it.
        row = db.execute(
            'SELECT spec, weights, lr, length, batch, steps, precision, '
            'decay, cosine FROM conf WHERE id = ?',
            (rng.choice(ids),)).fetchone()
        db.execute(
            'INSERT OR REPLACE INTO conf (spec, weights, lr, length, '
            'batch, steps, precision, decay, cosine, num_layers, '
            'median_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            tuple(row) + (rng.choice([None, 1, 2]),
                          rng.choice(_LOSSES + [None])))
        db.commit()
    elif op == 'delete_estimate' and ids:
        db.execute(
            'DELETE FROM conf_time_estimate WHERE conf_id = ? '
            'AND system = ?', (rng.choice(ids), rng.choice(_SYSTEMS)))
        db.commit()
    elif op == 'update_all_scores':
        writer.update_all_scores()
    return op


def _fresh_state(reader: DbReader) -> tuple[dict, ConfIndex]:
    fresh = ConfIndex()
    with reader._snapshot():
        fresh.sync(reader._db)
    return fresh.state(), fresh


def _compare_queries(reader: DbReader, rng: random.Random, n: int, ctx):
    for _ in range(n):
        template = _random_template(rng)
        system = rng.choice(_SYSTEMS + ['unknown'])
        mw = rng.choice([None, 0, 20, 100, 9000, 20000, INF])
        mt = rng.choice([None, 0, 1.0, 2.0, 5.0, 100.0])
        limit = rng.choice([None, 1, 3, 10])
        case = (f'{ctx}: top_confs_for_system({system!r}, {template}, '
                f'max_weights={mw}, max_time={mt}, limit={limit})')
        got = [_top_key(c) for c in reader.top_confs_for_system(
            system, template, mw, mt, limit)]
        want = [_top_key(c) for c in reader._top_confs_for_system_sql(
            system, template, mw, mt, limit)]
        assert got == want, f'{case}\nindex={got}\nsql={want}'

        mw = rng.choice([None, 20, 9000, 20000, INF])
        mt = rng.choice([None, 1.0, 2.0, 5.0, 100.0])
        case = (f'{ctx}: confs_under_time({system!r}, {template}, '
                f'max_weights={mw}, max_time={mt})')
        got = [_cut_key(c) for c in reader.confs_under_time(
            template, system, mw, mt)]
        want = [_cut_key(c) for c in reader._confs_under_time_sql(
            template, system, mw, mt)]
        assert got == want, f'{case}\nindex={got}\nsql={want}'


def _compare_top_conf_at(writer: DbWriter, rng: random.Random, ctx):
    """Inside one write transaction: index vs SQL, then again after an
    uncommitted change the index must see."""
    cur = writer._db.cursor()
    cur.execute('BEGIN IMMEDIATE')
    ids = _conf_ids(writer)
    for step in range(2):
        for _ in range(10):
            args = (rng.choice(_SYSTEMS + ['unknown']),
                    rng.choice([20, 100, 9000, 20000, 70000]),
                    rng.choice(_TIMES + [0.5, 100.0]))
            got = writer._top_conf_at(cur, *args)
            want = writer._top_conf_at_sql(cur, *args)
            assert got == want, f'{ctx} step {step}: _top_conf_at{args}'
        if ids:
            cur.execute(
                'UPDATE conf SET median_score = ? WHERE id = ?',
                (rng.choice(_LOSSES), rng.choice(ids)))
    cur.execute('COMMIT')


def test_index_matches_sql_under_random_mutations(tmp_path):
    rng = random.Random(1234)
    path = str(tmp_path / 'test.db')
    # The writer's own index answers every changed_winner sample, with
    # the SQL shadow on every call; the reader's index answers the
    # comparisons below (its shadow off, the test does it explicitly).
    writer = DbWriter(path, conf_index=True, shadow_every=1)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    assert reader.conf_index is not None
    pool: list = []
    for _ in range(2000):
        _mutate(writer, rng, pool)
    for round_ in range(40):
        ops = [_mutate(writer, rng, pool) for _ in range(rng.randint(1, 12))]
        ctx = f'round {round_} after {ops}'
        _compare_queries(reader, rng, 25, ctx)
        state, fresh = _fresh_state(reader)
        index = reader.conf_index
        assert index.state() == state, ctx
        assert index.order_ok(), ctx
        assert fresh.order_ok()
        _compare_top_conf_at(writer, rng, ctx)
    assert reader.conf_index.builds == 1, reader.conf_index.last_build_reason
    checks, mismatches = writer.shadow_stats
    assert checks > 100 and mismatches == 0
    # Enough variety to mean something.
    (n_confs, n_null, n_nl_null) = writer._db.execute(
        'SELECT COUNT(*), SUM(median_score IS NULL), '
        'SUM(num_layers IS NULL) FROM conf').fetchone()
    assert n_confs > 250 and n_null > 5 and n_nl_null > 5
    sources = dict(writer._db.execute(
        'SELECT source, COUNT(*) FROM conf_time_estimate '
        'GROUP BY source').fetchall())
    assert sources['median'] > 50 and sources['predicted'] > 50
    (n_systems,) = writer._db.execute(
        'SELECT COUNT(DISTINCT system) FROM conf_time_estimate').fetchone()
    assert n_systems >= 3
    (ties,) = writer._db.execute(
        'SELECT COUNT(*) - COUNT(DISTINCT median_score) FROM conf '
        'WHERE median_score IS NOT NULL').fetchone()
    assert ties > 100
    reader.close()
    writer.close()


def test_changed_winner_same_with_and_without_index(tmp_path):
    rng_a, rng_b = random.Random(7), random.Random(7)
    with_index = DbWriter(str(tmp_path / 'a.db'), conf_index=True,
                          shadow_every=0)
    plain = DbWriter(str(tmp_path / 'b.db'))
    assert with_index.conf_index is not None and plain.conf_index is None
    pool_a: list = []
    pool_b: list = []
    for _ in range(600):
        _mutate(with_index, rng_a, pool_a)
        _mutate(plain, rng_b, pool_b)
    q = 'SELECT id, conf_id, changed_winner FROM run ORDER BY id'
    a = with_index._db.execute(q).fetchall()
    b = plain._db.execute(q).fetchall()
    assert [tuple(r) for r in a] == [tuple(r) for r in b]
    assert sum(r[2] == 1 for r in a) > 2
    assert sum(r[2] == 0 for r in a) > 5


def _setup(tmp_path, n_runs=60, seed=3):
    rng = random.Random(seed)
    path = str(tmp_path / 'test.db')
    writer = DbWriter(path)
    pool: list = []
    for _ in range(n_runs):
        if not pool or rng.random() < 0.5:
            pool.append(_random_conf(rng))
        writer.add_run(rng.choice(pool), Run(
            system=rng.choice(_SYSTEMS), loss=rng.choice(_LOSSES),
            train_time=rng.choice(_TIMES)))
    return path, writer, rng


def _base_template() -> Template:
    return Template(
        spec=None, lr=None, length=None, batch=None, steps=None,
        max_weights=None, precision=list(Precision))


def _assert_same(reader: DbReader, system='sys-a'):
    t = _base_template()
    got = [_top_key(c) for c in reader.top_confs_for_system(system, t)]
    want = [_top_key(c) for c in reader._top_confs_for_system_sql(system, t)]
    assert got == want
    got = [_cut_key(c) for c in reader.confs_under_time(t, system, INF, 100.0)]
    want = [_cut_key(c)
            for c in reader._confs_under_time_sql(t, system, INF, 100.0)]
    assert got == want


def _changes(writer: DbWriter) -> int:
    return writer._db.execute('SELECT COUNT(*) FROM conf_change').fetchone()[0]


def test_every_write_path_logs_changes(tmp_path):
    path, writer, rng = _setup(tmp_path, n_runs=5)
    db = writer._db
    conf = _random_conf(rng)

    def logs(fn) -> int:
        before = _changes(writer)
        fn()
        db.commit()
        return _changes(writer) - before

    cid = writer.find_or_add_conf(conf)
    assert _changes(writer) > 0
    assert logs(lambda: writer.add_run(conf, Run(
        system='sys-z', loss=1.0, train_time=1.0))) >= 2
    assert logs(lambda: writer.upsert_predicted_time_estimates(
        [(cid, 'sys-y', 2.0)])) == 1
    assert logs(lambda: writer.upsert_predicted_time_estimates(
        [(cid, 'sys-y', 3.0)])) == 1  # the DO UPDATE
    assert logs(lambda: writer.upsert_predicted_time_estimates(
        [(cid, 'sys-z', 3.0)])) == 0  # guarded: median row untouched
    assert logs(lambda: writer.upsert_time_estimates(
        [(cid, 'sys-y', 4.0, 'median')])) == 1
    assert logs(lambda: writer.add_pick_me_conf(conf, runs=3)) == 1
    assert logs(lambda: writer.add_pick_me_conf(
        _random_conf(random.Random(99)), runs=2)) >= 1
    assert logs(lambda: writer.clear_system('sys-y')) == 1
    assert logs(writer.update_all_scores) > 0
    db.execute('UPDATE conf SET num_layers = NULL WHERE id = ?', (cid,))
    db.commit()
    assert logs(writer.backfill_num_layers) >= 1
    # >= 1: on a connection with foreign keys on (this one created the
    # schema), the cascade's estimate deletes are logged too.
    assert logs(lambda: db.execute(
        'DELETE FROM conf WHERE id = ?', (cid,))) >= 1
    # Every logged row names a real conf id, and system only for
    # estimate changes.
    bad = db.execute(
        'SELECT COUNT(*) FROM conf_change WHERE conf_id IS NULL').fetchone()
    assert bad[0] == 0


def test_replace_logs_the_evicted_conf(tmp_path):
    path, writer, rng = _setup(tmp_path, n_runs=5)
    db = writer._db
    (victim,) = db.execute('SELECT MIN(id) FROM conf').fetchone()
    row = db.execute(
        'SELECT spec, weights, lr, length, batch, steps, precision, decay,'
        ' cosine FROM conf WHERE id = ?', (victim,)).fetchone()
    seq = db.execute('SELECT MAX(seq) FROM conf_change').fetchone()[0]
    db.execute(
        'INSERT OR REPLACE INTO conf (spec, weights, lr, length, batch, '
        'steps, precision, decay, cosine) VALUES (?,?,?,?,?,?,?,?,?)',
        tuple(row))
    db.commit()
    logged = {r[0] for r in db.execute(
        'SELECT conf_id FROM conf_change WHERE seq > ?', (seq,))}
    assert victim in logged
    assert db.execute(
        'SELECT COUNT(*) FROM conf WHERE id = ?', (victim,)).fetchone()[0] == 0


def test_prune_gap_forces_rebuild(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    index = reader.conf_index
    _assert_same(reader)
    assert index.builds == 1

    # Within the kept window: incremental catch-up, no rebuild.
    writer.add_run(_random_conf(rng), Run(
        system='sys-a', loss=1.0, train_time=1.0))
    assert writer.prune_conf_change(keep=1000) == 0
    _assert_same(reader)
    assert index.builds == 1

    # Changes the reader hasn't seen, then a prune past its position.
    for _ in range(20):
        writer.add_run(_random_conf(rng), Run(
            system='sys-a', loss=rng.choice(_LOSSES), train_time=1.0))
    assert writer.prune_conf_change(keep=5) > 0
    floor = writer._db.execute(
        'SELECT floor FROM conf_change_floor').fetchone()[0]
    assert floor > index.last_seq
    _assert_same(reader)
    assert index.builds == 2
    assert 'pruned past us' in index.last_build_reason
    reader.close()


def test_far_behind_rebuilds(tmp_path):
    path, writer, rng = _setup(tmp_path)
    index = ConfIndex(rebuild_rows=10)
    reader = DbReader(path)
    reader._index = index
    _assert_same(reader)
    for _ in range(3):
        writer.add_run(_random_conf(rng), Run(
            system='sys-a', loss=1.0, train_time=1.0))
    _assert_same(reader)
    assert index.builds == 1
    writer.update_all_scores()
    _assert_same(reader)
    assert index.builds == 2 and 'behind' in index.last_build_reason


def test_epoch_and_schema_changes_rebuild(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    index = reader.conf_index
    writer._db.execute("UPDATE conf_change_floor SET epoch = 'other'")
    writer._db.commit()
    _assert_same(reader)
    assert index.builds == 2 and 'epoch' in index.last_build_reason
    writer._db.execute('CREATE INDEX test_extra ON conf(lr)')
    writer._db.commit()
    _assert_same(reader)
    assert index.builds == 3 and 'schema' in index.last_build_reason


def test_rolled_back_writer_sync_is_detected(tmp_path):
    """The writer's index syncs its own uncommitted rows; if that
    transaction rolls back, the changelog's AUTOINCREMENT goes back
    and the next sync must rebuild rather than trust its seq."""
    path, writer, rng = _setup(tmp_path)
    writer.enable_conf_index()
    index = writer.conf_index
    cur = writer._db.cursor()
    cur.execute('BEGIN IMMEDIATE')
    cur.execute('UPDATE conf SET median_score = 0.001')
    index.sync(writer._db)
    best = writer._top_conf_at(cur, 'sys-a', 10**9, 100.0)
    assert writer._db.execute(
        'SELECT median_score FROM conf WHERE id = ?', (best,)
    ).fetchone()[0] == 0.001
    cur.execute('ROLLBACK')
    cur.execute('BEGIN IMMEDIATE')
    assert (writer._top_conf_at(cur, 'sys-a', 10**9, 100.0)
            == writer._top_conf_at_sql(cur, 'sys-a', 10**9, 100.0))
    cur.execute('COMMIT')
    assert index.builds == 2 and 'backwards' in index.last_build_reason


def test_failed_add_run_invalidates_writer_index(tmp_path, monkeypatch):
    path, writer, rng = _setup(tmp_path)
    writer.enable_conf_index()
    conf = _random_conf(rng)

    def boom(*args, **kwargs):
        raise RuntimeError('disk on fire')

    monkeypatch.setattr(writer, '_update_scores', boom)
    try:
        writer.add_run(conf, Run(system='sys-a', loss=1.0, train_time=1.0),
                       track_winner_change=True)
    except RuntimeError:
        pass
    else:
        assert False, 'expected the write to fail'
    assert not writer.conf_index.built
    assert not writer._db.in_transaction
    monkeypatch.undo()
    assert writer.add_run(
        conf, Run(system='sys-a', loss=1.0, train_time=1.0),
        track_winner_change=True) is not None
    assert writer.conf_index.built


def test_write_between_sync_and_fetch_is_invisible(tmp_path):
    """The eager path: a write committed right after the index synced
    must not leak into the rows fetched for that same query."""
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    t = _base_template()
    before = [_top_key(c) for c in reader._top_confs_for_system_sql(
        'sys-a', t, limit=10)]
    assert before
    index = reader.conf_index
    real_sync = index.sync

    def sync_then_write(conn):
        real_sync(conn)
        # Another connection commits: worsen the current best conf and
        # add a strong new one.
        other = sqlite3.connect(path)
        other.execute('UPDATE conf SET median_score = 99 WHERE id = ?',
                      (before[0][0],))
        other.commit()
        other.close()
        index.sync = real_sync

    index.sync = sync_then_write
    got = [_top_key(c) for c in reader.top_confs_for_system(
        'sys-a', t, limit=10)]
    assert got == before
    # And the next query sees the write.
    after = [_top_key(c) for c in reader.top_confs_for_system(
        'sys-a', t, limit=10)]
    assert after != before
    assert after == [_top_key(c) for c in reader._top_confs_for_system_sql(
        'sys-a', t, limit=10)]


def test_lazy_query_keeps_its_snapshot(tmp_path):
    path, writer, rng = _setup(tmp_path, n_runs=600)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    t = _base_template()
    want = [_cut_key(c) for c in reader._confs_under_time_sql(
        t, 'sys-a', INF, 100.0)]
    assert len(want) > 40  # spans several fetch chunks
    stream = reader.confs_under_time(t, 'sys-a', INF, 100.0)
    got = [_cut_key(next(stream))]
    # Commit changes to rows the stream hasn't fetched yet.
    writer._db.execute('UPDATE conf SET median_score = median_score / 2')
    writer._db.execute('DELETE FROM conf WHERE id = ?', (want[-1][0],))
    writer._db.commit()
    got += [_cut_key(c) for c in stream]
    assert got == want


def test_shadow_mismatch_answers_from_sql_and_rebuilds(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=1)
    t = _base_template()
    want = [_top_key(c) for c in reader._top_confs_for_system_sql(
        'sys-a', t, limit=5)]
    index = reader.conf_index
    # Corrupt the index so a drift check can't see it: drop the best
    # conf from the order (every row it does return is still correct).
    index._order = index._order[1:]
    index._order_score = index._order_score[1:]
    got = [_top_key(c) for c in reader.top_confs_for_system(
        'sys-a', t, limit=5)]
    assert got == want
    assert reader.shadow_stats == (1, 1)
    assert not index.built
    got = [_top_key(c) for c in reader.top_confs_for_system(
        'sys-a', t, limit=5)]
    assert got == want
    assert reader.shadow_stats == (2, 1)
    assert index.builds == 2


def test_lazy_shadow_mismatch_answers_from_sql(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=1)
    t = _base_template()
    want = [_cut_key(c) for c in reader._confs_under_time_sql(
        t, 'sys-a', INF, 100.0)]
    index = reader.conf_index
    index._order = index._order[::-1].copy()
    got = [_cut_key(c) for c in reader.confs_under_time(
        t, 'sys-a', INF, 100.0)]
    assert got == want
    assert reader.shadow_stats == (1, 1)


def test_drift_check_falls_back_to_sql(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    t = _base_template()
    want = [_top_key(c) for c in reader._top_confs_for_system_sql('sys-a', t)]
    index = reader.conf_index
    index._score[index._order[0]] = -1.0  # row disagrees with SQL
    got = [_top_key(c) for c in reader.top_confs_for_system('sys-a', t)]
    assert got == want
    assert not index.built


def test_unknown_condition_falls_back(tmp_path, monkeypatch):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    t = _base_template()
    want = [_top_key(c) for c in reader._top_confs_for_system_sql('sys-a', t)]

    def extra_condition(template):
        return ['frobnicate = 1'], {}

    monkeypatch.setattr(
        'texmo.db.conf_index._make_template_conditions', extra_condition)
    got = [_top_key(c) for c in reader.top_confs_for_system('sys-a', t)]
    assert got == want
    assert reader.conf_index is not None  # only this query fell back


def test_missing_trigger_disables_index(tmp_path):
    path, writer, rng = _setup(tmp_path)
    reader = DbReader(path, conf_index=True, shadow_every=0)
    writer._db.execute('DROP TRIGGER conf_change_conf_update')
    writer._db.commit()
    _assert_same(reader)  # schema change -> rebuild -> trigger missing
    assert reader.conf_index is None
    assert DbReader(path, conf_index=True).conf_index is None
    # The next read-write open restores it.
    DbWriter(path).close()
    assert DbReader(path, conf_index=True).conf_index is not None


def test_db_without_changelog_stays_on_sql(tmp_path):
    path = str(tmp_path / 'old.db')
    DbWriter(path).close()
    db = sqlite3.connect(path)
    db.executescript(
        'DROP TABLE conf_change; DROP TABLE conf_change_floor;')
    db.close()
    reader = DbReader(path, conf_index=True)
    assert reader.conf_index is None
    assert list(reader.top_confs_for_system('x', _base_template())) == []
