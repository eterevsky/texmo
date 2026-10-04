"""Light integration tests for ManagerJax.

Uses a minimal 'bits.1+bp|' model (no hidden layers, binary tokens)
trained on in-memory random bytes. Verifies the end-to-end pipeline
doesn't blow up, not the quality of training.
"""

import json
import logging
import math
import subprocess
from types import SimpleNamespace

import jax
import pytest

from texmo import manager_jax
from texmo.configuration import Configuration
from texmo.dataset import DataSet, DataSetWrapper
from texmo.manager import create_manager
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _make_dataset():
    return DataSet(data=b'hello world ' * 200)


def _make_conf(steps: int = 3):
    return Configuration(
        parse_model2('bits.1+bp|', precision=Precision.FP32),
        lr=0.01,
        length=32,
        batch=8,
        steps=steps,
        decay=1.0,
    )


def test_train_and_eval():
    manager = create_manager(
        'jax', conf=_make_conf(), system='test',
        dataset=_make_dataset(),
        test_sample_len=128, test_batch=4,
        verbose=False,
    )
    run, final_conf = manager.train_and_eval(steps=3, time_limit=None)
    assert run.steps == 3
    assert run.loss is not None
    # Random model on 2 tokens has loss near 1.0 b/byte; just check it's finite.
    assert 0 < run.loss < 100
    assert final_conf.steps == 3


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        create_manager(
            'nosuchbackend', conf=_make_conf(), system='test',
            dataset=_make_dataset(),
            test_sample_len=128, test_batch=4,
            verbose=False,
        )


@pytest.mark.parametrize('spec', [
    'bits.1+bp|suffix.2',
    'bits.1+bp|suffix.4-dense.4.gelu',
    'bits.2.oh+bp|dense.4.relu',
    'bits.1+bp|rnn.4.tanh',
    'bits.1+bp|rnn.4.gelu-dense.2.tanh',
    'bits.1+bp|gru.4',
    'bits.1+bp|gru.4-dense.2.tanh',
    'bits.1+bp|mgru.4',
    'bits.1+bp|mingru.4',
    'bits.1+bp|lstm.4',
    'bits.1+bp|latent.4.2',
    'bits.1+bp|lrnn.4.2',
    'bits.1+bp|split.add(dense.4.gelu, pass)-dense.8.gelu',
    'bits.1+bp|split.cat(dense.4.gelu-dense.4.gelu, pass)-dense.8.gelu',
    'bits.1+bp|split.cat(suffix.4-dense.4.tanh, pass)',
])
def test_train_various_specs(spec):
    """Consistency check: model builds and trains without shape errors."""
    conf = Configuration(
        parse_model2(spec, precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=2, decay=1.0,
    )
    manager = create_manager(
        'jax', conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    run, _ = manager.train_and_eval(steps=2, time_limit=None)
    assert run.steps == 2


@pytest.mark.parametrize('spec', [
    # Plain specs.
    'bits.1+bp|dense.4.gelu',
    'bits.1+bp|gru.4-dense.2.tanh',
    # Residual splits.
    'bits.1+bp|split.add(dense.4.gelu, pass)-dense.4.gelu',
    'bits.1+bp|split.cat(dense.4.gelu-dense.4.gelu, pass)-dense.4.gelu',
    # A gate split.
    'bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, pass)-dense.4.gelu',
])
def test_train_via_parse_model2_jax(spec):
    """End-to-end smoke for the Model2 train path -- the same path
    cli/train.py takes. Exercises Manager init, train_step, eval
    (which triggers the FP32 rebuild in manager_jax)."""
    conf = Configuration(
        parse_model2(spec, precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=2, decay=1.0,
    )
    manager = create_manager(
        'jax', conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    run, _ = manager.train_and_eval(steps=2, time_limit=None)
    assert run.steps == 2
    assert run.loss is not None


def test_emb_scale_logged(tmp_path, monkeypatch):
    """Tied-embedding runs append (X, exp(y)) to the local scale log;
    plain-codec runs don't."""
    log = tmp_path / 'emb_scale.jsonl'
    monkeypatch.setattr(manager_jax, '_EMB_SCALE_LOG', str(log))
    conf = Configuration(
        parse_model2('bytes.emb.2|dense.2.tanh', precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=2, decay=1.0,
    )
    manager = create_manager(
        'jax', conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    manager.train_and_eval(steps=2, time_limit=None)
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['x'] == 2
    assert abs(rec['scale'] - math.exp(rec['y'])) < 1e-9
    assert rec['spec'].startswith('bytes.emb.2|')
    assert rec['system'] == 'test'

    # A OneHotCodec model must not log.
    conf = Configuration(
        parse_model2('bits.1+bp|dense.4.gelu', precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=2, decay=1.0,
    )
    manager = create_manager(
        'jax', conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    manager.train_and_eval(steps=2, time_limit=None)
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize('spec', [
    'bits.1+bp|',
    'bits.1+bp|suffix.2',
    'bits.1+bp|suffix.4-dense.4.gelu',
    'bits.2.oh+bp|dense.4.relu',
    'bits.1+bp|rnn.4.tanh',
    'bits.1+bp|rnn.4.gelu-dense.2.tanh',
    'bits.1+bp|gru.4',
    'bits.1+bp|gru.4-dense.2.tanh',
    'bits.1+bp|mgru.4',
    'bits.1+bp|mingru.4',
    'bits.1+bp|lstm.4',
    'bits.1+bp|latent.4.2',
    'bits.1+bp|lrnn.4.2',
    'bits.1+bp|split.add(dense.4.gelu, pass)-dense.8.gelu',
    'bits.1+bp|split.cat(dense.4.gelu-dense.4.gelu, pass)-dense.8.gelu',
    'bits.1+bp|split.cat(suffix.4-dense.4.tanh, pass)',
    # Nested splits with mixed add and cat merging at the same point
    # (just before the output dense).
    'bits.1+bp|split.add(dense.4.gelu-split.cat(dense.4.gelu, pass), pass)'
    '-dense.8.gelu',
])
def test_jax_num_weights_matches_def(spec):
    """Model2Def.num_weights should equal the total element count of
    the JAX weight pytree produced by the manager.
    """
    conf = Configuration(
        parse_model2(spec, precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=1, decay=1.0,
    )
    manager = create_manager(
        'jax', conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    actual = sum(w.size for w in jax.tree.leaves(manager.weights))
    assert actual == conf.model.num_weights


def test_chunk_times_no_outlier_unchanged():
    """Steady chunks: the total is the plain sum."""
    times = [3.0] + [1.0] * 12
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(sum(times))


def test_chunk_times_single_outlier_corrected():
    """One suspend-sized gap is replaced by the median of the rest."""
    times = [1.0] * 6 + [31900.0] + [1.0] * 6
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    # 13 chunks of 1 s each: the 31900 s gap becomes the 1 s median.
    assert total == pytest.approx(13.0)


def test_chunk_times_correction_arithmetic_exact():
    """corrected = measured - outlier + median, on uneven chunks."""
    times = [5.0, 2.0, 2.5, 3.0, 2.0, 400.0, 3.5, 2.0, 3.0, 2.5]
    # Eligible sorted: 2, 2, 2, 2.5, [2.5], 3, 3, 3.5, 400 -> median 2.5.
    median = 2.5
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    assert total == pytest.approx(sum(times) - 400.0 + median)
    assert total == pytest.approx(28.0)


def test_chunk_times_two_outliers_raise():
    """A suspend straddling a chunk boundary smears into two chunks;
    that case is deliberately fatal rather than guessed at."""
    times = [1.0] * 5 + [20000.0, 12000.0] + [1.0] * 6
    with pytest.raises(RuntimeError, match='anomalous chunk times'):
        manager_jax._correct_chunk_times(times)

    # Two 5-hour naps over realistic 30 s chunks: over both the factor
    # (10x the 30 s median) and the floor, so still fatal.
    times = [30.0] * 5 + [18000.0, 18000.0] + [30.0] * 6
    with pytest.raises(RuntimeError, match='anomalous chunk times'):
        manager_jax._correct_chunk_times(times)


def test_chunk_times_too_few_chunks_unchanged():
    """Under the minimum, nothing is corrected and nothing raises --
    not even with one or several would-be outliers."""
    # 8 chunks -> 7 eligible, one short of _SUSPEND_MIN_CHUNKS.
    one = [1.0] * 4 + [5000.0] + [1.0] * 3
    total, n = manager_jax._correct_chunk_times(one)
    assert (total, n) == (pytest.approx(sum(one)), 0)

    several = [1.0] * 3 + [5000.0, 7000.0] + [1.0] * 3
    total, n = manager_jax._correct_chunk_times(several)
    assert (total, n) == (pytest.approx(sum(several)), 0)

    # A 512-step run is 2 chunks; a 0-step one is none at all.
    assert manager_jax._correct_chunk_times([9.0, 1.0]) == (10.0, 0)
    assert manager_jax._correct_chunk_times([]) == (0.0, 0)


def test_chunk_times_min_chunks_boundary():
    """Exactly _SUSPEND_MIN_CHUNKS eligible chunks do get corrected."""
    n_eligible = manager_jax._SUSPEND_MIN_CHUNKS
    assert n_eligible == 8
    times = [1.0] + [1.0] * 4 + [900.0] + [1.0] * 3
    assert len(times) - 1 == n_eligible
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    assert total == pytest.approx(9.0)


def test_chunk_times_first_chunk_excluded_but_counted():
    """Chunk 0 carries the JIT compile: it never triggers the outlier
    test and never enters the median, but its time stays in the total."""
    times = [500.0] + [1.0] * 12
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(512.0)

    # It also must not drag the median up and thereby mask a real gap
    # among the eligible chunks.
    times = [500.0] + [1.0] * 6 + [900.0] + [1.0] * 5
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    assert total == pytest.approx(512.0)


def test_chunk_times_outlier_threshold_is_strict():
    """Exactly _SUSPEND_OUTLIER_FACTOR x median is not an outlier.

    Chunks are 10 s here so that the factor, not the absolute floor,
    is the binding condition -- 10x a 10 s median is well past
    _SUSPEND_MIN_GAP_S.
    """
    times = [10.0] * 6 + [100.0] + [10.0] * 6
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(sum(times))

    # Just above it, the factor does bind.
    times = [10.0] * 6 + [101.0] + [10.0] * 6
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    assert total == pytest.approx(130.0)


def test_chunk_times_floor_is_strict():
    """Exactly _SUSPEND_MIN_GAP_S is not an outlier either."""
    floor = manager_jax._SUSPEND_MIN_GAP_S
    times = [1.0] * 6 + [floor] + [1.0] * 6
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(sum(times))

    # Just above it -- and 10x the 1 s median as well -- it corrects.
    times = [1.0] * 6 + [floor + 0.1] + [1.0] * 6
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 1
    assert total == pytest.approx(13.0)


def test_chunk_times_burst_stalls_are_not_suspends():
    """The input-bound incident: 512 chunks with a 217 ms median, one
    in every ten swallowing a ~2.2 s wait for the next prefetch burst.

    Ten-plus times the median, but seconds, not hours: the pipeline,
    not a nap. Nothing is corrected and -- the part that actually
    crashed a worker -- nothing raises."""
    times = [1.0]  # chunk 0: JIT compile.
    for i in range(1, 512):
        times.append(2.2 + 0.01 * (i % 50) if i % 10 == 0 else 0.217)

    # Without the absolute floor these all look like suspends, and
    # more than one is fatal.
    over_factor = [t for t in times[1:] if t > 10.0 * 0.217]
    assert len(over_factor) > 1

    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(math.fsum(times))


def test_chunk_times_queue_warmup_is_not_a_suspend():
    """The other incident: a 4096-step run on an SBC, 16 chunks with a
    91 ms median, the first few at ~1.4 s while the tokenizer-heavy
    sampler filled the prefetch queue."""
    times = [1.4, 1.4, 1.4] + [0.091] * 13
    assert len([t for t in times[1:] if t > 10.0 * 0.091]) > 1
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(math.fsum(times))


def test_chunk_times_single_sub_floor_gap_reported_as_measured():
    """One 2.5 s stall over a 217 ms median: over 10x, far under the
    floor. Not corrected either -- the time is reported as measured."""
    times = [1.0] + [0.217] * 6 + [2.5] + [0.217] * 5
    total, n = manager_jax._correct_chunk_times(times)
    assert n == 0
    assert total == pytest.approx(math.fsum(times))


def test_chunk_times_warning_logged(caplog):
    """The reported incident: 13 chunks of ~1 s, one of which swallowed
    an 8h52m suspend."""
    times = [1.0] * 6 + [31900.0] + [1.0] * 6
    with caplog.at_level(logging.WARNING):
        manager_jax._correct_chunk_times(times)
    assert 'suspend detected' in caplog.text
    assert '8 h 52 m' in caplog.text  # the outlier (and the old total)
    assert '1.00 s' in caplog.text    # the median
    assert '13.0 s' in caplog.text    # the corrected total


def test_input_bound_log_line(caplog):
    """One line either way; worded as a diagnosis only above the
    threshold, as a neutral statistic below it."""
    with caplog.at_level(logging.INFO):
        manager_jax._log_input_bound(117.0, 246.0)
    assert ('input-bound: 32% of wall time waiting for data '
            '(sample 1 m 57 s, compute 4 m 6 s)') in caplog.text

    # The boundary counts as input-bound.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        manager_jax._log_input_bound(10.0, 90.0)
    assert 'input-bound: 10% of wall time' in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        manager_jax._log_input_bound(1.0, 99.0)
    assert ('data wait: 1% of wall time '
            '(sample 1.00 s, compute 1 m 39 s)') in caplog.text
    assert 'input-bound' not in caplog.text

    # No chunks ran: nothing to report, and no division by zero.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        manager_jax._log_input_bound(0.0, 0.0)
    assert caplog.text == ''


def test_train_logs_sampling_compute_split(monkeypatch, caplog):
    """Wiring: the chunk loop times sampling and compute separately and
    the split reaches the summary line.

    The stubbed clock ticks once at the start and then twice per chunk
    (after sampling, after the host sync): 1 s of queue wait and 3 s of
    compute, three times over."""
    ticks = iter([0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0])
    monkeypatch.setattr(manager_jax, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 1)
    manager = create_manager(
        'jax', conf=_make_conf(), system='test',
        dataset=_make_dataset(),
        test_sample_len=64, test_batch=2,
        verbose=False,
    )
    with caplog.at_level(logging.INFO):
        total_time, _ = manager.train(steps=3, time_limit=None)
    # Three 4 s chunks, too few to correct.
    assert total_time == pytest.approx(12.0)
    assert ('input-bound: 25% of wall time waiting for data '
            '(sample 3.00 s, compute 9.00 s)') in caplog.text


def test_train_time_comes_from_chunk_correction(monkeypatch):
    """Wiring: the chunked loop times every chunk and returns whatever
    the correction says, which is what lands in run.train_time."""
    seen = []
    seen_records = []

    def fake_correct(chunk_times, records=None):
        seen.append(list(chunk_times))
        seen_records.append(list(records))
        return 42.0, 1

    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 1)
    monkeypatch.setattr(manager_jax, '_correct_chunk_times', fake_correct)
    manager = create_manager(
        'jax', conf=_make_conf(), system='test',
        dataset=_make_dataset(),
        test_sample_len=64, test_batch=2,
        verbose=False,
    )
    run, _ = manager.train_and_eval(steps=3, time_limit=None)
    assert len(seen) == 1
    assert len(seen[0]) == 3  # one entry per chunk, chunk size 1
    assert all(t >= 0 for t in seen[0])
    # The records ride along, one per chunk, and carry the same times.
    records = seen_records[0]
    assert [r.index for r in records] == [0, 1, 2]
    assert [(r.first_step, r.steps) for r in records] == [(0, 1), (1, 1), (2, 1)]
    assert [r.wall for r in records] == seen[0]
    # The jit cache grows on the first chunk only.
    assert [r.recompiled for r in records] == [True, False, False]
    # The corrected total is what gets reported -- and eval, which runs
    # after train(), is not folded into it.
    assert run.train_time == 42.0


def _record(index: int, wall: float, **fields):
    """A synthetic record of a 256-step chunk: all compute and all CPU
    unless overridden, nothing platform-specific."""
    base = dict(
        index=index, first_step=256 * index, steps=256, wall=wall,
        sample=0.0, compute=wall, clock=wall, cpu=wall,
        utime=None, stime=None, majflt=None, inblock=None, nivcsw=None,
        recompiled=None, queue_depth=None)
    base.update(fields)
    return manager_jax._ChunkRecord(**base)


def _incident_records():
    """The dedicated-4090 incident, replayed with Linux counters: 32
    chunks, a 1.08 s median, two outliers of 52.8 s (chunk 10) and
    31.1 s (chunk 21) in which the process was awake (clock = wall)
    yet barely used the CPU. Chunk 0 carries a 45 s compile -- over the
    floor, and never judged."""
    steady = dict(sample=0.002, compute=1.078, clock=1.08, cpu=1.25,
                  utime=1.05, stime=0.2, majflt=0, inblock=0, nivcsw=1,
                  recompiled=False, queue_depth=2)
    records = [_record(0, 45.0, sample=0.5, compute=44.5, clock=45.0,
                       cpu=47.0, utime=46.0, stime=1.0, majflt=12,
                       inblock=800, nivcsw=40, recompiled=True,
                       queue_depth=0)]
    for i in range(1, 32):
        if i == 10:
            records.append(_record(
                10, 52.8, sample=0.0012, compute=52.7988, clock=52.8,
                cpu=1.31, utime=1.1, stime=0.21, majflt=0, inblock=0,
                nivcsw=2, recompiled=False, queue_depth=2))
        elif i == 21:
            records.append(_record(
                21, 31.1, sample=0.0011, compute=31.0989, clock=31.1,
                cpu=1.28, utime=1.08, stime=0.2, majflt=0, inblock=0,
                nivcsw=2, recompiled=False, queue_depth=2))
        else:
            records.append(_record(i, 1.08, **steady))
    return records


def test_chunk_record_deltas_and_format():
    """Counters become per-chunk deltas; the line names every field."""
    start = manager_jax._Usage(
        clock=1000.0, cpu=10.0, utime=8.0, stime=2.0, majflt=5,
        inblock=100, nivcsw=7)
    end = manager_jax._Usage(
        clock=1052.8, cpu=11.31, utime=9.1, stime=2.21, majflt=5,
        inblock=100, nivcsw=9)
    r = manager_jax._make_chunk_record(
        index=10, first_step=2560, steps=256, wall=52.8, sample=0.0012,
        compute=52.7988, start=start, end=end, recompiled=False,
        queue_depth=2)
    assert r.clock == pytest.approx(52.8)
    assert r.cpu == pytest.approx(1.31)
    assert (r.majflt, r.inblock, r.nivcsw) == (0, 0, 2)
    assert manager_jax._format_chunk_record(r) == (
        'chunk 10 (steps 2560-2815): wall 52.8 s (sample 1.20 ms, '
        'compute 52.8 s), clock 52.8 s, cpu 1.31 s (user 1.10 s, '
        'sys 210 ms), majflt 0, inblock 0, nivcsw 2, recompiled no, '
        'queue depth 2')


def test_chunk_record_format_without_rusage():
    """Windows-like: no rusage, no jit-cache API, no prefetch queue --
    those fields are simply left out."""
    start = manager_jax._Usage(clock=0.0, cpu=0.0)
    end = manager_jax._Usage(clock=31900.0, cpu=0.5)
    r = manager_jax._make_chunk_record(
        index=3, first_step=768, steps=100, wall=31900.0, sample=0.01,
        compute=31899.99, start=start, end=end, recompiled=None,
        queue_depth=None)
    assert r.utime is None and r.majflt is None
    assert manager_jax._format_chunk_record(r) == (
        'chunk 3 (steps 768-867): wall 8 h 52 m (sample 10.0 ms, '
        'compute 8 h 52 m), clock 8 h 52 m, cpu 500 ms')
    # A clock stepped backwards, and an unknown value.
    assert manager_jax._fmt_s(-0.5) == '-500 ms'
    assert manager_jax._fmt_s(None) == '?'


def test_online_outlier_threshold():
    """The last chunk against the eligible ones before it: the same
    factor and floor as the correction, from three chunks on."""
    threshold = manager_jax._online_outlier_threshold
    assert manager_jax._ONLINE_MIN_CHUNKS == 3
    # Two eligible chunks before the last: too few, however long it is.
    assert threshold([5.0, 1.0, 1.0, 1000.0]) is None
    assert threshold([5.0, 1.0, 1.0, 1.0, 1000.0]) == 30.0
    # The floor binds: 25x the median, but under 30 s.
    assert threshold([5.0, 1.0, 1.0, 1.0, 25.0]) is None
    # The factor binds, strictly.
    assert threshold([5.0, 10.0, 10.0, 10.0, 100.0]) is None
    assert threshold([5.0, 10.0, 10.0, 10.0, 101.0]) == 100.0
    # Chunk 0's compile never enters the median...
    assert threshold([1000.0, 1.0, 1.0, 1.0, 900.0]) == 30.0
    # ...and one earlier outlier among three cannot drag it up.
    assert threshold([5.0, 1.0, 900.0, 1.0, 800.0]) == 30.0
    assert threshold([1000.0]) is None
    assert threshold([]) is None


def test_online_detection_fires_exactly_on_outliers(monkeypatch, caplog):
    """Fed chunk by chunk, the incident warns twice -- on chunks 10 and
    21, as each ends -- and on nothing else, chunk 0 included."""
    monkeypatch.setattr(manager_jax, '_system_snapshot', lambda: 'SNAPSHOT')
    records = _incident_records()
    fired = []
    with caplog.at_level(logging.WARNING):
        for i in range(len(records)):
            if manager_jax._warn_if_anomalous(records[:i + 1]):
                fired.append(i)
    assert fired == [10, 21]
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2
    first = warnings[0]
    assert first.startswith(
        'anomalous chunk: chunk 10 (steps 2560-2815): wall 52.8 s '
        '(sample 1.20 ms, compute 52.8 s), clock 52.8 s, cpu 1.31 s')
    assert '-- over the threshold 30.0 s; typical chunk ' in first
    assert 'wall 1.08 s' in first
    assert first.endswith('; system: SNAPSHOT')
    assert '\n' not in first
    assert warnings[1].startswith(
        'anomalous chunk: chunk 21 (steps 5376-5631)')


def test_two_outliers_error_quotes_records():
    """The incident's crash: the traceback alone carries each outlier's
    record and a typical one for scale. Without records the message
    stays the old one-liner."""
    records = _incident_records()
    times = [r.wall for r in records]
    with pytest.raises(RuntimeError) as e:
        manager_jax._correct_chunk_times(times, records)
    lines = str(e.value).split('\n')
    assert len(lines) == 4
    assert lines[0].startswith(
        'anomalous chunk times: 2 chunks over the suspend threshold 30.0 s')
    assert lines[0].endswith('must not be submitted.')
    assert lines[1] == (
        '  outlier chunk 10 (steps 2560-2815): wall 52.8 s (sample 1.20 ms, '
        'compute 52.8 s), clock 52.8 s, cpu 1.31 s (user 1.10 s, '
        'sys 210 ms), majflt 0, inblock 0, nivcsw 2, recompiled no, '
        'queue depth 2')
    assert lines[2].startswith(
        '  outlier chunk 21 (steps 5376-5631): wall 31.1 s')
    assert lines[3].startswith('  typical chunk ')
    assert 'wall 1.08 s' in lines[3]

    with pytest.raises(RuntimeError) as e:
        manager_jax._correct_chunk_times(times)
    assert '\n' not in str(e.value)


def test_suspend_warning_quotes_record(caplog):
    """The single-outlier warning carries the record too, on one line."""
    records = [_record(i, 1.0) for i in range(13)]
    records[6] = _record(6, 31900.0, cpu=0.4)
    with caplog.at_level(logging.WARNING):
        total, n = manager_jax._correct_chunk_times(
            [r.wall for r in records], records)
    assert (total, n) == (pytest.approx(13.0), 1)
    message = caplog.records[0].getMessage()
    assert message.startswith('suspend detected: one chunk took 8 h 52 m')
    assert ('; outlier chunk 6 (steps 1536-1791): wall 8 h 52 m '
            '(sample 0 ns, compute 8 h 52 m), clock 8 h 52 m, '
            'cpu 400 ms; typical chunk ') in message
    assert '\n' not in message


def test_proc_probes_parse(tmp_path):
    """PSI, loadavg and meminfo, read from stand-in files."""
    pressure = tmp_path / 'pressure'
    pressure.mkdir()
    (pressure / 'cpu').write_text(
        'some avg10=1.50 avg60=0.40 avg300=0.10 total=123\n')
    (pressure / 'io').write_text(
        'some avg10=0.00 avg60=0.00 avg300=0.00 total=4\n'
        'full avg10=0.00 avg60=0.00 avg300=0.00 total=2\n')
    # No memory file: that resource is just left out.
    assert manager_jax._probe_psi(str(pressure)) == (
        'psi: cpu some avg10=1.50 avg60=0.40 avg300=0.10 total=123 | '
        'io some avg10=0.00 avg60=0.00 avg300=0.00 total=4')
    assert manager_jax._probe_psi(str(tmp_path / 'missing')) is None

    loadavg = tmp_path / 'loadavg'
    loadavg.write_text('0.52 0.58 0.59 1/345 12345\n')
    assert manager_jax._probe_loadavg(str(loadavg)) == (
        'loadavg 0.52 0.58 0.59 1/345 12345')

    meminfo = tmp_path / 'meminfo'
    meminfo.write_text(
        'MemTotal:       65536000 kB\n'
        'MemFree:            1000 kB\n'
        'MemAvailable:   33554432 kB\n'
        'SwapTotal:       2097152 kB\n'
        'SwapFree:        1048576 kB\n')
    assert manager_jax._probe_meminfo(str(meminfo)) == (
        'MemAvailable 32.00 GiB, SwapFree 1.00 GiB')


def test_gpu_probe(monkeypatch):
    """No driver: nothing. A hang or an error exit: said so, briefly.
    Success: one labeled line, GPUs separated."""
    def raising(exc):
        def run(*args, **kwargs):
            raise exc
        return run

    def returning(code: int, stdout: str, stderr: str = ''):
        def run(args, **kwargs):
            return subprocess.CompletedProcess(args, code, stdout, stderr)
        return run

    monkeypatch.setattr(subprocess, 'run', raising(FileNotFoundError()))
    assert manager_jax._probe_gpu() is None

    monkeypatch.setattr(
        subprocess, 'run', raising(subprocess.TimeoutExpired('nvidia-smi', 5)))
    assert manager_jax._probe_gpu() == (
        'gpu: nvidia-smi timed out after 5.00 s')

    monkeypatch.setattr(subprocess, 'run', returning(
        6, '', 'Field "clocks_throttle_reasons.active" is not a valid field '
        'to query.\n\n'))
    assert manager_jax._probe_gpu() == (
        'gpu: nvidia-smi exit 6 (Field "clocks_throttle_reasons.active" is '
        'not a valid field to query.)')

    monkeypatch.setattr(subprocess, 'run', returning(
        0, '97 %, 2520 MHz, 0x0000000000000000, 64, 351.20 W, 2210 MiB\n'
        '0 %, 210 MHz, 0x0000000000000001, 35, 20.10 W, 1 MiB\n'))
    assert manager_jax._probe_gpu() == (
        'gpu (util, sm clock, throttle reasons, temp C, power, mem): '
        '97 %, 2520 MHz, 0x0000000000000000, 64, 351.20 W, 2210 MiB | '
        '0 %, 210 MHz, 0x0000000000000001, 35, 20.10 W, 1 MiB')


def test_snapshot_swallows_probe_failures(monkeypatch):
    """A probe that raises is dropped; the snapshot itself never
    raises, even with nothing left to report."""
    def boom(*args, **kwargs):
        raise PermissionError('probe exploded')

    monkeypatch.setattr(manager_jax, '_probe_psi', boom)
    monkeypatch.setattr(
        manager_jax, '_probe_loadavg', lambda: 'loadavg 0.10 0.20 0.30 1/2 3')
    monkeypatch.setattr(manager_jax, '_probe_meminfo', boom)
    monkeypatch.setattr(manager_jax, '_probe_gpu', lambda: None)
    assert manager_jax._system_snapshot() == 'loadavg 0.10 0.20 0.30 1/2 3'

    monkeypatch.setattr(manager_jax, '_probe_loadavg', boom)
    assert manager_jax._system_snapshot() == 'nothing to report'

    # The real GPU probe, failing in an unexpected way, goes the same way.
    monkeypatch.undo()
    for name in ('_probe_psi', '_probe_loadavg', '_probe_meminfo'):
        monkeypatch.setattr(manager_jax, name, boom)
    monkeypatch.setattr(subprocess, 'run', boom)
    with pytest.raises(PermissionError):
        manager_jax._probe_gpu()
    assert manager_jax._system_snapshot() == 'nothing to report'


def test_counter_probes_never_raise(monkeypatch):
    """getrusage failing or missing, a jax without the cache API, a
    sampler without (or with a broken) prefetch queue: None, no error."""
    def boom(*args, **kwargs):
        raise OSError('no counters today')

    monkeypatch.setattr(
        manager_jax, 'resource', SimpleNamespace(RUSAGE_SELF=0, getrusage=boom))
    usage = manager_jax._read_usage()
    assert usage.clock > 0
    assert (usage.utime, usage.stime, usage.majflt) == (None, None, None)

    fake_rusage = SimpleNamespace(
        ru_utime=1.5, ru_stime=0.25, ru_majflt=3, ru_inblock=8, ru_nivcsw=2)
    monkeypatch.setattr(manager_jax, 'resource', SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda who: fake_rusage))
    usage = manager_jax._read_usage()
    assert (usage.utime, usage.stime, usage.majflt, usage.inblock,
            usage.nivcsw) == (1.5, 0.25, 3, 8, 2)

    monkeypatch.setattr(manager_jax, 'resource', None)  # Windows
    assert manager_jax._read_usage().utime is None

    assert manager_jax._jit_cache_size(object()) is None
    assert manager_jax._jit_cache_size(jax.jit(lambda x: x)) == 0

    assert manager_jax._queue_depth(_make_dataset(), 8, 4, 'bytes') is None

    class _BrokenWrapper(DataSetWrapper):
        def __init__(self):
            pass  # no worker threads

        def queue_depth(self, ntokens, batch, tokenset_name):
            raise RuntimeError('queue exploded')

    assert manager_jax._queue_depth(_BrokenWrapper(), 8, 4, 'bytes') is None


def test_diagnostics_failure_never_masks_anything(monkeypatch, caplog):
    """A broken formatter costs the details, never the training run --
    and never the end-of-run verdict."""
    def boom(record):
        raise ValueError('bad record')

    monkeypatch.setattr(manager_jax, '_format_chunk_record', boom)
    monkeypatch.setattr(manager_jax, '_system_snapshot', lambda: 'SNAPSHOT')
    records = [_record(i, 1.0) for i in range(5)] + [_record(5, 100.0)]
    with caplog.at_level(logging.WARNING):
        assert manager_jax._warn_if_anomalous(records)
    assert 'anomalous-chunk diagnostics failed (ignored)' in caplog.text

    records = _incident_records()
    with pytest.raises(RuntimeError, match='anomalous chunk times') as e:
        manager_jax._correct_chunk_times([r.wall for r in records], records)
    assert ("(chunk details unavailable: ValueError('bad record'))"
            in str(e.value))


def test_train_warns_on_anomalous_chunk(monkeypatch, caplog):
    """Wiring: an outlying chunk is reported with its record the moment
    it ends. 1 s chunks (250 ms waiting for data) except chunk 5, which
    takes 100 s; 7 chunks are too few for the correction, not for the
    online check."""
    ticks = [0.0]
    t = 0.0
    for i in range(7):
        duration = 100.0 if i == 5 else 1.0
        ticks += [t + 0.25, t + duration]
        t += duration
    monkeypatch.setattr(manager_jax, 'perf_counter', iter(ticks).__next__)
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 1)
    monkeypatch.setattr(manager_jax, '_system_snapshot', lambda: 'SNAPSHOT')
    manager = create_manager(
        'jax', conf=_make_conf(steps=7), system='test',
        dataset=_make_dataset(),
        test_sample_len=64, test_batch=2,
        verbose=False,
    )
    with caplog.at_level(logging.WARNING):
        total_time, _ = manager.train(steps=7, time_limit=None)
    assert total_time == pytest.approx(106.0)
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].startswith(
        'anomalous chunk: chunk 5 (steps 5-5): wall 1 m 40 s '
        '(sample 250 ms, compute 1 m 40 s), clock ')
    assert 'recompiled no' in warnings[0]
    assert warnings[0].endswith('system: SNAPSHOT')


def test_continue_prefix():
    manager = create_manager(
        'jax', conf=_make_conf(steps=2), system='test',
        dataset=_make_dataset(),
        test_sample_len=64, test_batch=2,
        verbose=False,
    )
    manager.train_and_eval(steps=2, time_limit=None)
    out = manager.continue_prefix('hi', length=8, temperature=1.0)
    assert isinstance(out, bytes)
    assert len(out) > 0


class _TaggedDataSet(DataSet):
    """A single-byte corpus that records each sampling call, so a test
    can tell which corpus a chunk was drawn from."""

    def __init__(self, byte: bytes, log: list):
        super().__init__(data=byte * 4096)
        self.byte = byte
        self.log = log

    def sample_tokens(self, ntokens: int, batch: int, tokenset_name: str):
        data = super().sample_tokens(ntokens, batch, tokenset_name)
        self.log.append((self.byte, int(data[0, 0])))
        return data


def _make_bytes_conf(steps: int):
    """A tiny model over the 'bytes' tokenset, so sampled tokens are
    the corpus bytes themselves."""
    return Configuration(
        parse_model2('bytes.emb.2|dense.2.tanh', precision=Precision.FP32),
        lr=0.01, length=8, batch=2, steps=steps, decay=1.0,
    )


def _make_switch_manager(steps: int, dataset):
    return create_manager(
        'jax', conf=_make_bytes_conf(steps), system='test',
        dataset=dataset, test_sample_len=32, test_batch=2, verbose=False,
    )


def _corpora() -> tuple[list, _TaggedDataSet, _TaggedDataSet]:
    log = []
    return log, _TaggedDataSet(b'a', log), _TaggedDataSet(b'b', log)


def test_data_switch_at_chunk_boundary(monkeypatch):
    """Everything up to the switch step comes from the first corpus,
    everything after from the second."""
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 4)
    log, corpus_a, corpus_b = _corpora()
    manager = _make_switch_manager(16, corpus_a)
    manager.set_data_switch(8, corpus_b)
    manager.train(steps=16, time_limit=None)

    # One sample call per chunk of 4 steps: 2 chunks each side.
    assert [byte for byte, _ in log] == [b'a', b'a', b'b', b'b']
    # ...and the tokens really are that corpus's bytes.
    assert all(token == ord(byte) for byte, token in log)
    assert manager.dataset is corpus_b
    assert manager.run.steps == 16


def test_data_switch_rounds_to_nearest_chunk(monkeypatch):
    """A switch step inside a chunk snaps to the nearest boundary --
    a chunk is one uninterruptible scan."""
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 4)
    log, corpus_a, corpus_b = _corpora()
    manager = _make_switch_manager(16, corpus_a)
    manager.set_data_switch(9, corpus_b)  # -> 8
    manager.train(steps=16, time_limit=None)
    assert [byte for byte, _ in log] == [b'a', b'a', b'b', b'b']

    log, corpus_a, corpus_b = _corpora()
    manager = _make_switch_manager(16, corpus_a)
    manager.set_data_switch(11, corpus_b)  # -> 12
    manager.train(steps=16, time_limit=None)
    assert [byte for byte, _ in log] == [b'a', b'a', b'a', b'b']


def test_data_switch_alignment_clamped_to_one_chunk():
    """Rounding must never produce step 0 (which would put the whole
    run on the post-switch corpus)."""
    log, corpus_a, corpus_b = _corpora()
    manager = _make_switch_manager(16, corpus_a)
    manager.set_data_switch(1, corpus_b)
    manager._align_data_switch(256)
    assert manager._data_switch == (256, corpus_b)


def test_data_switch_per_step_loop(monkeypatch):
    """--no-scan takes the per-step loop, which switches on the exact
    step (no chunk to align to)."""
    log, corpus_a, corpus_b = _corpora()
    manager = _make_switch_manager(4, corpus_a)
    manager.scan_train = False
    manager.set_data_switch(2, corpus_b)
    manager.train(steps=4, time_limit=None)
    assert [byte for byte, _ in log] == [b'a', b'a', b'b', b'b']


def test_data_switch_retires_old_wrapper(monkeypatch):
    """The pre-switch sampler's worker threads are torn down once
    nothing will ask it for another batch."""
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 4)
    log, corpus_a, corpus_b = _corpora()
    wrapper_a = DataSetWrapper(corpus_a, num_workers=1)
    wrapper_b = DataSetWrapper(corpus_b, num_workers=1)
    try:
        manager = _make_switch_manager(8, wrapper_a)
        manager.set_data_switch(4, wrapper_b)
        manager.train(steps=8, time_limit=None)
        assert wrapper_a.joined
        assert not wrapper_b.joined
        assert manager.dataset is wrapper_b
    finally:
        wrapper_a.join()
        wrapper_b.join()


def test_no_data_switch_keeps_dataset(monkeypatch):
    monkeypatch.setattr(manager_jax, '_CHUNK_SIZE', 4)
    log, corpus_a, _ = _corpora()
    manager = _make_switch_manager(8, corpus_a)
    manager.train(steps=8, time_limit=None)
    assert [byte for byte, _ in log] == [b'a', b'a']
    assert manager.dataset is corpus_a
