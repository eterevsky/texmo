"""Light integration tests for ManagerJax.

Uses a minimal 'bits.1+bp|' model (no hidden layers, binary tokens)
trained on in-memory random bytes. Verifies the end-to-end pipeline
doesn't blow up, not the quality of training.
"""

import json
import logging
import math

import jax
import pytest

from texmo import manager_jax
from texmo.configuration import Configuration
from texmo.dataset import DataSet
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

    def fake_correct(chunk_times):
        seen.append(list(chunk_times))
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
    # The corrected total is what gets reported -- and eval, which runs
    # after train(), is not folded into it.
    assert run.train_time == 42.0


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
