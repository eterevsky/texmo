"""Light integration tests for ManagerTorch and ManagerJax.

Uses a minimal 'bits.1+bp|' model (no hidden layers, binary tokens)
trained on in-memory random bytes. Verifies the end-to-end pipeline
doesn't blow up, not the quality of training.
"""

import json
import math

import jax
import pytest

from texmo import manager_jax
from texmo.configuration import Configuration
from texmo.dataset import DataSet
from texmo.manager import create_manager
from texmo.model import build_model_def
from texmo.precision import Precision
from texmo.spec_parser import parse_model2


def _make_dataset():
    return DataSet(data=b'hello world ' * 200)


def _make_conf(steps: int = 3):
    return Configuration(
        build_model_def('bits.1+bp|', precision=Precision.FP32),
        lr=0.01,
        length=32,
        batch=8,
        steps=steps,
        decay=1.0,
    )


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_train_and_eval(backend):
    manager = create_manager(
        backend, conf=_make_conf(), system='test',
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


@pytest.mark.parametrize('backend', ['torch', 'jax'])
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
    'bits.1+bp|skip.1.add-dense.4.gelu-dense.8.gelu',
    'bits.1+bp|skip.2.cat-dense.4.gelu-dense.4.gelu-dense.8.gelu',
    'bits.1+bp|skip.2.cat-suffix.4-dense.4.tanh',
])
def test_train_various_specs(backend, spec):
    """Consistency check: model builds and trains without shape errors."""
    conf = Configuration(
        build_model_def(spec, precision=Precision.FP32),
        lr=0.01, length=32, batch=4, steps=2, decay=1.0,
    )
    manager = create_manager(
        backend, conf=conf, system='test',
        dataset=_make_dataset(),
        test_sample_len=32, test_batch=2,
        verbose=False,
    )
    run, _ = manager.train_and_eval(steps=2, time_limit=None)
    assert run.steps == 2


@pytest.mark.parametrize('spec', [
    # Plain specs the parser handles unchanged.
    'bits.1+bp|dense.4.gelu',
    'bits.1+bp|gru.4-dense.2.tanh',
    # Skip specs that get translated to split.op(...) form at parse
    # time -- the manager should see no SkipDef.
    'bits.1+bp|skip.1.add-dense.4.gelu-dense.4.gelu',
    'bits.1+bp|skip.2.cat-dense.4.gelu-dense.4.gelu-dense.4.gelu',
    # Hand-authored split spec.
    'bits.1+bp|dense.4.gelu-split.mul(dense.4.gelu, pass)-dense.4.gelu',
])
def test_train_via_parse_model2_jax(spec):
    """End-to-end smoke for the Model2 train path -- the same path
    cli/train.py now takes. Exercises Manager init, train_step,
    eval (which triggers the FP32 rebuild dispatch in
    manager_jax)."""
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
    'bits.1+bp|skip.1.add-dense.4.gelu-dense.8.gelu',
    'bits.1+bp|skip.2.cat-dense.4.gelu-dense.4.gelu-dense.8.gelu',
    'bits.1+bp|skip.2.cat-suffix.4-dense.4.tanh',
    # Two overlapping skips with mixed add and cat landing at the same
    # merge point (just before the output dense).
    'bits.1+bp|skip.3.add-dense.4.gelu-skip.1.cat-dense.4.gelu-dense.8.gelu',
])
def test_jax_num_weights_matches_def(spec):
    """ModelDef.num_weights should equal the total element count of the
    JAX weight pytree produced by the manager.
    """
    conf = Configuration(
        build_model_def(spec, precision=Precision.FP32),
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


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_continue_prefix(backend):
    manager = create_manager(
        backend, conf=_make_conf(steps=2), system='test',
        dataset=_make_dataset(),
        test_sample_len=64, test_batch=2,
        verbose=False,
    )
    manager.train_and_eval(steps=2, time_limit=None)
    out = manager.continue_prefix('hi', length=8, temperature=1.0)
    assert isinstance(out, bytes)
    assert len(out) > 0
