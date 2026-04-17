"""Light integration tests for ManagerTorch and ManagerJax.

Uses a minimal 'bits.1+bp|' model (no hidden layers, binary tokens)
trained on in-memory random bytes. Verifies the end-to-end pipeline
doesn't blow up, not the quality of training.
"""

import pytest

from texmo.configuration import Configuration
from texmo.dataset import DataSet
from texmo.manager import create_manager
from texmo.model import build_model_def
from texmo.precision import Precision


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
