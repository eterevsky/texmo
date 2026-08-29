import os
import subprocess
import sys

import pytest

from texmo import xla_flags

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(**overrides) -> dict[str, str]:
    env = {}
    env.update(overrides)
    return env


def test_apply_adds_the_flag():
    env = _env()
    out = xla_flags.apply(env)
    assert '--xla_gpu_cublas_fallback=false' in out
    assert env['XLA_FLAGS'] == out


def test_apply_is_idempotent():
    env = _env()
    xla_flags.apply(env)
    once = env['XLA_FLAGS']
    xla_flags.apply(env)
    assert env['XLA_FLAGS'] == once


def test_apply_keeps_existing_flags():
    env = _env(XLA_FLAGS='--xla_gpu_autotune_level=2')
    out = xla_flags.apply(env)
    assert '--xla_gpu_autotune_level=2' in out
    assert '--xla_gpu_cublas_fallback=false' in out


def test_apply_does_not_override_the_user():
    env = _env(XLA_FLAGS='--xla_gpu_cublas_fallback=true')
    out = xla_flags.apply(env)
    assert out == '--xla_gpu_cublas_fallback=true'


def test_apply_can_be_opted_out():
    env = _env(TEXMO_XLA_WORKAROUNDS='0')
    assert xla_flags.apply(env) == ''
    assert 'XLA_FLAGS' not in env


def test_importing_texmo_applies_it():
    # `texmo/__init__.py` ran when this module was collected.
    assert '--xla_gpu_cublas_fallback=false' in os.environ['XLA_FLAGS']


# --- the miscompile itself ------------------------------------------
#
# conftest.py pins the whole session to CPU, and the bug is an XLA:GPU
# codegen bug, so this one runs in a subprocess with the platform left
# to JAX. It skips on a machine with no GPU.

_PROBE = '''
import sys
import numpy as np
import texmo  # noqa: F401  -- installs the XLA_FLAGS workaround
import jax
import jax.numpy as jnp

if jax.devices()[0].platform != 'gpu':
    print('NOGPU')
    sys.exit(0)

# A square dot whose output is consumed transposed (layout {0,1}),
# plus a bias: the shape XLA:GPU fused into a wrong cuBLASLt BIAS
# epilogue.  See texmo/xla_flags.py.
M = N = 16
K = 32
rng = np.random.default_rng(0)
x = jnp.asarray(rng.standard_normal((M, K)), dtype=jnp.float32)
w = jnp.asarray(rng.standard_normal((N, K)), dtype=jnp.float32)
b = jnp.asarray(10.0 * rng.standard_normal(N), dtype=jnp.float32)
got = np.asarray(
    jax.jit(lambda x, w, b: jnp.transpose(x @ w.T + b))(x, w, b),
    dtype=np.float64)
ref = (np.asarray(x, np.float64) @ np.asarray(w, np.float64).T
       + np.asarray(b, np.float64)).T
print('MAXERR', np.abs(got - ref).max())
'''


def _run_probe() -> str:
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    env.pop('XLA_FLAGS', None)
    env['PYTHONPATH'] = _ROOT
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    r = subprocess.run(
        [sys.executable, '-c', _PROBE], cwd=_ROOT, env=env,
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_gpu_dot_with_bias_and_transposed_output():
    out = _run_probe()
    if 'NOGPU' in out:
        pytest.skip('no GPU on this machine')
    err = float(out.split('MAXERR')[1].split()[0])
    # Without the workaround this is ~3.6e+01 (the bias lands on the
    # wrong axis); the fp32/TF32 GEMM noise floor is ~1e-2.
    assert err < 0.1, f'dot+bias with a transposed output is wrong: {err}'
