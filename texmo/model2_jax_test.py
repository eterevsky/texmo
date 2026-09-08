"""Model2Jax runtime tests, mostly about the recurrent path.

`forward_recurrent` pads its batch away from the model's layer widths
so the vmapped dots are never square -- the shape XLA:GPU miscompiles
into a wrong-axis cuBLASLt bias epilogue (docs/findings.md, "XLA:GPU
adds a fused bias on the wrong axis (2026-08-29)"). The CPU tests here
cover the padding arithmetic and check that the dummy rows change
nothing; the two GPU tests are the regressions for the miscompile and
for the flag that used to work around it. Last comes the machine
diagnostic for the miscompile itself, behind `TEXMO_XLA_DIAG=1`.
"""
import functools
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from texmo.precision import Precision
from texmo.spec_parser import parse_model2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A split.cat model: the merge is what gives some of the per-step dots
# a {0,1} output layout, so this is the shape the bug needs.
_SPEC = 'bits.1+bp|dense.4.gelu-split.cat(dense.4.tanh, pass)-gru.4'


def _build(spec=_SPEC, seed=0):
    md = parse_model2(spec, Precision.FP32)
    model = md.build_jax()
    weights = model.init_weights(jax.random.PRNGKey(seed))
    return md, model, weights


def _batch(model, batch_size, length=7, seed=3):
    return jax.random.randint(
        jax.random.PRNGKey(seed), (batch_size, length),
        0, model.ntokens).astype(jnp.int32)


# --- the width set --------------------------------------------------


def test_layer_widths_walks_the_whole_tree():
    md = parse_model2(_SPEC, Precision.FP32)
    widths = md.layer_widths()
    # dense.4 / gru.4 outputs, the cat merge (4 + 4), the codec's own
    # width, the vocabulary and the head's logit count.
    assert 4 in widths
    assert 8 in widths
    assert md.codec.size in widths
    assert md.ntokens in widths
    assert md.output.input_size in widths
    assert md.output.size in widths


def test_layer_widths_of_a_real_manifest_spec():
    md = parse_model2(
        'tokens.32.hexbpe.oh|rnn.32.gelu-split.cat(split.add('
        'dense.16.gelu-rglru.16-dense.32.gelu-dense.16, pass)-mgru.16, '
        'pass)-dense.32.gelu-rmsnorm', Precision.FP32)
    # The widths that made `texmo.py eval --chunk 16 / 32` wrong.
    assert {16, 32} <= md.layer_widths()


# --- the padded batch size ------------------------------------------


def test_recurrent_batch_size_skips_layer_widths():
    _, model, _ = _build()
    widths = model._layer_widths
    for n in range(1, 40):
        padded = model._recurrent_batch_size(n)
        assert padded >= n
        assert padded not in widths
        # It is the *first* free size at or above n.
        assert all(k in widths for k in range(n, padded))


def test_recurrent_batch_size_is_identity_off_the_widths():
    _, model, _ = _build()
    free = max(model._layer_widths) + 1
    assert model._recurrent_batch_size(free) == free


# --- the padding rows do not change anything ------------------------


def test_padding_does_not_change_the_logits():
    md, model, weights = _build()
    batch_size = 4
    assert batch_size in md.layer_widths()  # the interesting case
    batch = _batch(model, batch_size)

    padded = model.forward_recurrent(weights, batch)
    # `_forward_recurrent` is the same computation without the pad
    # rows -- correct everywhere except on an affected GPU.
    unpadded = model._forward_recurrent(weights, batch)
    assert padded.shape == unpadded.shape
    assert padded.shape[:2] == (batch_size, batch.shape[1])
    np.testing.assert_allclose(
        np.asarray(padded), np.asarray(unpadded), atol=1e-6)


def test_padded_recurrent_loss_matches_the_parallel_loss():
    md, model, weights = _build()
    batch_size = 4
    assert model._recurrent_batch_size(batch_size) != batch_size
    batch = _batch(model, batch_size)
    lengths = jnp.asarray([7, 5, 3, 1], dtype=jnp.int32)

    rec = float(model.loss_batch_masked_recurrent(weights, batch, lengths))
    par = float(model.loss_batch_masked(weights, batch, lengths))
    assert rec == pytest.approx(par, rel=1e-5)


def test_masked_loss_ignores_the_pad_rows():
    """The mask is built from `lengths`, which is sized to the real
    batch -- so the pad rows must be gone before the reduction."""
    _, model, weights = _build()
    batch_size = 4
    batch = _batch(model, batch_size)
    lengths = jnp.asarray([7, 7, 7, 7], dtype=jnp.int32)

    whole = float(model.loss_batch_masked_recurrent(weights, batch, lengths))
    rows = sum(
        float(model.loss_batch_masked_recurrent(
            weights, batch[i:i + 1], lengths[i:i + 1]))
        for i in range(batch_size))
    assert whole == pytest.approx(rows, rel=1e-5)


def test_padding_accepts_numpy_batches():
    """`ManagerJax.eval` hands the recurrent loss raw numpy arrays."""
    _, model, weights = _build()
    batch = np.asarray(_batch(model, 4), dtype=np.int32)
    lengths = np.asarray([7, 6, 5, 4], dtype=np.int32)
    rec = float(model.loss_batch_masked_recurrent(weights, batch, lengths))
    par = float(model.loss_batch_masked(weights, batch, lengths))
    assert rec == pytest.approx(par, rel=1e-5)


def test_forward_recurrent_matches_forward_at_a_colliding_batch():
    _, model, weights = _build()
    batch = _batch(model, 4)
    np.testing.assert_allclose(
        np.asarray(model.forward(weights, batch)),
        np.asarray(model.forward_recurrent(weights, batch)),
        atol=1e-5)


# --- GPU regressions ------------------------------------------------
#
# conftest.py pins the session to CPU and both bugs are XLA:GPU codegen
# bugs, so these run in subprocesses with the platform left to JAX and
# with any inherited `XLA_FLAGS` dropped -- stock XLA, which is what
# the whole fleet runs since 2026-09-08.

_GPU_PROBE = 'import jax; print("PLATFORM", jax.devices()[0].platform)'


def _probe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    env.pop('XLA_FLAGS', None)
    env['PYTHONPATH'] = _ROOT
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    return env


def _run(argv: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, cwd=_ROOT, env=_probe_env(),
        capture_output=True, text=True, timeout=timeout)


@functools.cache
def _has_gpu() -> bool:
    r = _run([sys.executable, '-c', _GPU_PROBE], timeout=600)
    return r.returncode == 0 and 'PLATFORM gpu' in r.stdout


def _require(*paths: str):
    if not _has_gpu():
        pytest.skip('no GPU on this machine')
    for p in paths:
        if not os.path.exists(os.path.join(_ROOT, p)):
            pytest.skip(f'{p} not present on this machine')


def test_gpu_trains_a_triton_uncompilable_gemm():
    """Regression for `--xla_gpu_cublas_fallback=false`.

    That flag was injected fleet-wide at `import texmo` from
    2026-08-29 until `texmo/xla_flags.py` was deleted on 2026-09-08.
    It removed XLA's cuBLAS fallback for GEMM fusions Triton cannot
    compile, so this conf died with `RET_CHECK ...
    !candidates.empty()` instead of training: the backward dot of the
    suffix stack is f32[8,258] x f32[258,128]. It must train under
    stock flags.
    """
    _require('data/soda_s3.txt', 'tokens/tokens.64.hexbpe.json')
    r = _run([
        sys.executable, 'texmo.py', 'train',
        '-s', 'tokens.64.hexbpe.oh|suffix.2-dense.8.silu-'
              'suffix.2-dense.16.silu',
        '-p', 'fp32', '-b', '2', '--lr', '1/16', '--cosine',
        '-l', '128', '--steps', '8', '-d', 'data/soda_s3.txt',
        '--no-graph',
    ])
    assert 'RET_CHECK' not in r.stderr, r.stderr[-2000:]
    assert r.returncode == 0, r.stderr[-2000:]


_LOSS_PROBE = '''
import random
import sys

import jax
import jax.numpy as jnp

from texmo.dataset import DataSet
from texmo.model_store import load_model
from texmo.precision import Precision
from texmo.tokens import set_tokens_dir

if jax.devices()[0].platform != 'gpu':
    print('NOGPU')
    sys.exit(0)

set_tokens_dir('tokens')
md, weights = load_model('models/hb32-8k-s3.json', Precision.FP32)
model = md.build_jax()
dataset = DataSet(path='data/soda_s3.txt')

for batch_size in (16, 32):
    random.seed(f'model2-jax-test:{batch_size}')
    batch, lengths = dataset.sample_bytes(
        nbytes=256, batch=batch_size, tokenset_name=md.codec.tokens_name)
    batch = jnp.asarray(batch)
    lengths = jnp.asarray(lengths)
    rec = float(model.loss_batch_masked_recurrent(weights, batch, lengths))
    par = float(model.loss_batch_masked(weights, batch, lengths))
    print('LOSS', batch_size, repr(rec), repr(par))
'''


def test_gpu_recurrent_loss_matches_parallel_at_layer_widths():
    """Regression for the wrong-axis bias epilogue.

    `models/hb32-8k-s3.json` has 16- and 32-wide layers, so those batch
    sizes used to make the recurrent loss grossly wrong (10.84 vs 1.36
    b/B at 32, 2.08 vs 1.29 at 16). The batch padding in
    `forward_recurrent` is what keeps them apart now; this runs with
    stock flags, so it fails if that padding regresses.
    """
    _require('models/hb32-8k-s3.json', 'data/soda_s3.txt',
             'tokens/tokens.32.hexbpe.json')
    r = _run([sys.executable, '-c', _LOSS_PROBE])
    assert r.returncode == 0, r.stderr[-2000:]
    if 'NOGPU' in r.stdout:
        pytest.skip('no GPU on this machine')
    rows = [l.split() for l in r.stdout.splitlines() if l.startswith('LOSS')]
    assert len(rows) == 2, r.stdout
    for _, batch_size, rec, par in rows:
        rec, par = float(rec), float(par)
        assert rec == pytest.approx(par, rel=1e-4), (
            f'batch {batch_size}: recurrent {rec} vs parallel {par}')


# --- the miscompile itself, as a machine diagnostic -----------------
#
# Not a regression test: it asks whether THIS GPU still fuses a bias
# epilogue on the wrong axis. Nothing in texmo suppresses the bug any
# more -- `texmo/xla_flags.py` and its `--xla_gpu_cublas_fallback=false`
# were deleted on 2026-09-08, because the flag also removed XLA's
# cuBLAS fallback for Triton-uncompilable GEMMs and killed workers
# outright (the test above). So on an affected machine -- this one --
# this test FAILS BY DESIGN, reporting a maxerr the size of the bias
# instead of the GEMM noise floor.
#
# That failure is the point of keeping it: it is the reason
# `forward_recurrent` pads the batch off the layer widths, which is
# what actually protects texmo and is what every test above covers.
# The bug is upstream of us; a standalone repro lives in the winjax
# fork's docs (`docs/repro_cublaslt_bias.py`).
#
# It therefore only runs when asked for, so the normal suite never
# sees it:
#
#     TEXMO_XLA_DIAG=1 uv run pytest texmo/model2_jax_test.py -k diag

_BIAS_DIAG_PROBE = '''
import sys

import numpy as np
import jax
import jax.numpy as jnp

if jax.devices()[0].platform != 'gpu':
    print('NOGPU')
    sys.exit(0)

# A square dot whose output is consumed transposed (layout {0,1}),
# plus a bias: the shape XLA:GPU fuses into a cuBLASLt BIAS epilogue
# that broadcasts the bias along the wrong axis.
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


@pytest.mark.skipif(
    os.environ.get('TEXMO_XLA_DIAG', '') not in ('1', 'true', 'yes', 'on'),
    reason='machine diagnostic; set TEXMO_XLA_DIAG=1 to run')
def test_gpu_bias_epilogue_diag():
    """Is this GPU affected by the wrong-axis bias epilogue at all?

    FAILS BY DESIGN on an affected machine -- see the comment above.
    """
    r = _run([sys.executable, '-c', _BIAS_DIAG_PROBE], timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    if 'NOGPU' in r.stdout:
        pytest.skip('no GPU on this machine')
    err = float(r.stdout.split('MAXERR')[1].split()[0])
    # On an affected machine this is ~3e+01 -- 31.0 measured on the
    # sm_120 machine that found the bug, i.e. the scale of the bias
    # itself; the fp32/TF32 GEMM noise floor is ~1e-2.
    assert err < 0.1, (
        f'this GPU miscompiles dot+bias with a transposed output: '
        f'maxerr {err}')
