"""XLA_FLAGS mitigations, applied before the backend initializes.

XLA reads `XLA_FLAGS` once, when the PJRT client is created (the first
device use), so anything here only has to run before the first array
lands on a device -- importing `texmo` is early enough for every entry
point (texmo.py, scripts/, the search client, pytest).

Currently one entry:

`--xla_gpu_cublas_fallback=false`
    Works around an XLA:GPU miscompile (seen on jax/jaxlib 0.11.0 with
    the winjax CUDA 13 plugin, sm_120): when layout assignment gives a
    `dot` the output layout `{0,1}` -- i.e. its result is consumed
    transposed -- and a bias add follows, the GEMM autotuner may pick
    the cuBLASLt backend and fuse the add as an `epilogue:"BIAS"`. The
    epilogue then adds the bias along the *physical* rows, so the
    result is `out[m,n] += b[m]` instead of `out[m,n] += b[n]`. When
    M == N that is shape-compatible, so it is silent; XLA declines the
    fusion when M != N, which is why the corruption only appears for a
    square dot.

    In texmo that pattern is reached from
    `Model2Jax.forward_recurrent`: `jax.vmap` of `step` over the batch
    inside a `lax.scan` gives every layer a `[batch, width]` dot plus
    bias, and a `split.cat` merge downstream makes some of those dots
    feature-major ({0,1}). Whenever `batch == width` the eval loss came
    out grossly wrong (10.8 b/B instead of 1.36 for
    `models/hb32-8k-s3.json` at batch 32) while the parallel forward
    and every other batch size were correct.

    The flag keeps autotuning on but drops cuBLAS(Lt) from the GEMM
    fusion candidates, so the dot is emitted by the Triton path, which
    broadcasts the bias correctly. Measured on this repo's shapes it is
    perf-neutral to favorable (fp32 [131072,256]x[256,256] with a bias
    ran ~2x faster; a bare bf16 2048^3 matmul ran slower, but texmo has
    no such shape). Set `TEXMO_XLA_WORKAROUNDS=0` to skip it, or set
    `xla_gpu_cublas_fallback` yourself in `XLA_FLAGS` to win.

`xla_flags_test.py` runs that pattern in a subprocess on machines that
have a GPU. To find out whether another machine needs the workaround
at all, run the same test with the workaround off -- it fails exactly
on an affected machine and skips where there is no GPU:

    TEXMO_XLA_WORKAROUNDS=0 uv run pytest texmo/xla_flags_test.py -k gpu
"""

import os

FLAGS = ('--xla_gpu_cublas_fallback=false',)

_OPT_OUT = 'TEXMO_XLA_WORKAROUNDS'


def apply(env: dict[str, str] | os._Environ = os.environ) -> str:
    """Append the mitigation flags to `XLA_FLAGS` in `env`.

    Idempotent, and never overrides a value the caller already set:
    a flag whose name already appears in `XLA_FLAGS` is left alone.
    Returns the resulting `XLA_FLAGS` value.
    """
    flags = env.get('XLA_FLAGS', '')
    if env.get(_OPT_OUT, '1') in ('0', 'false', 'no'):
        return flags
    for flag in FLAGS:
        name = flag.split('=', 1)[0]
        if name in flags:
            continue
        flags = f'{flags} {flag}'.strip()
    env['XLA_FLAGS'] = flags
    return flags
