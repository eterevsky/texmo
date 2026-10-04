import json
import logging
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import resource  # Unix only
except ImportError:  # Windows: the rusage fields of a chunk stay None
    resource = None

from . import generate
from .common import ttoa3
from .configuration import Configuration
from .dataset import DataSetWrapper
from .layer_jax import LayerWeights
from .manager import Manager
from .model2_jax import Model2Jax
from .precision import Precision
from .predict import LossTrend
from .run import Run
from .spec_parser import parse_model2

# Number of training steps batched into a single JIT'd `lax.scan`.
# Removes the per-step Python overhead and the host-sync caused by
# `float(loss)` in the loop -- both of which dominate for tiny models
# on accelerators. Caps the data tensor size so it stays comfortable
# even for the widest configs (256 steps * 64 batch * 8192 length is
# ~128 MB of int8).
_CHUNK_SIZE = 256

# Minimum wall-clock seconds between progress logs in quiet mode.
# Verbose mode still prints every chunk.
_QUIET_PROGRESS_INTERVAL = 30.0

# Suspend detection over the per-chunk wall times. A machine that
# sleeps mid-run resumes correctly (everything here is step-based, so
# the loss stays valid) but the wall clock keeps running, and the whole
# gap lands in a single inter-chunk interval -- an 8-hour "training
# time" for a 5-minute run. The timing model fits L2 per (system,
# precision), so one such point outvotes thousands of honest ones.
#
# Chunk 0 is excluded from both the median and the outlier test: it
# carries the JIT compile of the scan, which is legitimately 10x+ the
# steady-state chunk on fast models. Its measured time still counts
# toward the total.
#
# Nothing triggers on a machine that is merely slow: an SBC's chunks
# are uniformly slow, so the median is slow too and no chunk stands
# 10x above it. The correction only fires on a gap that is anomalous
# *relative to the same run*, which is why it is safe for a fleet
# spanning a Pi and a GPU box.
#
# The factor alone is not enough: chunk medians across the fleet run
# from ~90 ms (an SBC on a tiny conf) to tens of seconds, and at a
# sub-second median 10x is a second or two -- the size of an ordinary
# input-pipeline stall, not of a nap. Two workers crashed on exactly
# that. A Mac mini with a 217 ms median saw 53 chunks at 2.2-2.7 s,
# one in every 9 or 10: its prefetch workers deliver in lockstep
# bursts, training drains a burst at 217 ms a chunk and then waits
# ~2.2 s for the next one (that conf is input-bound there, ~32% of
# wall time spent waiting for data). An SBC with a 91 ms median saw
# its first 3 chunks at ~1.4 s while the tokenizer-heavy hexbpe-64
# sampler filled the queue. So a suspend must also clear an absolute
# floor: the incident this detector was built for was a single 8h52m
# gap between ~30 s chunks, and any real suspend is minutes to hours.
# Chunks over the factor but under the floor are simply not outliers
# -- no correction, no crash, the time is reported as measured.
_SUSPEND_MIN_CHUNKS = 8
_SUSPEND_OUTLIER_FACTOR = 10.0
_SUSPEND_MIN_GAP_S = 30.0

# Diagnostics for the chunks the detector above flags. An outlier on
# its own is just a number at the end of the run -- "52.8 s and 31.1 s
# against a 1.08 s median" on a dedicated GPU box says nothing about
# why. So every chunk keeps a few cheap counters (`_ChunkRecord`: a
# handful of clock reads and one getrusage per chunk), the end-of-run
# warning and crash quote the outliers' records, and a chunk over the
# threshold is also reported the moment it ends, together with a
# system snapshot (`_system_snapshot`: PSI, load, memory, nvidia-smi)
# taken while the cause may still be visible. Nothing is snapshotted
# in the steady state: a chunk under the absolute floor costs one
# comparison.
#
# What the counters separate:
# - wall (perf_counter) vs clock (time.time()): equal when the process
#   was awake for the whole chunk; clock far ahead means the machine
#   slept (or the system clock was stepped). Linux's perf_counter is
#   CLOCK_MONOTONIC, which stops during suspend, so there a long chunk
#   is always awake time.
# - sample vs compute: waiting on the prefetch queue (queue depth at
#   chunk start; 0 means the chunk had to wait) vs the jitted scan and
#   its host sync.
# - process CPU time (all threads) against wall, plus, where rusage
#   exists, user/sys split, major page faults and block reads (paging
#   or disk) and involuntary context switches (preemption).
# - recompiled: the jit cache of the chunk function changed size
#   during the chunk. Always true for chunk 0, and for a shorter last
#   chunk.
#
# Online, a chunk is judged against the median of the eligible chunks
# before it, with the same factor and floor. It needs fewer of them
# than the correction does: a false alarm costs one log line rather
# than a corrected or crashed run, and an outlier among the first few
# chunks of a short run is just as worth a snapshot. Three is the
# fewest whose median a single earlier outlier cannot drag up.
_ONLINE_MIN_CHUNKS = 3

# nvidia-smi can hang on a wedged GPU -- itself worth knowing, so a
# timeout is reported rather than swallowed.
_NVIDIA_SMI_TIMEOUT_S = 5.0
_NVIDIA_SMI_QUERY = (
    'utilization.gpu,clocks.sm,clocks_throttle_reasons.active,'
    'temperature.gpu,power.draw,memory.used')
_NVIDIA_SMI_LABEL = 'util, sm clock, throttle reasons, temp C, power, mem'

# Below this share of training wall time spent waiting on the input
# pipeline, the summary line is phrased as a neutral statistic rather
# than as a diagnosis.
_INPUT_BOUND_FRACTION = 0.10

# Local log of the learned tied-embedding input scale, appended after
# every completed run of an EmbeddingCodec model. One JSON object per
# line: {time, system, spec, lr, steps, x, y, scale, loss}. Collected
# manually across machines for the X-vs-exp(y) analysis -- deliberately
# no client/server protocol.
_EMB_SCALE_LOG = 'results/emb_scale.jsonl'


@dataclass(frozen=True)
class _Usage:
    """Process counters read at a chunk boundary. The rusage fields
    are None where the `resource` module is missing (Windows) or
    getrusage failed."""
    clock: float
    cpu: float
    utime: float | None = None
    stime: float | None = None
    majflt: int | None = None
    inblock: int | None = None
    nivcsw: int | None = None


def _read_usage() -> _Usage:
    """The counters `_ChunkRecord` takes deltas of. Never raises."""
    clock = time.time()
    cpu = time.process_time()
    if resource is None:
        return _Usage(clock, cpu)
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return _Usage(clock, cpu, ru.ru_utime, ru.ru_stime, ru.ru_majflt,
                      ru.ru_inblock, ru.ru_nivcsw)
    except Exception:
        return _Usage(clock, cpu)


def _delta(start: float | None, end: float | None) -> float | None:
    return None if start is None or end is None else end - start


@dataclass(frozen=True)
class _ChunkRecord:
    """What one training chunk cost, for the anomaly diagnostics (see
    the comment above `_ONLINE_MIN_CHUNKS`). Durations in seconds;
    None where the platform or the installed jax cannot tell."""
    index: int
    first_step: int
    steps: int
    wall: float  # perf_counter, sample + compute: what the detector judges
    sample: float
    compute: float
    clock: float  # time.time() over the same interval
    cpu: float  # process CPU time, all threads
    utime: float | None
    stime: float | None
    majflt: int | None
    inblock: int | None
    nivcsw: int | None
    recompiled: bool | None
    queue_depth: int | None  # prefetched batches at chunk start


def _make_chunk_record(
    index: int,
    first_step: int,
    steps: int,
    wall: float,
    sample: float,
    compute: float,
    start: _Usage,
    end: _Usage,
    recompiled: bool | None,
    queue_depth: int | None,
) -> _ChunkRecord:
    return _ChunkRecord(
        index=index, first_step=first_step, steps=steps,
        wall=wall, sample=sample, compute=compute,
        clock=end.clock - start.clock,
        cpu=end.cpu - start.cpu,
        utime=_delta(start.utime, end.utime),
        stime=_delta(start.stime, end.stime),
        majflt=_delta(start.majflt, end.majflt),
        inblock=_delta(start.inblock, end.inblock),
        nivcsw=_delta(start.nivcsw, end.nivcsw),
        recompiled=recompiled,
        queue_depth=queue_depth,
    )


def _jit_cache_size(fn) -> int | None:
    """How many compiled variants a jitted function holds; None where
    the installed jax lacks this private API."""
    try:
        return fn._cache_size()
    except Exception:
        return None


def _queue_depth(
    dataset, ntokens: int, batch: int, tokenset_name: str
) -> int | None:
    """Prefetched batches waiting for this request shape; None for a
    sampler without a prefetch queue."""
    if not isinstance(dataset, DataSetWrapper):
        return None
    try:
        return dataset.queue_depth(ntokens, batch, tokenset_name)
    except Exception:
        return None


def _fmt_s(t: float | None) -> str:
    """`ttoa3`, extended to None and to negative intervals (a clock
    stepped backwards)."""
    if t is None:
        return '?'
    return f'-{ttoa3(-t)}' if t < 0 else ttoa3(t)


def _format_chunk_record(r: _ChunkRecord) -> str:
    """Everything a chunk record knows, on one line. Fields the
    platform could not measure are left out."""
    cpu = f'cpu {_fmt_s(r.cpu)}'
    if r.utime is not None and r.stime is not None:
        cpu += f' (user {_fmt_s(r.utime)}, sys {_fmt_s(r.stime)})'
    parts = [
        f'chunk {r.index} (steps {r.first_step}-'
        f'{r.first_step + r.steps - 1}): wall {_fmt_s(r.wall)} '
        f'(sample {_fmt_s(r.sample)}, compute {_fmt_s(r.compute)})',
        f'clock {_fmt_s(r.clock)}',
        cpu,
    ]
    if r.majflt is not None:
        parts.append(f'majflt {r.majflt}')
    if r.inblock is not None:
        parts.append(f'inblock {r.inblock}')
    if r.nivcsw is not None:
        parts.append(f'nivcsw {r.nivcsw}')
    if r.recompiled is not None:
        parts.append(f'recompiled {"yes" if r.recompiled else "no"}')
    if r.queue_depth is not None:
        parts.append(f'queue depth {r.queue_depth}')
    return ', '.join(parts)


def _typical_record(records: list[_ChunkRecord]) -> _ChunkRecord:
    """The record at the (lower) median wall time: the steady-state
    chunk an outlier is set against."""
    return sorted(records, key=lambda r: r.wall)[(len(records) - 1) // 2]


def _first_line(path: str) -> str:
    with open(path) as f:
        return f.readline().strip()


def _probe_psi(root: str = '/proc/pressure') -> str | None:
    """Linux pressure stall information: the first (`some`) line for
    cpu, io and memory -- the share of the last 10/60/300 s in which
    at least one task stalled on that resource."""
    parts = []
    for name in ('cpu', 'io', 'memory'):
        try:
            parts.append(f'{name} {_first_line(os.path.join(root, name))}')
        except OSError:
            pass  # no PSI on this kernel or platform
    return f'psi: {" | ".join(parts)}' if parts else None


def _probe_loadavg(path: str = '/proc/loadavg') -> str | None:
    return f'loadavg {_first_line(path)}'


def _probe_meminfo(path: str = '/proc/meminfo') -> str | None:
    """MemAvailable and SwapFree from /proc/meminfo, in GiB."""
    wanted = ('MemAvailable', 'SwapFree')
    found = {}
    with open(path) as f:
        for line in f:
            key, _, rest = line.partition(':')
            if key in wanted:
                found[key] = int(rest.split()[0]) / 2**20  # kB -> GiB
    return ', '.join(
        f'{key} {found[key]:.2f} GiB' for key in wanted if key in found
    ) or None


def _probe_gpu() -> str | None:
    """One nvidia-smi query; None on a machine without the driver."""
    try:
        out = subprocess.run(
            ['nvidia-smi', f'--query-gpu={_NVIDIA_SMI_QUERY}',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return (f'gpu: nvidia-smi timed out after '
                f'{ttoa3(_NVIDIA_SMI_TIMEOUT_S)}')
    if out.returncode != 0:
        err = (out.stderr or out.stdout or '').strip().splitlines()
        reason = f' ({err[0].strip()})' if err else ''
        return f'gpu: nvidia-smi exit {out.returncode}{reason}'
    gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not gpus:
        return None
    return f'gpu ({_NVIDIA_SMI_LABEL}): {" | ".join(gpus)}'


def _system_snapshot() -> str:
    """Whatever this machine can say about its state right now, on one
    line. A probe that fails or does not apply here is left out; the
    snapshot itself never raises."""
    parts = []
    for probe in (_probe_psi, _probe_loadavg, _probe_meminfo, _probe_gpu):
        try:
            part = probe()
        except Exception:
            continue
        if part:
            parts.append(part)
    return '; '.join(parts) or 'nothing to report'


def _suspend_threshold(eligible: list[float]) -> tuple[float, float]:
    """`(median, threshold)` over the eligible chunk times. Both
    conditions at once: relative to this run *and* long enough to be
    a nap rather than a data-pipeline hiccup."""
    median = statistics.median(eligible)
    return median, max(_SUSPEND_OUTLIER_FACTOR * median, _SUSPEND_MIN_GAP_S)


def _online_outlier_threshold(chunk_times: list[float]) -> float | None:
    """The threshold the latest chunk exceeded, judged against the
    eligible chunks before it; None if it is no outlier, or if there
    are fewer than `_ONLINE_MIN_CHUNKS` of those to judge by."""
    previous = chunk_times[1:-1]
    if len(previous) < _ONLINE_MIN_CHUNKS:
        return None
    _, threshold = _suspend_threshold(previous)
    return threshold if chunk_times[-1] > threshold else None


def _warn_if_anomalous(records: list[_ChunkRecord]) -> bool:
    """Warn, with its record and a system snapshot, the moment the
    latest chunk turns out anomalous; returns whether it did. Never
    raises -- the verdict that counts is `_correct_chunk_times`'s at
    the end of the run."""
    # The steady state: under the floor no chunk is ever an outlier,
    # so no list and no median.
    if records[-1].wall <= _SUSPEND_MIN_GAP_S:
        return False
    threshold = _online_outlier_threshold([r.wall for r in records])
    if threshold is None:
        return False
    try:
        logging.warning(
            f'anomalous chunk: {_format_chunk_record(records[-1])} -- over '
            f'the threshold {ttoa3(threshold)}; typical '
            f'{_format_chunk_record(_typical_record(records[1:-1]))}; '
            f'system: {_system_snapshot()}')
    except Exception:
        logging.exception('anomalous-chunk diagnostics failed (ignored)')
    return True


def _outlier_details(
    records: list[_ChunkRecord] | None, outliers: list[int], sep: str
) -> str:
    """The outlying chunks' records and, for scale, the typical one,
    each led by `sep`. Empty without records. A formatting failure
    degrades to a note rather than masking the caller's verdict."""
    if not records:
        return ''
    try:
        lines = [f'outlier {_format_chunk_record(records[i])}'
                 for i in outliers]
        lines.append(
            f'typical {_format_chunk_record(_typical_record(records[1:]))}')
    except Exception as e:
        return f'{sep}(chunk details unavailable: {e!r})'
    return ''.join(sep + line for line in lines)


def _correct_chunk_times(
    chunk_times: list[float],
    records: list[_ChunkRecord] | None = None,
) -> tuple[float, int]:
    """Total training wall time, with a single suspend-sized gap repaired.

    `chunk_times` are the measured durations of the training chunks,
    in order. Returns `(total_seconds, n_outliers)`, where the total
    has any single outlying chunk replaced by the median of the
    others. A chunk is an outlier only if it is both over
    `_SUSPEND_OUTLIER_FACTOR` times the median of the eligible chunks
    and over `_SUSPEND_MIN_GAP_S` in absolute terms (see the
    constants above for what counts and why).

    `records`, if given, are the same chunks' `_ChunkRecord`s; the
    suspend warning and the error then quote the outliers' records
    and a typical one, so the log line or the traceback alone tells
    what the outlying chunks were doing.

    Raises:
        RuntimeError: more than one outlying chunk. A suspend that
            straddles a chunk boundary smears across two chunks and
            lands here too -- deliberately: from the outside that is
            indistinguishable from repeated suspends or a broken
            clock, and guessing the true time back is not something
            this function should do. Crashing is what gets noticed on
            a headless worker, and it keeps the run from being
            submitted with garbage timing.
    """
    total = math.fsum(chunk_times)
    # Chunk 0 pays for the JIT compile; it is never a suspend signal.
    eligible = chunk_times[1:]
    if len(eligible) < _SUSPEND_MIN_CHUNKS:
        # Too few chunks for the median to mean anything (a 512-step
        # run is 2 chunks). Short runs are accepted as uncorrectable.
        return total, 0

    median, threshold = _suspend_threshold(eligible)
    # Indices into `chunk_times`, so the records can be quoted.
    outliers = [i for i in range(1, len(chunk_times))
                if chunk_times[i] > threshold]
    if not outliers:
        return total, 0

    if len(outliers) > 1:
        durations = ', '.join(ttoa3(chunk_times[i]) for i in outliers)
        raise RuntimeError(
            f'anomalous chunk times: {len(outliers)} chunks over the '
            f'suspend threshold {ttoa3(threshold)} '
            f'({_SUSPEND_OUTLIER_FACTOR:g}x the median {ttoa3(median)}, '
            f'floor {ttoa3(_SUSPEND_MIN_GAP_S)}): {durations} -- out of a '
            f'measured total of {ttoa3(total)}. Repeated suspends or a '
            f'broken clock -- the timing of this run is garbage and it '
            f'must not be submitted.'
            + _outlier_details(records, outliers, '\n  '))

    outlier = chunk_times[outliers[0]]
    corrected = total - outlier + median
    logging.warning(
        f'suspend detected: one chunk took {ttoa3(outlier)} against a '
        f'median of {ttoa3(median)}; correcting train time '
        f'{ttoa3(total)} -> {ttoa3(corrected)}'
        + _outlier_details(records, outliers, '; '))
    return corrected, 1


def _log_input_bound(sample_time: float, compute_time: float) -> None:
    """Log how much of the training wall time went to waiting on the
    input pipeline rather than to the model.

    Cheap and always logged: a conf whose chunks are mostly queue wait
    trains no faster on a better accelerator, and the same waits are
    what the suspend detector's absolute floor exists to ignore. The
    two arguments are the measured phase totals and sum to the
    measured training wall time -- after a suspend correction the gap
    is still inside whichever phase swallowed it.
    """
    wall = sample_time + compute_time
    if wall <= 0:
        return
    frac = sample_time / wall
    detail = f'(sample {ttoa3(sample_time)}, compute {ttoa3(compute_time)})'
    if frac >= _INPUT_BOUND_FRACTION:
        logging.info(
            f'input-bound: {frac:.0%} of wall time waiting for data {detail}')
    else:
        logging.info(f'data wait: {frac:.0%} of wall time {detail}')


class ManagerJax(Manager):
    """JAX training backend."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dtype = self.conf.precision.jax_dtype

        logging.info(f'{self.conf}')
        self.model: Model2Jax = self.model_def.build_jax()
        rng = jax.random.PRNGKey(random.randrange(2**32))
        self.weights: list[LayerWeights] = self.model.init_weights(rng)

        if self.verbose:
            logging.info('Creating optimizer')
        self._build_optimizer()
        self.run = Run(loss_trend=LossTrend(), system=self.system)

        # JIT-compile combined loss+gradient computation (single fwd+bwd pass).
        self._loss_grad = jax.jit(jax.value_and_grad(self.model.loss_batch))
        # JIT-compile a chunk of training steps (one fwd+bwd+update
        # per scan iteration). One Python dispatch + one host sync per
        # `_CHUNK_SIZE` steps, which is the main speedup vs the per-
        # step loop in `Manager.train`.
        self._train_chunk = jax.jit(self._build_train_chunk_fn())

    def _build_train_chunk_fn(self):
        loss_fn = self.model.loss_batch
        optimizer = self.optimizer

        def chunk(weights, opt_state, batches):
            def step(carry, batch):
                w, opt = carry
                loss, grads = jax.value_and_grad(loss_fn)(w, batch)
                updates, opt = optimizer.update(grads, opt, w)
                w = optax.apply_updates(w, updates)
                return (w, opt), loss
            (w_out, opt_out), losses = jax.lax.scan(
                step, (weights, opt_state), batches)
            return w_out, opt_out, losses

        return chunk

    def _build_optimizer(self):
        lr = self.conf.lr
        if self.conf.cosine:
            lr = optax.cosine_decay_schedule(
                init_value=self.conf.lr,
                decay_steps=self.conf.steps,
                alpha=0.0,
            )
        elif self.conf.decay != 1:
            initial_lr = self.conf.lr
            decay = self.conf.decay
            steps = self.conf.steps
            lr = lambda count: initial_lr * decay ** (count / steps)

        eps = 1e-4 if self.conf.precision == Precision.FP16 else 1e-8
        # Apply weight decay only to weight matrices, not biases.
        mask_bias = lambda tree: jax.tree.map(
            lambda x: x.ndim >= 2, tree)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr, weight_decay=0.01, eps=eps, mask=mask_bias),
        )
        self._opt_state = self.optimizer.init(self.weights)

    def _get_batch(self):
        data = self.dataset.sample_tokens(
            ntokens=self.conf.length,
            batch=self.conf.batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        return jnp.asarray(data)

    def train_step(self, batch) -> float:
        loss, grads = self._loss_grad(self.weights, batch)

        updates, self._opt_state = self.optimizer.update(
            grads, self._opt_state, self.weights)
        self.weights = optax.apply_updates(self.weights, updates)

        loss_val = self.tokenset.byte_loss(float(loss))
        self.run.add_step(loss_val)
        return loss_val

    # Toggleable from the train CLI for A/B benchmarking. When False,
    # falls back to the per-step `Manager.train` loop (which still
    # uses our `train_step` and so supports time_limit, divergence
    # early-exit, etc.).
    scan_train: bool = True

    def train(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
    ) -> tuple[Optional[float], Configuration]:
        """JAX-specific training: batches `_CHUNK_SIZE` steps into a
        JIT'd `lax.scan`. Drops the per-step loop's host sync, which
        is the main bottleneck for tiny models on accelerators.

        time_limit is unsupported on this path -- the search doesn't
        use it, and a chunk runs uninterruptibly. Wall-clock time
        includes the first chunk (which carries JIT-compile cost on
        the first occurrence of each unique tensor shape); the DB's
        median across runs filters the resulting outliers. A single
        suspend-sized gap in the remaining chunks is repaired by
        `_correct_chunk_times`; several of them raise. Each chunk's
        time is also split into input-pipeline wait and compute, and
        the ratio is logged once at the end. Every chunk also keeps a
        `_ChunkRecord` of process counters; one over the suspend
        threshold is logged with its record and a system snapshot as
        soon as it ends, and the end-of-run warning or error quotes
        the outliers' records.

        NaN losses propagate -- once the model diverges, subsequent
        steps run with NaN updates and the final eval will catch it.
        We don't try to early-exit.
        """
        if not self.scan_train:
            return super().train(steps, time_limit)
        if steps is None:
            steps = self.conf.steps
        if time_limit is not None:
            logging.warning(
                'time_limit is not supported with the chunked JAX '
                'trainer; ignoring. Pass --no-scan to use the per-step '
                'loop instead.')
        logging.info(f'Training for {steps} steps')
        # A chunk is one uninterruptible scan, so a corpus switch can
        # only land between chunks.
        self._align_data_switch(_CHUNK_SIZE)

        batch = self.conf.batch
        length = self.conf.length
        tokens_name = self.model_def.input.tokens_name

        start_time = perf_counter()
        last_progress_log = start_time
        chunk_start = start_time
        # One record per chunk. Its `wall` is the per-iteration total
        # the suspend detector needs -- a nap can land in either phase,
        # and only the total is guaranteed to contain it; the phase
        # split and the counters are there to explain an outlier.
        records: list[_ChunkRecord] = []
        usage = _read_usage()
        cache_size = _jit_cache_size(self._train_chunk)
        while self.step < steps:
            self._maybe_switch_data()
            n = min(_CHUNK_SIZE, steps - self.step)
            first_step = self.step
            queue_depth = _queue_depth(
                self.dataset, length, n * batch, tokens_name)
            # One big sample of (n*batch, length), reshaped to
            # (n, batch, length). Avoids n round-trips through the
            # prefetch queue and n separate host->device transfers.
            data = self.dataset.sample_tokens(
                ntokens=length,
                batch=n * batch,
                tokenset_name=tokens_name,
            )
            sampled = perf_counter()
            batches = jnp.asarray(data).reshape(n, batch, length)
            self.weights, self._opt_state, losses = self._train_chunk(
                self.weights, self._opt_state, batches)
            # JAX dispatch is async, so the chunk's compute is only
            # actually paid for at this host sync -- which keeps it on
            # the compute side of the split, where it belongs.
            losses_host = np.asarray(losses)
            for loss_val in losses_host:
                self.run.add_step(self.tokenset.byte_loss(float(loss_val)))
            now = perf_counter()
            usage_end = _read_usage()
            cache_end = _jit_cache_size(self._train_chunk)
            recompiled = (None if cache_size is None or cache_end is None
                          else cache_end != cache_size)
            records.append(_make_chunk_record(
                index=len(records), first_step=first_step, steps=n,
                wall=now - chunk_start, sample=sampled - chunk_start,
                compute=now - sampled, start=usage, end=usage_end,
                recompiled=recompiled, queue_depth=queue_depth))
            # A snapshot, when one is taken, lands in the next chunk's
            # sample phase -- usually well under a second (at worst the
            # nvidia-smi timeout), and only right after an outlier.
            _warn_if_anomalous(records)
            usage, cache_size = usage_end, cache_end
            chunk_start = now
            if (
                self.verbose
                or now - last_progress_log >= _QUIET_PROGRESS_INTERVAL
            ):
                last = self.tokenset.byte_loss(float(losses_host[-1]))
                logging.info(f'{self.step}  {last:.4f} b/B')
                last_progress_log = now

        total_time, _ = _correct_chunk_times(
            [r.wall for r in records], records)
        logging.info(f'Trained for {self.step} steps in {ttoa3(total_time)}')
        _log_input_bound(math.fsum(r.sample for r in records),
                         math.fsum(r.compute for r in records))
        return total_time, self.conf.replace(steps=self.step)

    def train_and_eval(
        self,
        steps: Optional[int],
        time_limit: Optional[float],
    ):
        run, final_conf = super().train_and_eval(steps, time_limit)
        self._log_emb_scale()
        return run, final_conf

    def _log_emb_scale(self) -> None:
        """Append the learned input scale of tied-embedding models to
        `_EMB_SCALE_LOG`. Best-effort: a logging failure must never
        fail the run."""
        w0 = self.weights[0]
        if not isinstance(w0, dict) or 'y' not in w0:
            return  # not an EmbeddingCodec model
        try:
            y = float(w0['y'])
            loss = self.run.loss
            rec = {
                'time': datetime.now().isoformat(timespec='seconds'),
                'system': self.system,
                'spec': str(self.model_def),
                'lr': self.conf.lr,
                'steps': self.run.steps,
                'x': int(w0['emb'].shape[-1]),
                'y': y,
                'scale': math.exp(y),
                'loss': loss if math.isfinite(loss) else None,
            }
            log_dir = os.path.dirname(_EMB_SCALE_LOG)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(_EMB_SCALE_LOG, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except Exception:
            logging.exception('emb-scale logging failed (ignored)')

    def eval(self) -> float:
        """Evaluate the model on a random test batch in fp32.

        Samples by bytes (not tokens) so the eval metric is consistent
        across tokenizations with different bytes-per-token ratios.
        """
        weights32 = jax.tree.map(
            lambda x: x.astype(jnp.float32), self.weights)
        model32 = parse_model2(
            self.model_def.spec, Precision.FP32).build_jax()

        batch, lengths = self.dataset.sample_bytes(
            nbytes=self.test_sample_len,
            batch=self.test_batch,
            tokenset_name=self.model_def.input.tokens_name,
        )
        # Recurrent form avoids the O(B*H*L^2) scores tensor that the
        # parallel forward materializes — required to fit 1024x1024 eval
        # in 16 GB on systems with msr layers.
        total_loss = model32.loss_batch_masked_recurrent(
            weights32, batch, lengths)
        # Already per-byte (sampled by bytes); fold sets add their
        # forgetting charge on top.
        return (float(total_loss) / (self.test_sample_len * self.test_batch)
                + self.tokenset.residual_bits_per_byte)

    def continue_prefix(
        self, prefix: str, length: int, temperature: float
    ) -> bytes:
        """Sample text continuation from the model."""
        return generate.continue_prefix(
            self.model, self.weights, self.model_def.input.tokens_name,
            prefix, length, temperature)
