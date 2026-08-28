import json
import logging
import math
import os
import random
import statistics
from datetime import datetime
from time import perf_counter
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import generate
from .common import ttoa3
from .configuration import Configuration
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


def _correct_chunk_times(chunk_times: list[float]) -> tuple[float, int]:
    """Total training wall time, with a single suspend-sized gap repaired.

    `chunk_times` are the measured durations of the training chunks,
    in order. Returns `(total_seconds, n_outliers)`, where the total
    has any single outlying chunk replaced by the median of the
    others. A chunk is an outlier only if it is both over
    `_SUSPEND_OUTLIER_FACTOR` times the median of the eligible chunks
    and over `_SUSPEND_MIN_GAP_S` in absolute terms (see the
    constants above for what counts and why).

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

    median = statistics.median(eligible)
    # Both conditions at once: relative to this run *and* long enough
    # to be a nap rather than a data-pipeline hiccup.
    threshold = max(_SUSPEND_OUTLIER_FACTOR * median, _SUSPEND_MIN_GAP_S)
    outliers = [t for t in eligible if t > threshold]
    if not outliers:
        return total, 0

    if len(outliers) > 1:
        durations = ', '.join(ttoa3(t) for t in outliers)
        raise RuntimeError(
            f'anomalous chunk times: {len(outliers)} chunks over the '
            f'suspend threshold {ttoa3(threshold)} '
            f'({_SUSPEND_OUTLIER_FACTOR:g}x the median {ttoa3(median)}, '
            f'floor {ttoa3(_SUSPEND_MIN_GAP_S)}): {durations} -- out of a '
            f'measured total of {ttoa3(total)}. Repeated suspends or a '
            f'broken clock -- the timing of this run is garbage and it '
            f'must not be submitted.')

    corrected = total - outliers[0] + median
    logging.warning(
        f'suspend detected: one chunk took {ttoa3(outliers[0])} against a '
        f'median of {ttoa3(median)}; correcting train time '
        f'{ttoa3(total)} -> {ttoa3(corrected)}')
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
        the ratio is logged once at the end.

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
        chunk_times: list[float] = []
        # Phase split of the same intervals: how much of each chunk
        # went to waiting for the sampler and how much to the model.
        # Kept apart from `chunk_times`, which stays the per-iteration
        # total the suspend detector needs -- a nap can land in either
        # phase, and only the total is guaranteed to contain it.
        sample_time = 0.0
        compute_time = 0.0
        while self.step < steps:
            self._maybe_switch_data()
            n = min(_CHUNK_SIZE, steps - self.step)
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
            sample_time += sampled - chunk_start
            compute_time += now - sampled
            chunk_times.append(now - chunk_start)
            chunk_start = now
            if (
                self.verbose
                or now - last_progress_log >= _QUIET_PROGRESS_INTERVAL
            ):
                last = self.tokenset.byte_loss(float(losses_host[-1]))
                logging.info(f'{self.step}  {last:.4f} b/B')
                last_progress_log = now

        total_time, _ = _correct_chunk_times(chunk_times)
        logging.info(f'Trained for {self.step} steps in {ttoa3(total_time)}')
        _log_input_bound(sample_time, compute_time)
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
