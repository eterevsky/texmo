"""JAX GRU.256 training benchmark with three data-delivery modes.

The same per-step compute (jitted loss + grad + optimizer update) is
driven from three different data sources, to isolate where time goes:

  disk  -- sample from the memory-mapped file each step (real pipeline).
  host  -- read a chunk of the file into a host (CPU RAM) buffer once,
           then sample + copy host->device each step. Removes disk I/O,
           keeps the host->device (PCIe) transfer.
  gpu   -- load the chunk onto the GPU once and sample on-device each
           step. Removes disk I/O AND the per-step PCIe transfer; this
           is the pure-compute floor.

Reading the result:
  - disk >> host  => disk I/O dominates (e.g. mmap readahead).
  - host >> gpu   => host->device transfer dominates (PCIe / a riser
                     degrading the link).
  - all three ~=  => compute / driver / config is the bottleneck.
"""
import argparse
import math
import os
from time import perf_counter

import numpy as np
import jax
import jax.numpy as jnp
import optax

from texmo.dataset import DataSet

_1_BY_LOG2 = 1.0 / math.log(2.0)


def init_weights(rng_key, vocab_size=256, hidden_size=256):
    keys = jax.random.split(rng_key, 5)
    scale = 0.01
    return {
        'wi': jax.random.normal(keys[0], (hidden_size, vocab_size)) * scale,
        'wh': jax.random.normal(keys[1], (hidden_size, hidden_size)) * scale,
        'bi': jnp.zeros((hidden_size,)),
        'wr': jax.random.normal(keys[2], (hidden_size, vocab_size)) * scale,
        'wrh': jax.random.normal(keys[3], (hidden_size, hidden_size)) * scale,
        'br': jnp.zeros((hidden_size,)),
        'wz': jax.random.normal(keys[2], (hidden_size, vocab_size)) * scale,
        'wzh': jax.random.normal(keys[3], (hidden_size, hidden_size)) * scale,
        'bz': jnp.zeros((hidden_size,)),
        'wout': jax.random.normal(keys[4], (vocab_size, hidden_size)) * scale,
        'bout': jnp.zeros((vocab_size,)),
    }


def build_gru(vocab_size=256, hidden_size=256):
    def pipeline(weights, input_tokens):
        # input_tokens: (batch, seq_len)
        oh = jax.nn.one_hot(input_tokens, vocab_size, dtype=jnp.float32)
        # Shift right: prepend zeros, drop last
        oh = jnp.pad(oh[:, :-1], ((0, 0), (1, 0), (0, 0)))

        def step(h, x):
            # x: (batch, vocab_size), h: (batch, hidden_size)
            r = jax.nn.sigmoid(x @ weights['wr'].T + h @ weights['wrh'].T + weights['br'])
            z = jax.nn.sigmoid(x @ weights['wz'].T + h @ weights['wzh'].T + weights['bz'])
            h_candidate = jnp.tanh(x @ weights['wi'].T + (r * h) @ weights['wh'].T + weights['bi'])
            h_new = (1 - z) * h + z * h_candidate
            return h_new, h_new

        # Transpose to (seq_len, batch, vocab_size) for scan
        oh_t = jnp.transpose(oh, (1, 0, 2))
        init_h = jnp.zeros((input_tokens.shape[0], hidden_size))
        _, mid = jax.lax.scan(step, init_h, oh_t)
        # mid: (seq_len, batch, hidden_size) -> (batch, seq_len, hidden_size)
        mid = jnp.transpose(mid, (1, 0, 2))

        logits = mid @ weights['wout'].T + weights['bout']
        return logits  # (batch, seq_len, vocab_size)

    return pipeline


# --- data sources -------------------------------------------------------
#
# Every sampler returns a (batch, ntokens) int32 array. We standardize on
# int32 so the jitted loss compiles once and is shared across all modes.


def _load_host_buffer(path: str, max_bytes: int) -> np.ndarray:
    """Read up to max_bytes from the file into a host uint8 array."""
    size = os.path.getsize(path)
    n = min(size, max_bytes)
    buf = np.fromfile(path, dtype=np.uint8, count=n)
    print(f'  loaded {n / 2**30:.2f} GiB into host buffer '
          f'(file is {size / 2**30:.2f} GiB)')
    return buf


def make_disk_sampler(dataset: DataSet, ntokens: int, batch: int):
    def sample():
        data = dataset.sample_tokens(
            ntokens=ntokens, batch=batch, tokenset_name='bytes')
        return np.asarray(data, dtype=np.int32)
    return sample


def make_host_sampler(buf: np.ndarray, ntokens: int, batch: int):
    n = buf.shape[0]
    cols = np.arange(ntokens)

    def sample():
        offs = np.random.randint(0, n - ntokens, size=batch)
        idx = offs[:, None] + cols[None, :]            # (batch, ntokens)
        return buf[idx].astype(np.int32)               # host->device on use
    return sample


def make_gpu_sampler(buf: np.ndarray, ntokens: int, batch: int):
    # Keep the big buffer on-device as uint8 (1 byte/token); gather + cast
    # to int32 happen on-device, so no per-step host->device transfer.
    buf_dev = jax.device_put(jnp.asarray(buf, dtype=jnp.uint8))
    n = int(buf_dev.shape[0])
    cols = jnp.arange(ntokens)

    @jax.jit
    def _sample(buf_dev, key):
        offs = jax.random.randint(key, (batch,), 0, n - ntokens)
        idx = offs[:, None] + cols[None, :]            # (batch, ntokens)
        return buf_dev[idx].astype(jnp.int32)

    state = {'key': jax.random.PRNGKey(1)}

    def sample():
        state['key'], sub = jax.random.split(state['key'])
        return _sample(buf_dev, sub)
    return sample


def run_mode(name, sample, loss_grad, optimizer, rng_key, steps):
    """Warm up once (compile + first dispatch), then time `steps`."""
    weights = init_weights(rng_key)
    opt_state = optimizer.init(weights)

    start = None
    last_loss = None
    for step in range(steps + 1):  # step 0 is warmup, not timed
        data = sample()
        loss, grads = loss_grad(weights, data)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        last_loss = loss
        if step == 0:
            jax.block_until_ready(weights)  # finish compile + warmup
            start = perf_counter()

    jax.block_until_ready(weights)          # flush async dispatch
    elapsed = perf_counter() - start
    ms = elapsed / steps * 1000
    print(f'{name:5s}  {elapsed:7.2f}s  ({ms:6.1f} ms/step)  '
          f'last_loss={float(last_loss):.4f}')
    return name, elapsed, ms


def main():
    parser = argparse.ArgumentParser(description='JAX GRU benchmark')
    parser.add_argument('-d', '--data', type=str, default='data/books3.txt',
                        help='path to data file')
    parser.add_argument('--steps', type=int, default=256,
                        help='number of timed training steps per mode')
    parser.add_argument('--lr', type=float, default=1/128,
                        help='learning rate (default: 1/128)')
    parser.add_argument('--mode', choices=('disk', 'host', 'gpu', 'all'),
                        default='all', help='data-delivery mode(s) to run')
    parser.add_argument('--mem-gb', type=float, default=1.0,
                        help='chunk size (GiB) for the host and gpu modes')
    parser.add_argument('--read-mode', choices=('mmap', 'pread'),
                        default='mmap', help='disk-mode read path')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--ntokens', type=int, default=256)
    args = parser.parse_args()

    print(f'Backend: {jax.default_backend()}')
    print(f'Devices: {jax.devices()}')

    modes = ('disk', 'host', 'gpu') if args.mode == 'all' else (args.mode,)
    max_bytes = int(args.mem_gb * 2**30)

    pipeline = build_gru()

    def loss_fn(weights, batch):
        logits = pipeline(weights, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch)
        return _1_BY_LOG2 * jnp.mean(loss)

    loss_grad = jax.jit(jax.value_and_grad(loss_fn))

    mask_bias = lambda tree: jax.tree_util.tree_map(lambda g: len(g.shape) > 1, tree)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, mask=mask_bias, weight_decay=0.01),
    )

    rng_key = jax.random.PRNGKey(0)

    # Build the samplers each mode needs (load the host buffer once and
    # share it between host and gpu modes).
    host_buf = None
    if 'host' in modes or 'gpu' in modes:
        host_buf = _load_host_buffer(args.data, max_bytes)

    samplers = {}
    if 'disk' in modes:
        ds = DataSet(path=args.data, read_mode=args.read_mode)
        samplers['disk'] = make_disk_sampler(ds, args.ntokens, args.batch)
    if 'host' in modes:
        samplers['host'] = make_host_sampler(host_buf, args.ntokens, args.batch)
    if 'gpu' in modes:
        samplers['gpu'] = make_gpu_sampler(host_buf, args.ntokens, args.batch)

    print(f'\nGRU.256  batch={args.batch} ntokens={args.ntokens}  '
          f'{args.steps} steps/mode\n')
    results = []
    for name in modes:
        results.append(
            run_mode(name, samplers[name], loss_grad, optimizer,
                     rng_key, args.steps))

    if len(results) > 1:
        floor = min(ms for _, _, ms in results)
        print('\nsummary (ms/step, x over fastest):')
        for name, _, ms in results:
            print(f'  {name:5s}  {ms:6.1f}  ({ms / floor:.2f}x)')


if __name__ == '__main__':
    main()
