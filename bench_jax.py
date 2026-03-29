import argparse
import math
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


def train(args):
    print(f'Backend: {jax.default_backend()}')
    print(f'Devices: {jax.devices()}')

    dataset = DataSet(path=args.data)
    rng_key = jax.random.PRNGKey(0)

    weights = init_weights(rng_key)
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
    opt_state = optimizer.init(weights)

    for step in range(args.steps):
        if step == 1:
            start = perf_counter()

        data = dataset.sample_tokens(ntokens=256, batch=256, tokenset_name='bytes')
        loss, grads = loss_grad(weights, data)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        if step < 10 or step % 10 == 0:
            print(f'{step} {loss:.4f}')

    finish = perf_counter()
    elapsed = finish - start
    ms_per_step = elapsed / (args.steps - 1) * 1000
    print(f'Latency: {elapsed:.2f}s ({ms_per_step:.1f} ms/step)')


def main():
    parser = argparse.ArgumentParser(description='JAX GRU benchmark')
    parser.add_argument('-d', '--data', type=str, default='data/books3.txt',
                        help='path to data file')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of training steps')
    parser.add_argument('--lr', type=float, default=0.0078125,
                        help='learning rate (default: 1/128)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
