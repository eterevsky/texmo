import argparse
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .dataset import DataSet
from .tokens import set_tokens_dir
from .prng import Rng



def init_weights(rng, size: int, input_size: int) -> dict:
    return {
        'ws': rng.he((size, size), dtype=jnp.float32) * 0.01,
        'wi': rng.he((size, input_size), dtype=jnp.float32) * 0.01,
        'b': rng.normal((size,), dtype=jnp.float32) * 0.01,
        'wout': rng.he((256, size), dtype=jnp.float32) * 0.01,
        'bout': rng.normal((input_size,), dtype=jnp.float32) * 0.01,
    }

def build1(size: int) -> Callable:
    """(batch, length)"""

    def step(weights, state, inp):
        out = weights['wi'] @ inp + weights['ws'] @ state + weights['b']
        out = jnp.tanh(out)
        return out, out

    def forward(weights, input_sample):
        _, out = jax.lax.scan(
            lambda state, inp: step(weights, state, inp),
            jnp.zeros((size,), dtype=jnp.float32),
            input_sample
        )
        return out
        
    forward_batch = jax.vmap(forward, in_axes=(None, 0), out_axes=0)

    def pipeline(weights: dict, input: jax.typing.ArrayLike) -> jax.Array:
        input_oh = jax.nn.one_hot(input, 256, dtype=jnp.float32)
        input_oh = jnp.pad(input_oh, ((0, 0), (1, 0), (0, 0)))

        mid = forward_batch(weights, input_oh)
        b = jnp.reshape(weights['bout'], (1, 1, -1))
        return jnp.einsum('oi,bni->bno', weights['wout'], mid) + b

    return pipeline        


# def forward2(weights, input: jax.typing.ArrayLike) -> jax.Array:
#     """(length, batch)"""
#     input_oh = jax.nn.one_hot(input, 256, axis=1, dtype=jnp.float32)
#     print(input_oh.shape)
#     print(input_oh.dtype)
#     # (length, oh, batch)


def train(pipeline: Callable, steps: int, dataset, length: int, batch: int, lr: float, size):
    def _loss(weights, batch) -> float:
        out = pipeline(weights, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(out[:,:-1], batch)
        return (1/math.log(2)) * jnp.mean(loss)

    _loss_grad = jax.jit(jax.value_and_grad(_loss))

    rng = Rng()
    weights = init_weights(rng, size, 256)

    mask_bias = lambda tree: jax.tree_util.tree_map(lambda g: len(g.shape) > 1, tree)
    optimizer = optax.adamw(lr, mask=mask_bias, weight_decay=0.01)
    opt_state = optimizer.init(weights)

    for step in range(steps):
        data = dataset.sample_tokens(ntokens=length, batch=batch, tokenset_name='bytes')
        loss, grads = _loss_grad(weights, data)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

        if step < 10 or step % 10 == 0:
            print(step, loss)


def bench(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    dataset = DataSet(path=args.data, path_processed=None)
    lr = args.lr

    size = 128

    pipeline = build1(size)
    train(pipeline, steps=args.steps, dataset=dataset, length=args.length, batch=args.batch, lr=args.lr, size=size)




def init_args(parser: argparse.ArgumentParser, config):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help=f"a file with (default: '{config.DATA}')",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="number of training steps"
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        choices=["fp64", "fp32", "fp16", "bf16"],
        metavar="P",
        default="fp32",
        help="training precision",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default="0.0078125",
        help="learning rate, could be written as a float or as 2^-10 (default: 1/128)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training (default: 128)",
    )

    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')",
    )


    parser.set_defaults(func=bench)
