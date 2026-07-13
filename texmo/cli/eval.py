"""Evaluate a stored model with the training-eval protocol.

Same measurement as Manager.eval: random test samples of a fixed
size in BYTES (variable size in tokens after tokenization), losses
masked to each sample's true token length, total cross-entropy
divided by nominal bytes -- bits/byte, comparable across models and
tokenizations.

The batch is processed in chunks (masked bit-sums are additive, so
chunking is mathematically identical to one big batch): the
recurrent loss materializes logits of shape (chunk, T, vocab), which
for a 256k vocabulary is ~0.35 GB per sample in fp32 -- keep --chunk
small for large-vocabulary models.

    uv run python texmo.py eval -m models/recurrentgemma-2b.json \
        --test-batch 64 --chunk 4

Runs at the manifest's precision.
"""
import argparse
import time

import jax.numpy as jnp
import numpy as np

from ..dataset import DataSet
from ..model_store import load_model
from ..tokens import get_tokenizer, set_tokens_dir


def eval_model(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    dataset = DataSet(path=args.data, path_processed=args.data_processed)

    t0 = time.perf_counter()
    md, weights = load_model(args.model)
    model = md.build_jax()
    tokens_name = md.codec.tokens_name
    print(f'model:  {args.model} ({md.num_weights:,} weights, '
          f'{md.precision}), tokens: {tokens_name}')
    print(f'loaded in {time.perf_counter() - t0:.1f}s', flush=True)

    total_bits = 0.0
    total_bytes = 0
    total_tokens = 0
    done = 0
    t0 = time.perf_counter()
    while done < args.test_batch:
        b = min(args.chunk, args.test_batch - done)
        batch, lengths = dataset.sample_bytes(
            nbytes=args.test_sample_len, batch=b,
            tokenset_name=tokens_name)
        bits = float(model.loss_batch_masked_recurrent(
            weights, jnp.asarray(batch), jnp.asarray(lengths)))
        total_bits += bits
        total_bytes += b * args.test_sample_len
        total_tokens += int(np.sum(lengths))
        done += b
        print(f'  [{done:5d}/{args.test_batch}]  '
              f'running {total_bits / total_bytes:.4f} b/B  '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)

    print(f'\nloss:  {total_bits / total_bytes:.4f} bits/byte')
    print(f'       {total_bits / total_tokens:.4f} bits/token  '
          f'({total_bytes / total_tokens:.2f} bytes/token realized)')
    print(f'bytes: {total_bytes:,}  tokens: {total_tokens:,}  '
          f'wall: {time.perf_counter() - t0:.0f}s')


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        '-m', '--model', default='models/recurrentgemma-2b.json',
        help='model JSON manifest (default: models/recurrentgemma-2b.json)')
    parser.add_argument(
        '-d', '--data', default=config.DATA,
        help=f"data file to sample from (default: '{config.DATA}')")
    parser.add_argument(
        '--data-processed', default=config.DATA_CAPS_WORDS,
        help='pre-processed (capswords) data file, if the tokenset '
             'needs it')
    parser.add_argument(
        '--test-batch', type=int, default=1024,
        help='total number of test samples (default: 1024)')
    parser.add_argument(
        '--test-sample-len', type=int, default=1024,
        help='sample length in bytes (default: 1024)')
    parser.add_argument(
        '--chunk', type=int, default=8,
        help='samples per forward pass; the (chunk, T, vocab) logits '
             'tensor must fit in memory (default: 8)')
    parser.add_argument(
        '--tokens-dir', default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')")
    parser.set_defaults(func=eval_model)
