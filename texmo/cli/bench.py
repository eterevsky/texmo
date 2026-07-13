"""Benchmark loading + greedy generation of a stored model.

Runs greedy generation at several lengths (default 128 and 256).
Every run performs identical fixed work -- fresh states + the prompt
-- plus N generated tokens, all with a warm jit, so:

    per-token rate  = (T[2N] - T[N]) / N
    per-run overhead = 2*T[N] - T[2N]

while model load time and the jit compile (first step) are reported
separately. Greedy decoding and the manifest's own precision keep
runs deterministic and comparable, so the same command compares
machines directly:

    uv run python texmo.py benchmark-model
"""
import argparse
import time

import jax
import numpy as np

from ..model_store import load_model
from ..tokens import get_tokenizer, set_tokens_dir


def benchmark_model(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    lengths = sorted(int(n) for n in args.tokens.split(','))

    t0 = time.perf_counter()
    md, weights = load_model(args.model)
    t_load = time.perf_counter() - t0

    model = md.build_jax()
    step = jax.jit(model.step)
    tokenizer = get_tokenizer(md.codec.tokens_name)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)

    def fresh_states():
        return [model.codec.init_state(), model.layer_seq.init_state()]

    # First step compiles; measure it apart so runs are jit-warm.
    t0 = time.perf_counter()
    _, logits = step(weights, fresh_states(), prompt_ids[0])
    logits.block_until_ready()
    t_compile = time.perf_counter() - t0

    def run(n: int) -> tuple[float, list[int]]:
        states = fresh_states()
        out = list(prompt_ids)
        t0 = time.perf_counter()
        logits = None
        for t in prompt_ids:
            states, logits = step(weights, states, t)
        for _ in range(n):
            # np.asarray syncs with the device each step, so the
            # elapsed time covers the real computation.
            nxt = int(np.asarray(logits.argmax()))
            out.append(nxt)
            states, logits = step(weights, states, nxt)
        return time.perf_counter() - t0, out

    device = jax.devices()[0]
    print(f'model:     {args.model} ({md.num_weights:,} weights, '
          f'{md.precision})')
    print(f'system:    {args.system}  device: {device.platform}:'
          f'{device.device_kind}')
    print(f'load:      {t_load:7.2f}s')
    print(f'compile:   {t_compile:7.2f}s (first step, jit)')

    walls = {}
    out = list(prompt_ids)
    for n in lengths:
        walls[n], out = run(n)
        print(f'run[{n:4d}]: {walls[n]:7.2f}s  '
              f'({(len(prompt_ids) + n) / walls[n]:6.1f} steps/s)')

    if len(lengths) >= 2:
        n1, n2 = lengths[-2], lengths[-1]
        rate = (walls[n2] - walls[n1]) / (n2 - n1)
        overhead = walls[n1] - n1 * rate
        print(f'per-token: {rate * 1000:7.1f} ms  '
              f'({1 / rate:6.1f} tok/s steady-state)')
        print(f'overhead:  {overhead:7.2f}s per run (prompt + sync)')

    text = tokenizer.untokenize(out).decode('utf-8', errors='replace')
    print(f'\nlongest run text:\n{text}')


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        '-m', '--model', default='models/recurrentgemma-2b.json',
        help='model JSON manifest (default: models/recurrentgemma-2b.json)')
    parser.add_argument(
        '--tokens', default='128,256',
        help='comma-separated generation lengths (default: 128,256)')
    parser.add_argument(
        '--prompt', default='The capital of France is',
        help='prompt to continue')
    parser.add_argument(
        '--tokens-dir', default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')")
    parser.add_argument(
        '--system', default=config.SYSTEM_NAME,
        help=f"system name for the report (default: '{config.SYSTEM_NAME}')")
    parser.set_defaults(func=benchmark_model)
