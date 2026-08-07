"""Run the training benchmark suite locally -- no server, no DB.

Times the REAL training path: it builds a `ManagerJax` per entry, so
the model, optimizer, and both step paths are the ones the client
uses, and drives them directly to separate compile from steady state.

    scan    -- one jitted `lax.scan` over N steps: one dispatch and
               one host sync for the whole chunk (the client default).
    noscan  -- `train_step` per step, each with its own dispatch and
               `float(loss)` sync. The gap between the two IS the
               per-dispatch overhead, which is what decides whether a
               backend is usable for small models.

Data is synthetic by default (uniform random token ids), so the suite
runs on a machine with no corpus and measures compute rather than the
sampler; pass --data to drive it from a real corpus instead.

    uv run python scripts/bench_train.py                 # whole suite
    uv run python scripts/bench_train.py --max-weights 100000
    uv run python scripts/bench_train.py --source big --scan both
    uv run python scripts/bench_train.py -o results/bench-cpu.json
    uv run python scripts/bench_train.py --platform metal --system mac-metal
    uv run python scripts/bench_train.py --compare a.json b.json

-o writes both a JSON (metadata + rows, what --compare reads) and a
sibling .csv. Every CSV row repeats the machine identity, so the
files from several machines concatenate into one table.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
# torch before jax: winjax's PATH prepend otherwise breaks torch's
# bundled cudnn resolution (see texmo/conftest.py).
import torch  # noqa: F401

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax

import config
from texmo.configuration import Configuration
from texmo.dataset import DataSet
from texmo.manager import create_manager
from texmo.precision import Precision
from texmo.spec_parser import parse_model2
from texmo.tokens import set_tokens_dir


class _RandomDataSet(DataSet):
    """Synthetic sampler. Subclasses `DataSet` (without its file
    setup) only because `Manager` type-checks its dataset argument."""

    def __init__(self, ntokens: int, seed: int = 0):
        self.vocab = ntokens
        self.rng = np.random.default_rng(seed)

    def sample_tokens(self, ntokens: int, batch: int, tokenset_name: str):
        return self.rng.integers(
            0, self.vocab, size=(batch, ntokens), dtype=np.int32)


def _make_manager(entry: dict, args, steps: int):
    model = parse_model2(entry['spec'], Precision(
        args.precision or entry['precision']))
    conf = Configuration(
        model=model, lr=args.lr, length=entry['length'],
        batch=entry['batch'], steps=steps, decay=1.0, cosine=False)
    if args.data:
        dataset = DataSet(path=args.data, read_mode='pread')
    else:
        dataset = _RandomDataSet(model.codec.ntokens)
    manager = create_manager(
        'jax', conf=conf, system=args.system, dataset=dataset,
        verbose=False)
    return conf, manager


def _make_batches(manager, entry: dict, steps: int):
    """`steps` batches as one device array, sampled and transferred up
    front.

    Both timed paths take their input from here, so neither pays for
    sampling or the host->device copy inside the measurement. Timing
    one path with the data hoisted out and the other without would
    charge the per-step path for a transfer per step -- which on a GPU
    is far more expensive than the dispatch it is meant to measure.
    """
    data = manager.dataset.sample_tokens(
        ntokens=entry['length'], batch=steps * entry['batch'],
        tokenset_name=manager.model_def.input.tokens_name)
    return jax.numpy.asarray(data).reshape(
        steps, entry['batch'], entry['length'])


def _time_scan(manager, entry: dict, steps: int) -> tuple[float, float]:
    """(first-call seconds, seconds per step) for the chunked scan path.

    The warmup chunk must have the SAME shape as the timed one -- a
    chunk of a different step count is a different jit signature, so
    warming up on one would put the recompile inside the measurement.
    """
    data = _make_batches(manager, entry, steps)

    t0 = time.perf_counter()
    w, opt, losses = manager._train_chunk(
        manager.weights, manager._opt_state, data)
    jax.block_until_ready(losses)
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    w, opt, losses = manager._train_chunk(w, opt, data)
    jax.block_until_ready(losses)
    return first_s, (time.perf_counter() - t0) / steps


def _time_noscan(manager, entry: dict, steps: int) -> tuple[float, float]:
    """(first-call seconds, seconds per step) for the per-step path.

    Batches come pre-sampled and already on the device (see
    `_make_batches`), so what is timed is dispatch + compute + the
    `float(loss)` host sync -- the same work the scan path is timed
    on, once per step instead of once per chunk.
    """
    data = _make_batches(manager, entry, steps + 1)

    t0 = time.perf_counter()
    manager.train_step(data[0])
    first_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(1, steps + 1):
        # train_step syncs on float(loss), like the client's loop.
        manager.train_step(data[i])
    return first_s, (time.perf_counter() - t0) / steps


def _select(entries: list[dict], args) -> list[dict]:
    out = []
    for e in entries:
        if args.min_weights and e['weights'] < args.min_weights:
            continue
        if args.max_weights and e['weights'] > args.max_weights:
            continue
        if args.source and e['source'] not in args.source.split(','):
            continue
        if args.filter and args.filter not in e['spec'] \
                and args.filter not in e['name']:
            continue
        out.append(e)
    return out[:args.limit] if args.limit else out


def _run(args):
    set_tokens_dir(args.tokens_dir)
    with open(args.suite, encoding='utf-8') as f:
        suite = json.load(f)
    entries = _select(suite['entries'], args)

    modes = ['scan', 'noscan'] if args.scan == 'both' else [args.scan]
    device = jax.devices()[0]
    print(f'device: {device.platform}:{device.device_kind}   '
          f'system: {args.system}   entries: {len(entries)}   '
          f'modes: {",".join(modes)}   steps: {args.steps}')
    print(f'{"name":<20} {"weights":>10} {"b":>4} {"len":>5} {"mode":>6} '
          f'{"first":>8} {"ms/step":>9} {"ktok/s":>9}')

    results = []
    for entry in entries:
        for mode in modes:
            row = {**entry, 'mode': mode}
            try:
                _, manager = _make_manager(entry, args, args.steps)
                if mode == 'scan':
                    first_s, step_s = _time_scan(
                        manager, entry, args.steps)
                else:
                    first_s, step_s = _time_noscan(
                        manager, entry, args.steps)
                ktok = entry['batch'] * entry['length'] / step_s / 1000
                row.update(first_s=first_s, step_s=step_s, ktok_s=ktok)
                print(f'{entry["name"]:<20} {entry["weights"]:>10,} '
                      f'{entry["batch"]:>4} {entry["length"]:>5} '
                      f'{mode:>6} {first_s:>7.2f}s {step_s*1000:>8.2f} '
                      f'{ktok:>9.1f}')
            except Exception as exc:
                # One entry that OOMs or hits an unimplemented op must
                # not lose the rest of the sweep.
                row['error'] = f'{type(exc).__name__}: {exc}'[:200]
                print(f'{entry["name"]:<20} {entry["weights"]:>10,} '
                      f'{entry["batch"]:>4} {entry["length"]:>5} '
                      f'{mode:>6}    ERROR  {row["error"][:40]}')
            results.append(row)

    failed = [r for r in results if 'error' in r]
    if failed:
        print(f'\n{len(failed)}/{len(results)} runs failed')
    if args.out:
        meta = {
            'system': args.system,
            'platform': device.platform,
            'device_kind': device.device_kind,
            'steps': args.steps,
        }
        _write_results(args.out, meta, results)


# Flat CSV columns. Every row repeats the machine identity so files
# from several machines concatenate into one table for analysis.
_CSV_FIELDS = [
    'system', 'platform', 'device_kind', 'name', 'source', 'spec',
    'precision', 'weights', 'num_layers', 'batch', 'length', 'mode',
    'steps', 'first_s', 'step_s', 'ms_step', 'ktok_s', 'error',
]


def _write_results(out: str, meta: dict, results: list[dict]) -> None:
    """Write both a JSON (used by --compare) and a sibling CSV."""
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    stem, _ = os.path.splitext(out)
    json_path, csv_path = f'{stem}.json', f'{stem}.csv'

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({**meta, 'results': results}, f, indent=1)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            row = {**meta, **r}
            if 'step_s' in r:
                row['ms_step'] = round(r['step_s'] * 1000, 4)
            writer.writerow(row)
    print(f'wrote {json_path} and {csv_path}')


def _compare(paths: list[str]):
    """Print ms/step for the same entries across result files."""
    runs = []
    for p in paths:
        with open(p, encoding='utf-8') as f:
            runs.append(json.load(f))
    keyed = [
        {(r['name'], r['mode']): r for r in run['results']
         if 'error' not in r}
        for run in runs
    ]
    labels = [f"{run['system']}/{run['platform']}" for run in runs]
    common = set(keyed[0])
    for k in keyed[1:]:
        common &= set(k)

    head = ''.join(f'{lab[:13]:>14}' for lab in labels)
    print(f'{"name":<20} {"weights":>10} {"mode":>6}{head}   ratio')
    print(f'{"":<20} {"":>10} {"":>6}'
          + ''.join(f'{"ms/step":>14}' for _ in labels))
    for name, mode in sorted(
        common, key=lambda km: keyed[0][km]['weights']
    ):
        row = [k[(name, mode)] for k in keyed]
        cells = ''.join(f'{r["step_s"]*1000:>14.2f}' for r in row)
        ratio = row[0]['step_s'] / row[-1]['step_s']
        print(f'{name:<20} {row[0]["weights"]:>10,} {mode:>6}{cells}'
              f'   {ratio:>6.2f}x')
    print(f'\nratio = {labels[0]} / {labels[-1]} '
          f'(>1 means {labels[-1]} is faster)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='benchmarks/suite.json')
    parser.add_argument('--steps', type=int, default=20,
                        help='timed steps per entry (default: 20)')
    parser.add_argument('--scan', default='both',
                        choices=('scan', 'noscan', 'both'))
    parser.add_argument('--min-weights', type=int, default=0)
    parser.add_argument('--max-weights', type=int, default=0)
    parser.add_argument('--source', default=None,
                        help='comma-separated bands: db,mid,big')
    parser.add_argument('--filter', default=None,
                        help='substring match on spec or name')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--precision', default=None,
                        help='override the suite precision (fp32/bf16/...)')
    parser.add_argument('--lr', type=float, default=1 / 128)
    parser.add_argument('--data', default=None,
                        help='corpus to sample from (default: synthetic '
                             'random tokens, no corpus needed)')
    parser.add_argument('--tokens-dir', default=config.TOKENS_DIR)
    parser.add_argument('--system', default=config.SYSTEM_NAME)
    parser.add_argument('-o', '--out', default=None,
                        help='write results as JSON (for --compare) plus a '
                             'sibling .csv; every CSV row carries the '
                             'machine identity so files from several '
                             'machines concatenate into one table')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='compare result JSONs instead of running')
    parser.add_argument('--platform', type=str, default=None,
                        help="JAX platform(s) to use, e.g. 'cpu', 'metal' or "
                             "'metal,cpu' (a non-CPU platform needs its PJRT "
                             "plugin installed, e.g. metaljax for 'metal'). "
                             "Same effect as the JAX_PLATFORMS env var.")
    args = parser.parse_args()

    if args.platform:
        # Must land before anything touches a device.
        jax.config.update('jax_platforms', args.platform)

    if args.compare:
        _compare(args.compare)
    else:
        _run(args)


if __name__ == '__main__':
    main()
