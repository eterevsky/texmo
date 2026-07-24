"""Build benchmarks/suite.json: a fixed set of training benchmarks.

Two sources, so the suite covers both what the search actually likes
and sizes the DB has barely explored:

  db         -- winning configurations sampled off the global Pareto
                front, spread over weight decades and layer counts, so
                the mix of layer types is the one the search converged
                on. Covers ~1 to ~30k weights.
  mid / big  -- hand-written 30k-1M and 1-10M weight models. The DB is
                almost all tiny models, but backend performance
                (dispatch overhead vs raw matmul throughput) inverts
                with size: the mid band is where a CPU backend that
                wins on tiny models hands over to a GPU one, so it
                needs to be sampled densely rather than jumped over.

Each architecture is emitted at two (batch, length) shapes drawn from
a rotation, so the suite also spans the shape axis without the entry
count exploding. Scan on/off is a runner flag, not a suite entry.

The output is CHECKED IN: cross-machine numbers are only comparable if
every machine runs an identical list. Regenerate deliberately, not as
a side effect of DB growth.

Usage:
    uv run python scripts/make_bench_suite.py [--db results/db.sqlite]
"""
import argparse
import os
import sys
from collections import defaultdict

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.common import INF
from texmo.configuration import Template
from texmo.db import DbReader
from texmo.pjson import save_json
from texmo.precision import Precision
from texmo.spec_parser import parse_model2

# (batch, length) shapes, rotated across architectures. Spans the
# short-and-wide to long-and-narrow axis that decides whether a
# backend is dispatch-bound or throughput-bound.
_SHAPES = [(16, 128), (64, 256), (256, 512), (4, 1024), (128, 128)]

# Bigger models get smaller shapes: BPTT keeps every step's
# activations, so batch*length*width dominates memory once the model
# is hundreds of thousands of weights.
_MID_SHAPES = [(16, 256), (64, 128)]
_BIG_SHAPES = [(8, 256), (32, 128)]

# Candidate architectures past the searched range. Weights are
# computed with the real parser below and each candidate is filed
# into the band it lands in (or dropped), so this list can be edited
# freely without hand-arithmetic.
_BANDS = {'mid': (30_000, 1_000_000), 'big': (1_000_000, 10_000_000)}
_SYNTH_CANDIDATES = [
    # -- mid band: where a dispatch-bound backend hands over ----------
    'bits.4.oh+bp|gru.64',
    'bits.4.oh+bp|gru.128',
    'bits.4.oh+bp|lstm.128',
    'bytes.emb.128|gru.128',
    'bytes.emb.128|rnn.128.tanh',
    'bytes.emb.256|gru.256',
    'bytes.emb.256|mgru.256',
    'bytes.emb.256|lstm.256',
    'bytes.emb.128|gru.128-gru.128',
    'bytes.emb.256|conv.4-gru.256',
    'bytes.emb.256|split.add(gru.256, pass)',
    'bytes.emb.256|attn.256.4.64',
    'bytes.emb.256|msr.64.4',
    'bytes.emb.512|lrnn.512.2',
    'bytes.emb.512|attn.512.8.64',
    # Pre-norm transformer block, x + Attn(Norm(x)) then
    # x + MLP(Norm(x)) with a 4x inner width -- the one shape the
    # search never proposes (its priors favour recurrence) but every
    # backend is tuned for, so it belongs in a cross-backend suite.
    'bytes.emb.64|split.add(rmsnorm-attn.64.4.16, pass)'
    '-split.add(rmsnorm-dense.256.gelu-dense.64, pass)',
    # -- big band ----------------------------------------------------
    # Two stacked transformer blocks.
    'bytes.emb.512|split.add(rmsnorm-attn.512.8.64, pass)'
    '-split.add(rmsnorm-dense.2048.gelu-dense.512, pass)'
    '-split.add(rmsnorm-attn.512.8.64, pass)'
    '-split.add(rmsnorm-dense.2048.gelu-dense.512, pass)',
    'bytes.emb.512|gru.512',
    'bytes.emb.1024|gru.1024',
    'bytes.emb.512|lstm.512',
    'bytes.emb.1024|lstm.1024',
    'bytes.emb.1024|rnn.1024.tanh',
    'bytes.emb.512|mgru.512',
    'bytes.emb.512|gru.512-gru.512',
    'bytes.emb.512|dense.512.gelu-gru.512',
    'bytes.emb.512|split.add(gru.512, pass)',
    'bytes.emb.512|conv.4-gru.512',
    'bytes.emb.1024|msr.256.4',
    'bytes.emb.512|gru.512-rmsnorm',
    'bytes.emb.256|gru.256-gru.256-gru.256',
    'bytes|gru.512',
    'bytes|lstm.1024',
    'bits.4.oh+bp|gru.1024',
]

# How many DB architectures to take, and the ceiling per
# (weight decade, layer count) bucket so one crowded decade can't
# dominate.
_DB_ARCHS = 20
_PER_BUCKET = 3


def _weight_bucket(w: int) -> int:
    return w.bit_length()  # log2 decade


def _layer_bucket(n: int) -> int:
    return min(n, 4)  # 0,1,2,3,4+


def _db_architectures(path: str) -> list[dict]:
    """Sample winning confs off the global Pareto front, spread over
    (weight decade, layer count) buckets."""
    template = Template(
        spec=None, precision=[Precision.FP32],
        lr=(0, INF), length=(1, INF), batch=(1, INF),
        steps=(2, INF), max_weights=(1, INF),
    )
    with DbReader(path) as reader:
        top = list(reader.top_confs_global(template, min_num_runs=2))
    print(f'{len(top)} confs on the global Pareto front')

    by_bucket: dict[tuple[int, int], list] = defaultdict(list)
    for c in top:
        model = c.conf.model
        key = (_weight_bucket(model.num_weights),
               _layer_bucket(model.num_layers))
        by_bucket[key].append(c)

    # Round-robin over buckets so every (size, depth) region gets a
    # turn before any bucket takes a second slot.
    candidates, seen_specs = [], set()
    for rank in range(_PER_BUCKET):
        for key in sorted(by_bucket):
            bucket = by_bucket[key]
            if rank >= len(bucket):
                continue
            c = bucket[rank]
            spec = str(c.conf.model)
            if spec in seen_specs:
                continue
            seen_specs.add(spec)
            candidates.append({
                'spec': spec,
                'precision': str(c.conf.precision),
                'weights': c.conf.model.num_weights,
                'num_layers': c.conf.model.num_layers,
                'source': 'db',
            })
    # Subsample EVENLY over the weight-sorted list. Truncating the
    # round-robin instead would cut whichever buckets sort last --
    # i.e. drop the biggest models, the ones least like the rest of
    # the suite.
    candidates.sort(key=lambda a: a['weights'])
    if len(candidates) <= _DB_ARCHS:
        return candidates
    step = len(candidates) / _DB_ARCHS
    return [candidates[min(int(i * step), len(candidates) - 1)]
            for i in range(_DB_ARCHS)]


def _synthetic_architectures() -> dict[str, list[dict]]:
    """Parse the candidates and file each into its weight band."""
    out: dict[str, list[dict]] = {band: [] for band in _BANDS}
    for spec in _SYNTH_CANDIDATES:
        try:
            model = parse_model2(spec, Precision.FP32)
        except Exception as exc:
            print(f'  skip {spec}: {exc}')
            continue
        if not model.is_valid():
            # Usually a width mismatch the codec rejects (msr's output
            # is heads*dim, not its first argument) or a leading norm.
            # Benchmarking a spec the search can never propose would
            # measure a shape nothing else in the fleet runs.
            print(f'  skip {spec}: parses but is_valid() is False')
            continue
        w = model.num_weights
        band = next(
            (b for b, (lo, hi) in _BANDS.items() if lo <= w <= hi), None)
        if band is None:
            print(f'  skip {spec}: {w:,} weights outside every band')
            continue
        out[band].append({
            'spec': str(model),
            'precision': 'fp32',
            'weights': w,
            'num_layers': model.num_layers,
            'source': band,
        })
    for band, archs in out.items():
        archs.sort(key=lambda a: a['weights'])
    return out


def _entries(archs: list[dict], shapes: list[tuple[int, int]]) -> list[dict]:
    """Two shapes per architecture, rotating through `shapes`."""
    entries = []
    for i, arch in enumerate(archs):
        for k in range(2):
            batch, length = shapes[(i + k) % len(shapes)]
            entries.append({
                'name': f'{arch["source"]}{i:02d}-b{batch}l{length}',
                'spec': arch['spec'],
                'precision': arch['precision'],
                'batch': batch,
                'length': length,
                'weights': arch['weights'],
                'num_layers': arch['num_layers'],
                'source': arch['source'],
            })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default='results/db.sqlite')
    parser.add_argument('--out', default='benchmarks/suite.json')
    args = parser.parse_args()

    db_archs = _db_architectures(args.db)
    print(f'{len(db_archs)} architectures sampled from the DB')
    synth = _synthetic_architectures()
    for band, archs in synth.items():
        print(f'{len(archs)} synthetic {band}-band architectures kept')

    entries = (_entries(db_archs, _SHAPES)
               + _entries(synth['mid'], _MID_SHAPES)
               + _entries(synth['big'], _BIG_SHAPES))
    doc = {
        'entries': entries,
        'shapes': [list(s) for s in _SHAPES],
        'mid_shapes': [list(s) for s in _MID_SHAPES],
        'big_shapes': [list(s) for s in _BIG_SHAPES],
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        save_json(doc, f)

    ws = sorted(e['weights'] for e in entries)
    print(f'\nwrote {args.out}: {len(entries)} entries')
    print(f'weights: {ws[0]:,} .. {ws[-1]:,} (median {ws[len(ws)//2]:,})')
    for source in ('db', 'mid', 'big'):
        n = sum(1 for e in entries if e['source'] == source)
        print(f'  {source}: {n} entries')


if __name__ == '__main__':
    main()
