"""Regenerate `docs/best_models.md` from the live result DB.

The Pareto frontier by weight count: for each weight value, the conf
with the lowest median loss, keeping only the ones that improve on
every lighter conf. Same notion as the server's top-confs page and as
`scripts/export_top_confs.py` (which emits the identical frontier as
JSONL for external benchmarking) -- this one renders it as the
markdown table checked into `docs/`.

Opens the DB read-only (`DbReader` -> `file:...?mode=ro`), so it is
safe to run against a DB the search server is writing to.

    uv run python scripts/make_best_models.py \
        [--max-weights 3000] [--min-runs 2] [--db results/db.sqlite] \
        [--out docs/best_models.md]
"""
import argparse
import datetime
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.common import INF, ttoa3
from texmo.configuration import Template
from texmo.db import DbReader
from texmo.precision import Precision
from texmo.tokens import set_tokens_dir

_HEADER = """# Current best models

Pareto-best configurations by weight count: for each weight count, the
conf with the lowest median loss that also beats every lighter conf.
Loss is per-byte cross-entropy on the held-out set; time is the median
wall-clock training time on the system that trains it fastest. Every LR
schedule the search runs is included -- `a↘0` is cosine-to-zero, `a→b`
exponential decay from `a` to `b`, a bare `a` a constant rate.

**A dated snapshot of a moving target.** The search adds runs
continuously, so this table is stale the moment it is written; it is
kept in the repo as a readable checkpoint, not as the source of truth.
Snapshot of {rows} rows taken on {date} (confs with at least
{min_runs} runs, up to {max_weights} weights). Regenerate with:

    uv run python scripts/make_best_models.py

Spec conventions are explained in the [README](../README.md#what-good-means-here);
the layer types are documented in [`layers.md`](layers.md) and the input
encodings in [`io.md`](io.md).

| Weights | Loss | Runs | Time | Batch×Len | LR | Steps | P | Spec |
|--------:|-----:|-----:|-----:|----------:|---:|------:|--:|------|
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-weights', type=int, default=3000)
    parser.add_argument('--min-runs', type=int, default=2)
    parser.add_argument('--db', default='results/db.sqlite')
    parser.add_argument('--out', default='docs/best_models.md')
    args = parser.parse_args()

    set_tokens_dir('tokens')
    reader = DbReader(args.db)
    template = Template(
        spec=None, precision=list(Precision), lr=(0, INF),
        length=(1, INF), batch=(1, INF), steps=(1, INF),
        max_weights=(1, INF))

    rows = list(reader.top_confs_global(
        template, max_weights=args.max_weights,
        min_num_runs=args.min_runs))
    reader.close()

    lines = []
    for c in rows:
        conf = c.conf
        # Pipes inside the spec would split the markdown cell.
        spec = str(conf.model).replace('|', '\\|')
        lines.append(
            f'| {conf.num_weights} '
            f'| {c.median_score:.4f} '
            f'| {c.num_runs} '
            # Time only, never the system name: which physical machines
            # are in the fleet is not the repo's business (the header
            # says it is whichever trains the conf fastest).
            f'| {ttoa3(c.median_time)} '
            f'| {conf.batch}×{conf.length} '
            f'| {conf.learning_str} '
            f'| {conf.steps} '
            f'| {conf.precision} '
            f'| `{spec}` |\n')

    header = _HEADER.format(
        rows=len(rows),
        date=datetime.date.today().isoformat(),
        min_runs=args.min_runs,
        max_weights=args.max_weights)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(header)
        f.writelines(lines)

    print(f'wrote {args.out}: {len(rows)} confs '
          f'(weights {rows[0].conf.num_weights}..'
          f'{rows[-1].conf.num_weights}, '
          f'loss {rows[0].median_score:.4f}..{rows[-1].median_score:.4f})')


if __name__ == '__main__':
    main()
