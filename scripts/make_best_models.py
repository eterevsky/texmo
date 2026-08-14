"""Regenerate `docs/best_models.md` and the README frontier plot
from the live result DB.

The Pareto frontier by weight count: for each weight value, the conf
with the lowest median loss, keeping only the ones that improve on
every lighter conf. Same notion as the server's top-confs page and as
`scripts/export_top_confs.py` (which emits the identical frontier as
JSONL for external benchmarking) -- this one renders it as the
markdown table checked into `docs/` plus the log-log plot embedded in
the README, with a power-law fit of b/B against the weight count.

Opens the DB read-only (`DbReader` -> `file:...?mode=ro`), so it is
safe to run against a DB the search server is writing to.

    uv run python scripts/make_best_models.py \
        [--max-weights 10000] [--min-runs 2] [--db results/db.sqlite] \
        [--out docs/best_models.md] [--plot docs/images/graph.png]
"""
import argparse
import datetime
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def _write_plot(rows, path: str) -> tuple[float, float, float]:
    """Frontier step plot with a log-log power-law fit.

    Returns (coefficient, exponent, r2) for `b/B ~ coef * w**exponent`;
    r2 is measured in log space, where the fit is linear.
    """
    w = np.array([c.conf.num_weights for c in rows], dtype=float)
    y = np.array([c.median_score for c in rows], dtype=float)
    exponent, intercept = np.polyfit(np.log2(w), np.log2(y), 1)
    pred = intercept + exponent * np.log2(w)
    resid = np.log2(y) - pred
    r2 = 1.0 - float(np.sum(resid**2)
                     / np.sum((np.log2(y) - np.log2(y).mean())**2))
    coef = 2.0**intercept

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.step(w, y, where='post', color='#2f6fb0', lw=1.8,
            label='Pareto frontier')
    xs = np.array([w[0], w[-1]])
    ax.plot(
        xs, coef * xs**exponent, '--', color='#555555', lw=1.4,
        label=(f'$b/B \\approx {coef:.2f}\\,w^{{{exponent:.3f}}}$'
               f'  ($R^2 = {r2:.3f}$)'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_yticks([3, 4, 5, 6, 7])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel('weights')
    ax.set_ylabel('cross-entropy, bits/byte')
    ax.grid(True, which='major', color='#e5e5e5', lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    return coef, exponent, r2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-weights', type=int, default=10000)
    parser.add_argument('--min-runs', type=int, default=2)
    parser.add_argument('--db', default='results/db.sqlite')
    parser.add_argument('--out', default='docs/best_models.md')
    parser.add_argument(
        '--plot', default='docs/images/graph.png',
        help="frontier plot output; '' skips it")
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

    if args.plot:
        coef, exponent, r2 = _write_plot(rows, args.plot)
        halving = 2.0**exponent
        print(f'wrote {args.plot}: b/B ~ {coef:.3f} * w^{exponent:.4f} '
              f'(R2={r2:.4f} in log space; '
              f'x{halving:.4f} b/B per weight doubling)')


if __name__ == '__main__':
    main()
