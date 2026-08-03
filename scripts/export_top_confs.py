"""Export the global top-conf frontier as JSONL.

One line per Pareto entry from `DbReader.top_confs_global` (the same
notion as the server's top-confs page): the best-scoring conf at each
weight count, across all systems, with no train-time filter. Used to
drive external benchmarking (e.g. metaljax) on exactly the confs the
search considers the frontier.

Each line: the conf's `to_dict()` fields plus weights, median_score
and num_runs.

    uv run python scripts/export_top_confs.py \
        [--max-weights 3000] [--min-runs 2] [--db results/db.sqlite] \
        [--out benchmarks/top_confs.jsonl]
"""
import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.common import INF
from texmo.configuration import Template
from texmo.db import DbReader
from texmo.precision import Precision
from texmo.tokens import set_tokens_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-weights', type=int, default=3000)
    parser.add_argument('--min-runs', type=int, default=2)
    parser.add_argument('--db', default='results/db.sqlite')
    parser.add_argument('--out', default='benchmarks/top_confs.jsonl')
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

    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        for c in rows:
            line = dict(c.conf.to_dict())
            line['weights'] = c.conf.num_weights
            line['median_score'] = c.median_score
            line['num_runs'] = c.num_runs
            f.write(json.dumps(line) + '\n')

    print(f'wrote {args.out}: {len(rows)} confs '
          f'(weights {rows[0].conf.num_weights}..'
          f'{rows[-1].conf.num_weights}, min_runs {args.min_runs})')


if __name__ == '__main__':
    main()
