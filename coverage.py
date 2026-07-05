"""Per-system coverage report for the current global top confs.

Walks the cosine-decay Pareto front and, for each conf, prints which
of the active systems ({5060ti, m5, mini, pi5}) have a run on it.
Summary at the end: how many confs are fully covered, and the total
number of (conf, system) pairs that have ≥1 run.

Run: `uv run python coverage.py [path/to/db.sqlite]`
"""

import sys

from texmo.common import INF
from texmo.configuration import Precision, Template
from texmo.db import DbReader


SYSTEMS = ['5060ti', 'm5', 'mini', 'pi5']


def main(db_path: str):
    template = Template(
        spec=None, precision=list(Precision),
        lr=(0, INF), length=(1, INF), batch=(1, INF),
        steps=(2, INF), max_weights=(2, INF),
        decay_types=['cosine'],
    )
    db = DbReader(db_path)
    top = list(db.top_confs_global(template))

    total_pairs = 0
    fully_covered = 0
    for c in top:
        cid = db.get_conf_id(c.conf)
        counts = {s: 0 for s in SYSTEMS}
        if cid is not None:
            for s in SYSTEMS:
                sys_counts = db.get_run_counts([cid], system=s)
                _, sys_runs = sys_counts.get(cid, (0, 0))
                counts[s] = sys_runs
        present = [s for s, n in counts.items() if n > 0]
        total_pairs += len(present)
        if len(present) == len(SYSTEMS):
            fully_covered += 1
        cov = ' '.join(f'{s}:{n}' for s, n in counts.items())
        w = c.conf.model.num_weights
        print(
            f'  w={w:>5}  {c.median_score:.4f} ({c.num_runs})  '
            f'[{cov}]  {c.conf.model}'
        )

    n = len(top)
    print()
    print(f'{n} cosine top confs across {len(SYSTEMS)} systems = '
          f'{n * len(SYSTEMS)} possible (conf, system) pairs')
    print(f'  fully covered on all {len(SYSTEMS)} systems: '
          f'{fully_covered} / {n}')
    print(f'  (conf, system) pairs with >=1 run: '
          f'{total_pairs} / {n * len(SYSTEMS)}  '
          f'({100.0 * total_pairs / (n * len(SYSTEMS)):.1f}%)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/db.sqlite')
