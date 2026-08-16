"""Per-system coverage report for the current global top confs.

Walks the cosine-decay Pareto front and, for each conf, prints which
of the active systems have a run on it. Active = systems with a run
among the most recent `_RECENT_RUNS` runs, excluding the `other`
invalidation bucket; pass an explicit comma-separated list as the
second argument to override.

Run: `uv run python scripts/coverage.py [path/to/db.sqlite] [sys1,sys2,...]`
"""

import os
import sqlite3
import sys

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.common import INF
from texmo.configuration import Precision, Template
from texmo.db import DbReader
from texmo.tokens import set_tokens_dir

_RECENT_RUNS = 20000


def _active_systems(db_path: str) -> list[str]:
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT system FROM run
            WHERE id > (SELECT MAX(id) - ? FROM run)
              AND system != 'other'
            ORDER BY system
            """,
            (_RECENT_RUNS,),
        ).fetchall()
    finally:
        con.close()
    return [s for (s,) in rows]


def main(db_path: str, systems: list[str] | None = None):
    set_tokens_dir('tokens')
    SYSTEMS = systems or _active_systems(db_path)
    print(f'systems: {", ".join(SYSTEMS)}')
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
    main(
        sys.argv[1] if len(sys.argv) > 1 else 'results/db.sqlite',
        sys.argv[2].split(',') if len(sys.argv) > 2 else None,
    )
