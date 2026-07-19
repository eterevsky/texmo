"""One-off DB migration: rename `lrnn|latent|lmgu.X.Y` to
`norm-lrnn|norm-latent|norm-lmgu.X.Y`, bump `num_layers` by the
count of inserted `norm` layers, merge any rows that collide with
pre-existing `norm-X` confs (move runs, drop the duplicate conf),
and delete diverged runs for any spec mentioning
lrnn/latent/lmgu/norm (likely artifacts of the pre-Tikhonov
NaN-at-zero gradient bug, not genuine architectural failures).
Recompute median_score for confs that lost runs, and clear
conf_time_estimate for renamed confs (their timing changed by
~1 norm op per inserted layer); the server recomputes both lazily.

Run with the texmo server stopped to avoid lock contention:

    uv run python scripts/migrate_norm_rename.py

The DB at results/db.sqlite is mutated in place; back it up first.
"""
import logging
import re
import sqlite3
import statistics
import sys
import time

DB_PATH = 'results/db.sqlite'
BAD_LOSS = 1e6

# Match an occurrence of one of the three layer types within a spec,
# anchored to a layer boundary (after `|` or `-`). Idempotent: rows
# already containing `norm-lrnn.X.Y` etc. won't be rewritten because
# `repl` checks for a preceding `norm-`.
PATTERN = re.compile(r'(?<=[|\-])(lrnn|latent|lmgu)\.(\d+)\.(\d+)')

CONF_KEY_FIELDS = (
    'spec', 'precision', 'lr', 'decay', 'cosine',
    'length', 'batch', 'steps',
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)


def rename_spec(spec: str) -> tuple[str, int]:
    """Returns (new_spec, num_norms_inserted)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        start = m.start()
        if start >= 5 and spec[start - 5:start] == 'norm-':
            return m.group(0)
        count += 1
        return f'norm-{m.group(0)}'

    return PATTERN.sub(repl, spec), count


def main() -> int:
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    cur = con.cursor()

    # Step 1: index every conf by its unique key so we can detect
    # collisions when we rewrite specs.
    cur.execute(
        'SELECT id, ' + ', '.join(CONF_KEY_FIELDS) + ' FROM conf')
    key_to_id: dict[tuple, int] = {}
    for r in cur.fetchall():
        key = tuple(r[f] for f in CONF_KEY_FIELDS)
        key_to_id[key] = r['id']

    # Step 2: plan renames vs merges.
    cur.execute(
        'SELECT id, num_layers, '
        + ', '.join(CONF_KEY_FIELDS) + ' FROM conf '
        "WHERE spec LIKE '%lrnn.%' "
        "   OR spec LIKE '%latent.%' "
        "   OR spec LIKE '%lmgu.%'"
    )
    candidates = cur.fetchall()
    logging.info(f"Candidate confs to inspect: {len(candidates)}")

    renames: list[tuple[str, int | None, int]] = []
    renamed_ids: list[int] = []
    merges: list[tuple[int, int]] = []  # (source_id, target_id)
    skipped = 0
    for r in candidates:
        new_spec, inserted = rename_spec(r['spec'])
        if inserted == 0:
            skipped += 1
            continue
        new_key = (new_spec,) + tuple(
            r[f] for f in CONF_KEY_FIELDS[1:])
        existing = key_to_id.get(new_key)
        if existing is not None and existing != r['id']:
            merges.append((r['id'], existing))
            continue
        new_num_layers = (
            None if r['num_layers'] is None
            else r['num_layers'] + inserted
        )
        renames.append((new_spec, new_num_layers, r['id']))
        renamed_ids.append(r['id'])

    logging.info(
        f"Plan: {len(renames)} renames, {len(merges)} merges, "
        f"{skipped} already in `norm-` form.")

    if not renames and not merges:
        logging.info("Nothing to do.")
        return 0

    t0 = time.time()
    cur.execute('BEGIN IMMEDIATE')
    try:
        # Step 3: merges first — move runs to target, then drop the
        # source conf (cascade removes its time estimates). Track
        # target ids so we recompute their median later.
        merge_target_ids: list[int] = []
        for src_id, tgt_id in merges:
            cur.execute(
                'UPDATE run SET conf_id = ? WHERE conf_id = ?',
                (tgt_id, src_id))
            cur.execute(
                'DELETE FROM conf WHERE id = ?', (src_id,))
            merge_target_ids.append(tgt_id)
        logging.info(
            f"Merged {len(merges)} duplicate confs into existing "
            f"`norm-` siblings ({time.time() - t0:.1f}s).")

        # Step 4: apply renames.
        cur.executemany(
            'UPDATE conf SET spec = ?, num_layers = ? WHERE id = ?',
            renames,
        )
        logging.info(
            f"Renamed {len(renames)} conf rows "
            f"({time.time() - t0:.1f}s).")

        # Step 5: capture conf_ids that will lose runs to the
        # diverged-run delete (need median recompute below). Also
        # include merge targets, whose run sets changed.
        cur.execute("""
            CREATE TEMP TABLE conf_lost_runs AS
            SELECT DISTINCT r.conf_id AS id
            FROM run r JOIN conf c ON c.id = r.conf_id
            WHERE r.loss > ?
              AND (c.spec LIKE '%lrnn.%' OR c.spec LIKE '%latent.%'
                   OR c.spec LIKE '%lmgu.%' OR c.spec LIKE '%norm%')
        """, (BAD_LOSS,))
        for tgt_id in merge_target_ids:
            cur.execute(
                'INSERT OR IGNORE INTO conf_lost_runs(id) VALUES (?)',
                (tgt_id,))
        n_lost = cur.execute(
            'SELECT COUNT(*) FROM conf_lost_runs').fetchone()[0]
        logging.info(
            f"{n_lost} confs to recompute median for "
            f"(merge targets + diverged-run losers).")

        cur.execute("""
            DELETE FROM run
            WHERE loss > ?
              AND conf_id IN (
                  SELECT id FROM conf
                  WHERE spec LIKE '%lrnn.%' OR spec LIKE '%latent.%'
                     OR spec LIKE '%lmgu.%' OR spec LIKE '%norm%'
              )
        """, (BAD_LOSS,))
        deleted = cur.rowcount
        logging.info(
            f"Deleted {deleted} diverged runs (loss > {BAD_LOSS:g}).")

        # Step 6: clear time estimates for renamed confs only --
        # their spec changed, so the stored timing is stale. Other
        # affected confs (unchanged specs) keep their estimates.
        # Merge targets' specs didn't change either, but their run
        # set did; we still leave their estimates since the timing
        # they measure is per-spec, not per-run-aggregate.
        if renamed_ids:
            placeholders = ','.join('?' * len(renamed_ids))
            cur.execute(
                f'DELETE FROM conf_time_estimate WHERE conf_id IN '
                f'({placeholders})',
                renamed_ids,
            )
            cleared = cur.rowcount
            logging.info(
                f"Cleared {cleared} time-estimate rows for renamed "
                f"confs.")

        # Step 7: recompute median_score in Python (SQLite has no
        # MEDIAN()).
        cur.execute("""
            SELECT a.id, r.loss
            FROM conf_lost_runs a
            LEFT JOIN run r ON r.conf_id = a.id
            ORDER BY a.id
        """)
        losses_by_conf: dict[int, list[float]] = {}
        for row in cur.fetchall():
            losses_by_conf.setdefault(row['id'], [])
            if row['loss'] is not None:
                losses_by_conf[row['id']].append(row['loss'])
        median_updates = [
            (statistics.median(losses) if losses else None, conf_id)
            for conf_id, losses in losses_by_conf.items()
        ]
        cur.executemany(
            'UPDATE conf SET median_score = ? WHERE id = ?',
            median_updates,
        )
        no_runs_left = sum(
            1 for ms, _ in median_updates if ms is None)
        logging.info(
            f"Recomputed median_score on {len(median_updates)} confs "
            f"({no_runs_left} now NULL with no remaining runs).")

        cur.execute('DROP TABLE conf_lost_runs')
        cur.execute('COMMIT')
    except Exception:
        cur.execute('ROLLBACK')
        raise

    logging.info(
        f"Migration complete in {time.time() - t0:.1f}s.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
