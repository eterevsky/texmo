import argparse
import logging

from ..predict.model_thread import bootstrap as bootstrap_timing
from ..resultdb import ResultDB


def updatedb(args: argparse.Namespace):
    db = ResultDB.from_args(args.db)
    db.update_all_scores()


def updatedb_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the database",
    )

    parser.set_defaults(func=updatedb)


def clear_system(args: argparse.Namespace):
    db = ResultDB.from_args(args.db)
    systems = db.get_systems()
    if args.system not in systems:
        logging.warning(
            f"System {args.system!r} not found in the database. "
            f"Known systems: {systems}")
        return
    deleted = db.clear_system(args.system)
    logging.info(f"Deleted {deleted} runs from system {args.system!r}")


def clear_system_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the database",
    )
    parser.add_argument(
        "system",
        type=str,
        help="name of the system whose runs will be deleted",
    )

    parser.set_defaults(func=clear_system)


def bootstrap_estimates(args: argparse.Namespace):
    db = ResultDB.from_args(args.db)
    bootstrap_timing(db)


def bootstrap_estimates_init_args(
    parser: argparse.ArgumentParser, config
):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the database",
    )
    parser.set_defaults(func=bootstrap_estimates)


def strategy_stats(args: argparse.Namespace):
    """Per-strategy effectiveness: runs, winner-changes, %."""
    db = ResultDB(args.db, readonly=True)
    try:
        cur = db._db.execute(
            """
            SELECT COALESCE(strategy, '(none)') AS strategy,
                   COUNT(*) AS runs,
                   SUM(CASE WHEN changed_winner IS NULL THEN 1 ELSE 0 END)
                       AS untracked,
                   SUM(CASE WHEN changed_winner = 1 THEN 1 ELSE 0 END)
                       AS changes
            FROM run
            GROUP BY COALESCE(strategy, '(none)')
            ORDER BY runs DESC
            """
        )
        rows = list(cur)
    finally:
        db.close()

    print(f'{"strategy":<16} {"runs":>8} {"tracked":>8} {"changes":>8} '
          f'{"pct":>6}')
    for strategy, runs, untracked, changes in rows:
        tracked = runs - untracked
        pct = (100.0 * changes / tracked) if tracked > 0 else 0.0
        print(f'{strategy:<16} {runs:>8} {tracked:>8} {changes:>8} '
              f'{pct:>5.1f}%')


def strategy_stats_init_args(
    parser: argparse.ArgumentParser, config
):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the database",
    )
    parser.set_defaults(func=strategy_stats)
