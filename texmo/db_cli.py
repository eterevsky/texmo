import argparse
import logging

from .model_training_thread import bootstrap as bootstrap_timing
from .resultdb import ResultDB


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
