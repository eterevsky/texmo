import argparse

from .resultdb import ResultDB
from .tokens import set_tokens_dir


def importdb(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    db_from = ResultDB.from_args(args.db_from)
    db_into = ResultDB.from_args(args.db_into)

    present = {}
    new = {}

    for _, conf, run, timestamp in db_from.get_confs_runs(with_timestamps=True):
        # print(run.system, conf, run.loss, timestamp)
        assert timestamp is not None
        assert run.system is not None
        exists = db_into.check_run_exists(conf, run, timestamp)
        if exists:
            present[run.system] = present.get(run.system, 0) + 1
        else:
            new[run.system] = new.get(run.system, 0) + 1
            print(run.system, conf, run.loss, timestamp)

            db_into.add_run(conf, run, timestamp=timestamp, commit=False)

    print("Present:", present)
    print("New:", new)

    db_into.commit()


def importdb_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )

    parser.add_argument(
        "--db-from",
        type=str,
        required=True,
        help="path to the database that will be imported",
    )

    parser.add_argument(
        "--db-into",
        type=str,
        default=config.DB,
        help="databased to which the results will be copied",
    )

    parser.set_defaults(func=importdb)


def updatedb(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    db = ResultDB.from_args(args.db)


def updatedb_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the database that will be imported",
    )

    parser.set_defaults(func=importdb)
