import argparse

from texmo.resultdb import ResultDB, import_from_csv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        default=None,
        metavar="LOG",
        help="path to a CSV file for logging",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_db = ResultDB(args.db)
    import_from_csv(record_db, args.log)
