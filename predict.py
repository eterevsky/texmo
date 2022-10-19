import argparse

from texmo.resultdb import ResultDB


def parse_args():
    parser = argparse.ArgumentParser()

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
