import argparse
import logging

from texmo.tokens import small_tokens
from texmo import dataset

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="command to be executed")

    parser_dataset_sample = subparsers.add_parser(
        "dataset_sample", help="generate a typical sample of training data"
    )
    dataset.init_args(parser_dataset_sample)

    parser_count_bytes = subparsers.add_parser(
        "small_tokens", help="generate a set of sub-byte tokens"
    )
    small_tokens.init_args(parser_count_bytes)

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    args.func(args)
