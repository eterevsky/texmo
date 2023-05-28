import argparse
import logging

from texmo.tokens import small_tokens
from texmo import dataset


def help(parser):
    parser.print_help()


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="command to be executed")

    parser_sample = subparsers.add_parser(
        "sample", help="generate a typical sample of training data"
    )
    dataset.init_args(parser_sample)

    parser_count_bytes = subparsers.add_parser(
        "small_tokens", help="generate a set of sub-byte tokens"
    )
    small_tokens.init_args(parser_count_bytes)

    parser_help = subparsers.add_parser("help", help="Display help information")
    parser_help.set_defaults(func=lambda _: help(parser), parser=parser)

    args = parser.parse_args()
    if args.command is None:
        args.command = "help"
        args.func = lambda _: help(parser)

    return args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    args.func(args)
