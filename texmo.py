import argparse
import logging

from texmo.tokens import small_tokens
from texmo import dataset, search_cli, train_cli, latency


def help(parser):
    parser.print_help()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-timers",
        default=False,
        action="store_true",
        help="show runtime timers"
    )

    subparsers = parser.add_subparsers(dest="command", help="command to be executed")

    parser_sample = subparsers.add_parser(
        "sample", help="generate a typical sample of training data"
    )
    dataset.init_args(parser_sample)

    parser_count_bytes = subparsers.add_parser(
        "small-tokens", help="generate a set of sub-byte tokens"
    )
    small_tokens.init_args(parser_count_bytes)

    parser_train = subparsers.add_parser(
        "train", help="train a model"
    )
    train_cli.init_args(parser_train)

    parser_search = subparsers.add_parser(
        "search", help="optimize model and metaparameters"
    )
    search_cli.init_args(parser_search)

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

    if args.show_timers:
        latency.report()
