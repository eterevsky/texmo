import argparse
# import logging

import numpy as np
from rich.logging import RichHandler
import jax

import config
from texmo import (
    sample_cli,
    latency,
    train_cli,
    db_cli,
    report,
    server,
    client,
)
from texmo.common import console
from texmo.tokens import small_tokens
from texmo.predict import time_cli


# Need to do this before any JAX code is executed.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", config.JAX_PLATFORMS)

def help(parser):
    parser.print_help()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-timers",
        default=False,
        action="store_true",
        help="show runtime timers",
    )

    subparsers = parser.add_subparsers(dest="command", help="command to be executed")

    parser_sample = subparsers.add_parser(
        "sample", help="generate a typical sample of training data"
    )
    sample_cli.sample_init_args(parser_sample, config)

    parser_count_bytes = subparsers.add_parser(
        "small-tokens", help="generate a set of sub-byte tokens"
    )
    small_tokens.init_args(parser_count_bytes)

    parser_train = subparsers.add_parser("train", help="train a model")
    train_cli.init_args(parser_train, config)

    parser_benchmark_dataset = subparsers.add_parser(
        "benchmark-dataset", help="benchmark sampling the training data"
    )
    sample_cli.benchmark_init_args(parser_benchmark_dataset, config)

    parser_importdb = subparsers.add_parser(
        "importdb",
        help="Import configurations and runs from one databased into another",
    )
    db_cli.importdb_init_args(parser_importdb, config)

    parser_updatedb = subparsers.add_parser(
        "updatedb", help="Check databased consistency and update scores and neighbors"
    )
    db_cli.updatedb_init_args(parser_updatedb, config)

    parser_report = subparsers.add_parser(
        "report", help="Print a report of the results in the database"
    )
    report.init_args(parser_report, config)

    parser_server = subparsers.add_parser("server", help="Start the search server")
    server.init_args(parser_server, config)

    parser_client = subparsers.add_parser(
        "client",
        help="Start a client that will run configurations provided by the server",
    )
    client.init_args(parser_client, config)

    parser_predict_time = subparsers.add_parser(
        "time", help="Predict the time that it will take to train a configuration"
    )
    time_cli.init_args(parser_predict_time, config)

    parser_help = subparsers.add_parser("help")
    parser_help.set_defaults(func=lambda _: help(parser), parser=parser)

    args = parser.parse_args()
    if args.command is None:
        args.command = "help"
        args.func = lambda _: help(parser)

    return args


if __name__ == "__main__":
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     handlers=[RichHandler(console=console, show_level=False)]
    # )

    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)
    args = parse_args()
    args.func(args)

    if args.show_timers:
        latency.report()
