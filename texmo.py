import argparse
import logging

import jax
import numpy as np
import sys

import config
from texmo import sample_cli, latency, search_cli, train_cli, resultdb
from texmo.predict import predict_cli, predictor2, sample_timing, traintiming2
from texmo.tokens import small_tokens


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

    subparsers = parser.add_subparsers(
        dest="command", help="command to be executed"
    )

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

    parser_search = subparsers.add_parser(
        "search", help="optimize model and metaparameters"
    )
    search_cli.init_args(parser_search, config)

    parser_predict = subparsers.add_parser(
        "predict", help="train loss prediction model"
    )
    predict_cli.init_args(parser_predict)

    parser_predict2 = subparsers.add_parser(
        "predict2", help="train the new loss prediction model"
    )
    predictor2.init_args(parser_predict2, config)

    parser_predict_sample = subparsers.add_parser(
        "predict-sample", help="sample timing prediction model"
    )
    sample_timing.init_args(parser_predict_sample)

    parser_predict_train = subparsers.add_parser(
        "predict-train", help="train timing prediction model"
    )
    traintiming2.init_args(parser_predict_train)

    parser_benchmark_dataset = subparsers.add_parser(
        "benchmark-dataset", help="benchmark sampling the training data"
    )
    sample_cli.benchmark_init_args(parser_benchmark_dataset, config)

    parser_importdb = subparsers.add_parser(
        "importdb", help="Import configurations and runs from one databased into another"
    )
    resultdb.importdb_init_args(parser_importdb, config)

    parser_help = subparsers.add_parser("help")
    parser_help.set_defaults(func=lambda _: help(parser), parser=parser)

    args = parser.parse_args()
    if args.command is None:
        args.command = "help"
        args.func = lambda _: help(parser)

    return args


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)
    # For timing model
    # jax.config.update("jax_enable_x64", True)
    args = parse_args()
    args.func(args)

    if args.show_timers:
        latency.report()
