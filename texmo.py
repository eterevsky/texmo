import argparse
import os

import jax
import numpy as np

import config

# Must happen before any JAX code runs.
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platforms', config.JAX_PLATFORMS)
if not getattr(config, 'JAX_PREALLOCATE', True):
    # Read at backend initialization, so setting it here (before any
    # device use) is in time. Only the opt-out is exported; True
    # defers to JAX's default and any externally set variable.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from texmo import latency
from texmo.cli import bench, client, db, loss, report, sample, server, train
from texmo.cli import chat as chat_cli
from texmo.cli import eval as eval_cli
from texmo.cli import generate as generate_cli
from texmo.cli import time as predict_time


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
    sample.sample_init_args(parser_sample, config)

    parser_tokenize = subparsers.add_parser(
        "tokenize", help="tokenize a given text and show the tokens"
    )
    sample.tokenize_init_args(parser_tokenize, config)

    parser_train = subparsers.add_parser("train", help="train a model")
    train.init_args(parser_train, config)

    parser_benchmark_dataset = subparsers.add_parser(
        "benchmark-dataset", help="benchmark sampling the training data"
    )
    sample.benchmark_init_args(parser_benchmark_dataset, config)

    parser_benchmark_model = subparsers.add_parser(
        "benchmark-model",
        help="benchmark loading + greedy generation of a stored model",
    )
    bench.init_args(parser_benchmark_model, config)

    parser_eval = subparsers.add_parser(
        "eval",
        help="evaluate a stored model with the training-eval protocol "
             "(bits/byte on byte-fixed samples)",
    )
    eval_cli.init_args(parser_eval, config)

    parser_generate = subparsers.add_parser(
        "generate",
        help="sample a continuation of a prefix from a stored model",
    )
    generate_cli.init_args(parser_generate, config)

    parser_chat = subparsers.add_parser(
        "chat",
        help="chat with a stored model in a REPL",
    )
    chat_cli.init_args(parser_chat, config)

    parser_updatedb = subparsers.add_parser(
        "db-update",
        help="Recompute scores for all configurations in the database",
    )
    db.updatedb_init_args(parser_updatedb, config)

    parser_clear_system = subparsers.add_parser(
        "db-clear-system",
        help="Delete all runs from a given system",
    )
    db.clear_system_init_args(parser_clear_system, config)

    parser_bootstrap_estimates = subparsers.add_parser(
        "db-bootstrap-estimates",
        help=(
            "Backfill median time estimates, fit timing models, and "
            "write predicted estimates for all confs without runs"
        ),
    )
    db.bootstrap_estimates_init_args(parser_bootstrap_estimates, config)

    parser_backfill_num_layers = subparsers.add_parser(
        "db-backfill-num-layers",
        help=(
            "Populate the num_layers column for confs from before the "
            "column was added (parses spec, counts layers, updates row)"
        ),
    )
    db.backfill_num_layers_init_args(parser_backfill_num_layers, config)

    parser_strategy_stats = subparsers.add_parser(
        "strategy-stats",
        help="Per-strategy run counts and %% of runs that changed the winner",
    )
    db.strategy_stats_init_args(parser_strategy_stats, config)

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
    predict_time.init_args(parser_predict_time, config)

    parser_predict_loss = subparsers.add_parser(
        "loss",
        help="Train and evaluate a loss-prediction model",
    )
    loss.init_args(parser_predict_loss, config)

    parser_help = subparsers.add_parser("help")
    parser_help.set_defaults(func=lambda _: help(parser), parser=parser)

    args = parser.parse_args()
    if args.command is None:
        args.command = "help"
        args.func = lambda _: help(parser)

    return args


if __name__ == "__main__":
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)
    args = parse_args()
    args.func(args)

    if args.show_timers:
        latency.report()
