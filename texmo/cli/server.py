"""CLI entry point for the search server.

Owns argument parsing and the runtime setup that ties a `SearchServer`
to the Flask WSGI app provided by `texmo.server.serve`.
"""

import argparse
import logging

import matplotlib

from ..configuration import Template
from ..server import LOSS_REFIT_MODES, SearchServer
from ..tokens import set_tokens_dir


def main(args: argparse.Namespace):
    # Server renders graphs to bytes for the web UI, never to a display.
    matplotlib.use('Agg')
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    logging.info(f"Template: {template}")

    train_time = tuple(map(float, args.train_time.split("-")))
    logging.info(f"T ∈ {train_time} s")
    warmup_systems = [
        s.strip() for s in (args.warmup_systems or '').split(',') if s.strip()
    ]
    templates_json = None
    if args.templates:
        with open(args.templates, encoding='utf-8') as f:
            templates_json = f.read()
    server = SearchServer(
        args.db, template, train_time, args.default_spec,
        warmup_systems=warmup_systems,
        templates_json=templates_json,
        loss_refit=args.loss_refit,
    )

    server.serve(api_key=args.api_key or '')


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results, or a URL for a PostgreSQL database",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )

    # Template args
    parser.add_argument(
        "-s",
        "--spec",
        type=str,
        default=None,
        help="regex covering the acceptable specs or an exact spec",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp32,fp16,bf16",
        help="precision"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256"',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates",
    )
    parser.add_argument(
        '--decay-types',
        type=str,
        default="none,exp,cosine",
        help="comma-separated subset of LR-schedule types: "
             "none, exp, cosine (default: all)",
    )
    parser.add_argument(
        "--length",
        type=str,
        default=None,
        help="range of acceptable training sample lengths",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="range for the number of training steps; lower bound >= 2",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="1024-4294967296",
        help="range for the _maximal_ number of weights in the model",
    )
    parser.add_argument(
        "--num-layers",
        type=str,
        default=None,
        help="range for the number of intermediate layers in the model"
             " (e.g. '1-3'); counts every layer in the spec, including"
             " suffix/norm/skip",
    )
    parser.add_argument(
        "-t",
        "--train-time",
        default="1-16",
        help="range for the training time in seconds",
    )
    parser.add_argument("--default-spec", type=str, default=None, help="default model")
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="path to a JSON file splitting the select budget between"
             " several sub-searches, e.g."
             ' \'[{"name": "main", "share": 60},'
             ' {"name": "attn", "share": 40, "regex": ".*attn.*",'
             ' "default_spec": "..."}]\'. Entries share every template'
             " bound and differ only in their spec filter; shares are"
             " normalized. Omit for a single unrestricted search"
             " (the default, identical to not having the flag).",
    )
    parser.add_argument(
        "--warmup-systems",
        type=str,
        default=None,
        help="comma-separated systems to put in warmup mode: they only"
             " get confs whose steps/batch/length fit a ladder of"
             " gradually rising caps until it is climbed, so a machine"
             " or backend with no timing model yet can't be handed an"
             " hours-long run. Use for a new client or a new backend"
             " (the ladder replays on every server restart).",
    )
    parser.add_argument(
        "--loss-refit",
        type=str,
        choices=LOSS_REFIT_MODES,
        default=getattr(config, 'LOSS_REFIT', 'workers'),
        help="where loss-model refits run: 'workers' grants one job"
             " per 500 labeled runs to a client via /select,"
             " 'server' fits in-process on the model thread (which"
             " needs a fast JAX platform in the server process)."
             " Defaults to config.LOSS_REFIT.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=getattr(config, 'API_KEY', '') or '',
        help="Bearer token required on the authenticated (5001) port. "
             "Defaults to config.API_KEY. Empty disables that port "
             "(it returns 401 to everything).",
    )

    parser.set_defaults(func=main)
