import argparse

from ..configuration import Template
from ..report import format_top_conf_row
from ..resultdb import ResultDB
from ..tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    print(f'Template: {template}')

    db = ResultDB.from_args(args.db)
    print('Top configurations:')
    for c in db.top_confs_global(template):
        print(format_top_conf_row(c, with_system=True))


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results, or a URL for a PostgreSQL database",
    )
    parser.add_argument(
        '--tokens-dir',
        type=str,
        default=config.TOKENS_DIR,
        help='directory with tokensets',
    )

    # Template args (matching server CLI / Template.from_args).
    parser.add_argument(
        '-s',
        '--spec',
        type=str,
        default=None,
        help='exact spec or regex over acceptable specs',
    )
    parser.add_argument(
        '--length',
        type=str,
        default=None,
        help='range of acceptable training sample lengths',
    )
    parser.add_argument(
        '-b',
        '--batch',
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256"',
    )
    parser.add_argument(
        '--decay-types',
        type=str,
        default="none,exp,cosine",
        help="comma-separated subset of LR-schedule types: "
             "none, exp, cosine (default: all)",
    )
    parser.add_argument(
        '-l',
        '--lr',
        type=str,
        default=None,
        help='range of acceptable learning rates',
    )
    parser.add_argument(
        '--steps',
        type=str,
        default=None,
        help='range for the number of training steps; lower bound >= 2',
    )
    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        default='2-4294967296',
        help='range for the _maximal_ number of weights in the model',
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="bf16,fp16,fp32",
        help="precision"
    )


    parser.set_defaults(func=main)
