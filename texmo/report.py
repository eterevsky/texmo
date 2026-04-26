import argparse

from rich.table import Table

from .common import console, itoa3, ttoa3
from .configuration import Template
from .resultdb import ResultDB
from .tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    console.log('Template:', template)

    db = ResultDB.from_args(args.db)
    table = Table(title="Top configurations")
    table.add_column("Model", overflow='fold')
    table.add_column("P")
    table.add_column("Data", justify='right')
    table.add_column("LR")
    table.add_column("Steps", justify='right')
    table.add_column("Score")
    table.add_column("Time", justify='right')

    for c in db.top_confs_global(template):
        conf = c.conf
        spec = f'{conf.model} ({itoa3(conf.model.num_weights)})'
        score = f'{c.median_score:.3f} ({c.num_runs})'
        time = (
            f'{ttoa3(c.median_time)} on {c.system}'
            if c.median_time is not None else '?'
        )
        table.add_row(
            spec, str(conf.precision), f'{conf.batch}×{conf.length}',
            conf.learning_str, str(conf.steps), score, time,
        )

    console.print(table)


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
