import argparse

from .common import console, ttoa3
from .configuration import Template
from .resultdb import ResultDB
from .tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    console.log('Template:', template)

    db = ResultDB.from_args(args.db)
    for conf_score in db.top_confs_global(template):
        console.log(conf_score.conf)
        console.log(f'{conf_score.median_score:.3f} ({conf_score.num_runs})  {ttoa3(conf_score.median_time)} on {conf_score.system}')


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

    # Template args
    parser.add_argument(
        '-s',
        '--spec-regex',
        type=str,
        default=None,
        help='regex covering the acceptable specs',
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
        "--optimizer",
        type=str,
        metavar="O",
        default="adam,fromage",
        help="the optimizer algorithm",
    )
    parser.add_argument(
        '--decay',
        type=str,
        default="0.0-1.0",
        help="decay of the learning rate over the course of training, i.e. (LR at the last step) / (LR at the first step)",
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
