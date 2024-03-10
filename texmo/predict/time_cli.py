import argparse

from ..tokens import set_tokens_dir


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    print(args)


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        metavar="PATH",
        default=config.TOKENS_DIR,
        help=f"directory with token sets (default: '{config.TOKENS_DIR}')",
    )
    parser.add_argument(
        "--db",
        type=str,
        metavar="PATH",
        default=config.DB,
        help=f"path to the SQLite database with the results (default: {config.DB})",
    )

    parser.add_argument(
        "--system",
        type=str,
        default=config.SYSTEM_NAME,
        help=f"the name of the system that will be used to identify runs "
        + "in the DB (default: '{config.SYSTEM_NAME}')",
    )
    parser.add_argument(
        "-s",
        "--spec",
        default=None,
        help="layer-by-layer model specification",
    )
    parser.add_argument(
        "--steps", type=int,
        default=None,
        help="number of training steps"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=None,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0625,
        help="learning rate (doesn't affect time prediction)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=None,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training",
    )

    parser.set_defaults(func=main)
