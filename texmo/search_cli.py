import argparse


def search(args: argparse.Namespace):
    pass


def init_args(parser: argparse.ArgumentParser):
    # Data
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="a file with training data",
    )

    # Template args
    parser.add_argument(
        "--tokens-type",
        type=str,
        default=None,
        help="regex"
    )
    parser.add_argument(
        "--ntokens",
        type=str,
        default="2-64",
        help="number of tokens in a token set"
    )
    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex covering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default="1-8192",
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default="0.000001-10",
        help="range of acceptable learning rates (default: 0.001-10)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="2-1024",
        help="range of acceptable sample lens (default: 128)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined (default: unrestricted)",
    )

    parser.set_defaults(func=search)
