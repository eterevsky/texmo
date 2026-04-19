import argparse
import logging

from ..common import ttoa3
from ..configuration import Configuration
from ..model import build_model_def
from ..precision import Precision
from ..resultdb import ResultDB
from ..tokens import set_tokens_dir
from .timing import TrainTimingModel


def _matching_runs(db_runs, system, precision_str, spec, batch, length):
    """Find actual runs that match the query spec."""
    matches = []
    for conf, run in db_runs:
        if run.system != system:
            continue
        if str(conf.precision) != precision_str:
            continue
        if str(conf.model) != spec:
            continue
        if conf.batch != batch or conf.length != length:
            continue
        if run.train_time is None or conf.steps <= 1:
            continue
        per_step = run.train_time / (conf.steps - 1)
        matches.append(per_step)
    return matches


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    logging.info(f"Loading runs from {args.db}")
    db = ResultDB(args.db, readonly=True)
    db_runs = [(conf, run) for _, conf, run in db.get_confs_runs()]
    logging.info(f"Loaded {len(db_runs)} runs")

    model = TrainTimingModel()
    model.fit(db_runs, verbose=True)

    if not model.keys():
        logging.warning("No (system, precision) pairs had enough data to fit.")
        return

    if args.spec is None:
        return

    conf = Configuration(
        build_model_def(args.spec, precision=Precision(args.precision)),
        lr=args.lr,
        length=args.length,
        batch=args.batch,
        steps=args.steps or 100,
        decay=1.0,
    )
    per_step = model.predict(args.system, conf, batch=args.batch, length=args.length)
    steps = args.steps or conf.steps
    total = per_step * (steps - 1)

    print()
    print(f"system:    {args.system}")
    print(f"spec:      {args.spec}")
    print(
        f"shape:     batch={args.batch}, length={args.length}, precision={args.precision}"
    )
    print(f"predicted per-step:  {ttoa3(per_step)}")
    print(f"predicted {steps} steps: {ttoa3(total)}")

    # Compare to any existing runs with the same spec/shape/system/precision.
    actual = _matching_runs(
        db_runs, args.system, args.precision, args.spec, args.batch, args.length
    )
    if actual:
        import statistics

        median = statistics.median(actual)
        print()
        print(f"actual runs matching this query: {len(actual)}")
        print(f"  per-step median: {ttoa3(median)}")
        print(f"  per-step min:    {ttoa3(min(actual))}")
        print(f"  per-step max:    {ttoa3(max(actual))}")
        ratio = per_step / median
        print(f"  predicted / actual_median: {ratio:.2f}x")


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
        help=f"system name for the prediction (default: '{config.SYSTEM_NAME}')",
    )
    parser.add_argument(
        "-s",
        "--spec",
        default=None,
        help="model spec to predict (if omitted, just fits and reports losses)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        choices=["fp64", "fp32", "fp16", "bf16"],
        default="fp32",
        help="training precision (default: fp32)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of training steps",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size (default: 32)",
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
        default=128,
        metavar="NTOKENS",
        help="sample length in tokens (default: 128)",
    )

    parser.set_defaults(func=main)
