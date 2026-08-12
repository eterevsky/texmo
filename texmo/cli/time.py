import argparse
import logging
import math
import statistics
import time

from ..common import ttoa3
from ..configuration import Configuration
from ..db import DbReader
from ..precision import Precision
from ..predict.timing import TrainTimingModel
from ..spec_parser import parse_model2
from ..tokens import set_tokens_dir


def _matching_runs(pair_runs, spec, batch, length):
    """Per-step times of runs matching the query spec/shape.

    `pair_runs` is already filtered to the queried (system, precision).
    """
    matches = []
    for conf, run in pair_runs:
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

    db = DbReader(args.db)

    # Same shape as the server's bootstrap (model_thread.bootstrap):
    # per-pair reads capped to the most recent FIT_RUNS_CAP runs.
    # Fitting on full history instead stopped being viable once the
    # run table passed ~1M rows.
    model = TrainTimingModel()
    for system in db.get_systems():
        for precision in Precision:
            t0 = time.perf_counter()
            runs = list(db.get_runs_for_timing(
                system, precision, limit=TrainTimingModel.FIT_RUNS_CAP))
            if not runs:
                continue
            read_t = time.perf_counter() - t0
            t0 = time.perf_counter()
            model.fit(runs, verbose=False)
            fit_t = time.perf_counter() - t0
            key = (system, precision)
            if model.has_weights(system, precision):
                rmse = math.sqrt(model.loss(system, precision))
                print(
                    f"fit {key}: {len(runs)} runs "
                    f"(read {read_t:.1f}s, fit {fit_t:.1f}s), "
                    f"RMSE={rmse:.3f}s"
                )
            else:
                print(
                    f"skipping {key}: <{TrainTimingModel.MIN_SAMPLES} "
                    f"usable of {len(runs)} runs"
                )

    if not model.keys():
        logging.warning("No (system, precision) pairs had enough data to fit.")
        return

    if args.spec is None:
        return

    conf = Configuration(
        parse_model2(args.spec, precision=Precision(args.precision)),
        lr=args.lr,
        length=args.length,
        batch=args.batch,
        steps=args.steps or 100,
        decay=1.0,
    )
    per_step = model.predict_step_time(args.system, conf)
    total = model.predict(args.system, conf)
    if per_step is None or total is None:
        logging.warning(
            f"No timing model for ({args.system!r}, {args.precision}). "
            f"Need at least {TrainTimingModel.MIN_SAMPLES} runs to fit."
        )
        return
    steps = conf.steps

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
        db.get_runs_for_timing(args.system, Precision(args.precision)),
        args.spec, args.batch, args.length,
    )
    if actual:
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
