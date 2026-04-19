"""`texmo loss` — train/eval a loss-prediction model.

Loads all scored confs from the DB, splits by conf_id % 10 (10% into
validation), runs a placeholder predictor, and reports L2-log loss on
the validation set. The placeholder returns a constant prediction so
we have a baseline to beat.
"""

import argparse
import logging

import numpy as np

from ..configuration import Configuration
from ..resultdb import ResultDB
from ..tokens import set_tokens_dir
from .predict_common import MAX_LOSS, MIN_LOSS


def _train_val_split(
    labeled: list[tuple[int, Configuration, float]],
) -> tuple[list[tuple[Configuration, float]], list[tuple[Configuration, float]]]:
    """Split by conf_id so every run of a conf ends up on the same side."""
    train, val = [], []
    for conf_id, conf, loss in labeled:
        if conf_id % 10 == 0:
            val.append((conf, loss))
        else:
            train.append((conf, loss))
    return train, val


def _eval_log_l1(
    preds_log: np.ndarray, targets: np.ndarray,
) -> float:
    """L1 in log2-space. Targets clipped to [MIN_LOSS, MAX_LOSS]; preds left
    as-is so predictions outside the range still get a gradient.

    L1 rather than L2 so the optimum is the conditional median — the
    search also uses median_score, and a configuration that's great
    half the time and diverges the other half should be rated by its
    good runs, not their average with MAX.
    """
    targets = np.clip(targets, MIN_LOSS, MAX_LOSS)
    log_targets = np.log2(targets)
    return float(np.mean(np.abs(preds_log - log_targets)))


def _format_loss(loss: float) -> str:
    """Format an L1-log2 loss with its relative-error interpretation."""
    rel = (2.0 ** loss - 1.0) * 100.0
    return f"{loss:.4f} (~{rel:.1f}% typical error)"


def _log_clip_targets(losses: np.ndarray) -> np.ndarray:
    return np.log2(np.clip(losses, MIN_LOSS, MAX_LOSS))


def _train_median_predictor(
    train_data: list[tuple[Configuration, float]],
) -> float:
    """Best-constant baseline for L1: median of log-clipped training losses."""
    train_targets = np.array([loss for _, loss in train_data])
    return float(np.median(_log_clip_targets(train_targets)))


def _within_conf_oracle(
    val_data: list[tuple[Configuration, float]],
    val_log_targets: np.ndarray,
) -> np.ndarray:
    """For each val run, predict the median log-loss of its conf's val runs.

    The L1 floor for any conf-only predictor — runs of the same conf
    are indistinguishable from the model's point of view, so the best
    we can do is the per-conf median.
    """
    by_conf: dict[Configuration, list[int]] = {}
    for i, (conf, _) in enumerate(val_data):
        by_conf.setdefault(conf, []).append(i)
    preds = np.empty(len(val_data))
    for idxs in by_conf.values():
        m = float(np.median(val_log_targets[idxs]))
        for i in idxs:
            preds[i] = m
    return preds


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    logging.info(f"Loading labeled runs from {args.db}")
    db = ResultDB(args.db, readonly=True)
    labeled = list(db.iter_labeled_runs())
    logging.info(f"Loaded {len(labeled)} labeled runs")

    train_data, val_data = _train_val_split(labeled)
    logging.info(
        f"Split: train={len(train_data)}, val={len(val_data)}"
    )

    val_targets = np.array([loss for _, loss in val_data])
    val_log_targets = _log_clip_targets(val_targets)

    # Baseline: predict the median of log-clipped training losses.
    baseline_const = _train_median_predictor(train_data)
    baseline_preds = np.full(len(val_data), baseline_const)
    baseline_loss = _eval_log_l1(baseline_preds, val_targets)
    logging.info(
        f"Baseline (predict train median = {baseline_const:.4f}): "
        f"val {_format_loss(baseline_loss)}"
    )

    # Oracle lower bound: within-conf run-to-run absolute deviation.
    oracle_preds = _within_conf_oracle(val_data, val_log_targets)
    oracle_loss = _eval_log_l1(oracle_preds, val_targets)
    logging.info(
        f"Oracle (per-conf median over val runs): "
        f"val {_format_loss(oracle_loss)}"
    )


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )
    parser.set_defaults(func=main)
