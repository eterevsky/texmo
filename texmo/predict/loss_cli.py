"""`texmo loss` — train and evaluate loss-prediction models.

Loads all labeled runs from the DB, splits by conf_id % 10 so that
every run of a conf lands on the same side, trains a zoo of
predictors on the training runs, and reports their L1-log loss on the
validation runs. A couple of reference numbers (constant and oracle)
frame where each model sits on the "no information ↔ best possible"
scale.
"""

import argparse
import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

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


def _log_clip_targets(losses: np.ndarray) -> np.ndarray:
    return np.log2(np.clip(losses, MIN_LOSS, MAX_LOSS))


def _eval_log_l1(preds_log: np.ndarray, targets: np.ndarray) -> float:
    """L1 in log2-space. Targets clipped; preds left unclipped so
    predictions outside [MIN, MAX] still get a gradient."""
    log_targets = _log_clip_targets(targets)
    return float(np.mean(np.abs(preds_log - log_targets)))


def _format_loss(loss: float) -> str:
    rel = (2.0 ** loss - 1.0) * 100.0
    return f"{loss:.4f} (~{rel:.1f}% typical error)"


# -- Predictors --

class Predictor:
    name: str = None

    def fit(self, train_data: list[tuple[Configuration, float]]) -> None:
        raise NotImplementedError

    def predict(self, confs: list[Configuration]) -> np.ndarray:
        raise NotImplementedError


class ConstantPredictor(Predictor):
    """Best-constant baseline — median of log-clipped training losses."""
    name = "constant (train median)"

    def fit(self, train_data):
        targets = np.array([loss for _, loss in train_data])
        self._const = float(np.median(_log_clip_targets(targets)))

    def predict(self, confs):
        return np.full(len(confs), self._const)


def _rf_features(conf: Configuration) -> list[float]:
    return [
        np.log2(conf.num_weights),
        np.log2(conf.batch * conf.length * conf.steps),
    ]


class RandomForestPredictor(Predictor):
    """Random forest on log-weights and log(batch*length*steps).

    criterion='absolute_error' aligns leaf splits and leaf values
    (the median) with the L1-log evaluation metric.
    """
    name = "random forest (log_weights, log_data)"

    def fit(self, train_data):
        X = np.array([_rf_features(c) for c, _ in train_data])
        y = _log_clip_targets(
            np.array([loss for _, loss in train_data]))
        self._model = RandomForestRegressor(
            n_estimators=100, criterion='absolute_error',
            n_jobs=-1, random_state=0,
        )
        self._model.fit(X, y)

    def predict(self, confs):
        X = np.array([_rf_features(c) for c in confs])
        return self._model.predict(X)


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
    logging.info(f"Split: train={len(train_data)}, val={len(val_data)}")

    val_confs = [c for c, _ in val_data]
    val_targets = np.array([loss for _, loss in val_data])
    val_log_targets = _log_clip_targets(val_targets)

    predictors: list[Predictor] = [
        ConstantPredictor(),
        RandomForestPredictor(),
    ]
    for p in predictors:
        logging.info(f"Training {p.name}")
        p.fit(train_data)
        preds = p.predict(val_confs)
        logging.info(f"{p.name}: val {_format_loss(_eval_log_l1(preds, val_targets))}")

    # Oracle lower bound.
    oracle_preds = _within_conf_oracle(val_data, val_log_targets)
    logging.info(
        f"oracle (per-conf val median): val "
        f"{_format_loss(_eval_log_l1(oracle_preds, val_targets))}"
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
