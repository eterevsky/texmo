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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from ..configuration import Configuration
from ..layers.skip import SkipDef
from ..resultdb import ResultDB
from ..tokens import set_tokens_dir
from . import loss_rnn
from .predict_common import (
    MAX_LOSS,
    MIN_LOSS,
    discover_type_ids,
    layer_type_id,
)


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


def _rf_big_features(
    conf: Configuration,
    type_ids: list[str],
    missing_sentinel: float = -1.0,
    with_second_layer: bool = False,
) -> list[float]:
    log_decay = np.log2(conf.decay)
    out_size = conf.model.output.size

    # First and second non-skip layers.
    non_skip = [l for l in conf.model.layers if not isinstance(l, SkipDef)]
    first_non_skip = non_skip[0] if len(non_skip) > 0 else None
    second_non_skip = non_skip[1] if len(non_skip) > 1 else None

    counts = {t: 0 for t in type_ids}
    first_size = {t: missing_sentinel for t in type_ids}
    second_size = {t: missing_sentinel for t in type_ids}
    for layer in conf.model.layers:
        t = layer_type_id(layer)
        if t in counts:
            counts[t] += 1
    if first_non_skip is not None:
        t = layer_type_id(first_non_skip)
        if t in first_size:
            first_size[t] = float(np.log2(max(first_non_skip.size, 1)))
    if second_non_skip is not None:
        t = layer_type_id(second_non_skip)
        if t in second_size:
            second_size[t] = float(np.log2(max(second_non_skip.size, 1)))

    skip_add_sum = sum(
        layer.distance
        for layer in conf.model.layers
        if isinstance(layer, SkipDef) and layer.op == 'add'
    )
    skip_cat_sum = sum(
        layer.distance
        for layer in conf.model.layers
        if isinstance(layer, SkipDef) and layer.op == 'cat'
    )

    # Distance of skip.X.{add,cat} at raw positions 0 and 1, or 0 if
    # that position isn't a skip (or doesn't exist).
    def _skip_at(pos: int, op: str) -> float:
        if pos >= len(conf.model.layers):
            return 0.0
        layer = conf.model.layers[pos]
        if isinstance(layer, SkipDef) and layer.op == op:
            return float(layer.distance)
        return 0.0

    base = [
        np.log2(conf.num_weights),
        np.log2(conf.batch * conf.length * conf.steps),
        np.log2(conf.steps),
        np.log2(conf.batch),
        np.log2(conf.length),
        np.log2(conf.lr),
        log_decay,
        np.log2(max(out_size, 1)),
        float(skip_add_sum),
        float(skip_cat_sum),
    ]
    if with_second_layer:
        base.extend([
            _skip_at(0, 'add'),
            _skip_at(0, 'cat'),
            _skip_at(1, 'add'),
            _skip_at(1, 'cat'),
        ])
    for t in type_ids:
        base.append(first_size[t])
        base.append(float(counts[t]))
        if with_second_layer:
            base.append(second_size[t])
    return base


class RandomForestBigPredictor(Predictor):
    """Random forest with per-layer-type features in addition to shape.

    Feature set: log of shape params (weights, batch, length, steps,
    lr, decay, output_size, batch*length*steps), skip-distance sums by
    op, and per-layer-type pairs (first-layer-size-or-sentinel, count).
    """
    name = "random forest (big features)"

    def fit(self, train_data):
        self._type_ids = discover_type_ids(train_data)
        X = np.array(
            [_rf_big_features(c, self._type_ids) for c, _ in train_data])
        y = _log_clip_targets(
            np.array([loss for _, loss in train_data]))
        self._model = RandomForestRegressor(
            n_estimators=100, criterion='absolute_error',
            n_jobs=-1, random_state=0,
        )
        self._model.fit(X, y)

    def predict(self, confs):
        X = np.array([_rf_big_features(c, self._type_ids) for c in confs])
        return self._model.predict(X)


class HistGBRPredictor(Predictor):
    """Histogram gradient boosting on the big-feature set.

    "First layer is not of this type" is encoded as NaN;
    HistGradientBoostingRegressor learns a routing direction per split
    for missing values. loss='absolute_error' aligns with the L1-log
    eval metric.

    `with_second_layer=True` extends the feature set with:
      * second-non-skip-layer size per type (like first_size, but for
        the second non-skip layer);
      * skip.add / skip.cat distance at raw positions 0 and 1
        (4 extra slots — X or 0 if that position isn't a skip of that op).
    """

    def __init__(self, with_second_layer: bool = False):
        self._with_second_layer = with_second_layer
        suffix = " + 2nd layer" if with_second_layer else ""
        self.name = f"hist gbr (big features{suffix})"

    def fit(self, train_data):
        self._type_ids = discover_type_ids(train_data)
        X = np.array([
            _rf_big_features(
                c, self._type_ids, float('nan'), self._with_second_layer)
            for c, _ in train_data
        ])
        y = _log_clip_targets(
            np.array([loss for _, loss in train_data]))
        self._model = HistGradientBoostingRegressor(
            loss='absolute_error', random_state=0,
        )
        self._model.fit(X, y)

    def predict(self, confs):
        X = np.array([
            _rf_big_features(
                c, self._type_ids, float('nan'), self._with_second_layer)
            for c in confs
        ])
        return self._model.predict(X)


class RnnPredictor(Predictor):
    """Tiny RNN over the hidden-layer sequence.

    Hidden state is initialized from global features (data volume,
    weights, lr, ...), then updated once per layer via
    activation(W_h h + W_x layer_feats + b). A final dense maps the
    final hidden state to a predicted log-loss. Variable layer counts
    are handled by padding + masked jax.lax.scan.
    """

    def __init__(
        self,
        cell_activation: str = 'tanh',
        hidden: int = loss_rnn.HIDDEN,
        lr: float = 0.01,
        steps: int = 2000,
        lr_schedule: str = 'constant',
        feat_proj: int = 0,
        rnn_sub_steps: int = 1,
        cell_type: str = 'elman',
        pooling: str = 'last',
        out_hidden: int = 0,
        out_activation: str = 'gelu',
        batch_size: int = loss_rnn.BATCH_SIZE,
        seed: int | None = None,
    ):
        self._cell_activation = cell_activation
        self._hidden = hidden
        self._lr = lr
        self._steps = steps
        self._lr_schedule = lr_schedule
        self._feat_proj = feat_proj
        self._rnn_sub_steps = rnn_sub_steps
        self._cell_type = cell_type
        self._pooling = pooling
        self._out_hidden = out_hidden
        self._out_activation = out_activation
        self._batch_size = batch_size
        self._seed = seed
        sched = f" {lr_schedule}" if lr_schedule != 'constant' else ''
        proj = f" proj={feat_proj}" if feat_proj else ''
        sub = f" sub={rnn_sub_steps}" if rnn_sub_steps != 1 else ''
        pool = f" pool={pooling}" if pooling != 'last' else ''
        head = (
            f" head={out_hidden}.{out_activation}" if out_hidden > 0 else ''
        )
        bs = (
            f" bs={batch_size}"
            if batch_size != loss_rnn.BATCH_SIZE else ''
        )
        kind = cell_type if cell_type != 'elman' else cell_activation
        self.name = (
            f"rnn ({kind}, h={hidden}{proj}{sub}{pool}{head}, "
            f"lr={lr}{sched}, steps={steps}{bs})"
        )

    def fit(self, train_data):
        self._simple_types = loss_rnn.discover_simple_types(train_data)
        self._params, self._max_layers, trace = loss_rnn.fit(
            train_data, self._simple_types,
            cell_activation=self._cell_activation,
            hidden=self._hidden,
            lr=self._lr,
            steps=self._steps,
            lr_schedule=self._lr_schedule,
            feat_proj=self._feat_proj,
            rnn_sub_steps=self._rnn_sub_steps,
            cell_type=self._cell_type,
            pooling=self._pooling,
            out_hidden=self._out_hidden,
            out_activation=self._out_activation,
            batch_size=self._batch_size,
            seed=self._seed,
        )
        logging.info(f"  {self.name} scan depth: {self._max_layers}")
        n = len(trace)
        for i in range(0, n, max(1, n // 10)):
            logging.info(f"  {self.name} step {i:5d}: train L1 = {trace[i]:.4f}")
        logging.info(f"  {self.name} step {n-1:5d}: train L1 = {trace[-1]:.4f}")

    def predict(self, confs):
        return loss_rnn.predict(
            self._params, confs, self._simple_types, self._max_layers,
            cell_activation=self._cell_activation,
            feat_proj=self._feat_proj,
            rnn_sub_steps=self._rnn_sub_steps,
            cell_type=self._cell_type,
            pooling=self._pooling,
            out_hidden=self._out_hidden,
            out_activation=self._out_activation,
        )


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
        # RandomForestBigPredictor(),  # slow; re-enable for comparisons.
        HistGBRPredictor(),
        HistGBRPredictor(with_second_layer=True),
        RnnPredictor(
            cell_activation='tanh', hidden=32,
            lr=0.02, steps=8000, lr_schedule='cosine',
        ),
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
