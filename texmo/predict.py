import logging
import math
from collections import namedtuple
from collections.abc import Iterable
from statistics import median
from typing import List

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import latency
from .common import NCHAR, total_size
from .configuration import Configuration, conf_is_valid
from .model2 import Model2
from .predict_common import MAX_LOSS
from .results import ResultSet
from .run import Run

MAX_LOG_LOSS = math.log2(MAX_LOSS)


LayerStat = namedtuple(
    "LayerStat", ["name", "input", "state", "output", "weights", "suffix"]
)


def get_layer_stats(model):
    shape = (NCHAR,)

    layers = []

    for layer in model.layers:
        name = layer.name
        input = total_size(shape)
        if name in ("gru", "mgru", "rec"):
            state = layer.size
        elif name == "lstm":
            state = 2 * layer.size
        else:
            state = None
        weights = layer.weights
        shape = layer.output_shape
        output = total_size(shape)
        if name == "suffix":
            suffix = layer.length
        elif name == "attn":
            suffix = layer.length
        elif name == "dense":
            suffix = 1
        else:
            suffix = None
        if name in ("dense", "rec"):
            name += layer._activation_suffix

        layers.append(LayerStat(name, input, state, output, weights, suffix))

    return layers


def encode_loss(loss: np.ndarray):
    if loss is None:
        return None
    loss = np.minimum(loss, 2**16)
    loss = np.log2(loss)
    return loss


def decode_loss(loss):
    if loss is None:
        return None
    loss = np.minimum(loss, MAX_LOG_LOSS)
    return np.exp2(loss)


class FeatureProvider(object):
    """Provides configuration features for loss prediction."""

    def __init__(self, result_set: ResultSet):
        assert isinstance(result_set, ResultSet)
        self._result_set = result_set
        # conf -> median loss
        self._conf_loss = {}
        self._spec_features_cache = {}

        for conf_results in self._result_set._confs.values():
            self._conf_loss[conf_results.conf] = encode_loss(
                conf_results.median_score
            )

    @staticmethod
    def _get_metaparameter_features(conf) -> list:
        assert conf_is_valid(conf)
        return [
            math.log2(conf.lr),
            math.log2(conf.sample_len),
            math.log2(conf.batch),
            math.log2(conf.t),
            len(conf.model.layers),
        ]

    def _get_layer_features(self, model: Model2) -> list:
        res = self._spec_features_cache.get(model)
        if res is not None:
            return res

        features = []

        stats = get_layer_stats(model)

        layer_type_enc = {
            "dense.tanh": 1,
            "dense.relu": 2,
            "rec.tanh": 3,
            "rec.relu": 4,
            "gru": 5,
            "mgru": 6,
            "lstm": 7,
            "suffix": 8,
            "attn": 9,
        }
        for i in (0, 1, 2, -1):
            if i >= len(stats):
                features.extend([0, None, None, None])
                continue
            stat = stats[i]
            layer_type = layer_type_enc[stat.name]
            state = None if stat.state is None else math.log2(stat.state)
            output = math.log2(stat.output)
            suffix = None if stat.suffix is None else math.log2(stat.suffix)
            features.extend([layer_type, state, output, suffix])

        self._spec_features_cache[model] = features

        return features

    @staticmethod
    def _get_neighbors(conf):
        return [
            conf._replace(t=conf.t // 2),
            conf._replace(t=conf.t * 2),
            conf._replace(lr=conf.lr / 2),
            conf._replace(lr=conf.lr * 2),
            conf._replace(batch=conf.batch // 2),
            conf._replace(batch=conf.batch * 2),
            conf._replace(t=conf.t // 2, lr=conf.lr * 2),
            conf._replace(t=conf.t * 2, lr=conf.lr / 2),
            conf._replace(t=conf.t // 2, batch=conf.batch // 2),
            conf._replace(t=conf.t * 2, batch=conf.batch * 2),
            conf._replace(model=conf.model.remove_last_layer()),
            conf._replace(t=conf.t // 2, model=conf.model.remove_last_layer()),
            conf._replace(
                t=conf.t // 2,
                lr=conf.lr * 2,
                model=conf.model.remove_last_layer(),
            ),
            # conf,
        ]

    def _get_neighbor_features(self, conf) -> list:
        neighbors = self._get_neighbors(conf)
        neighbor_scores = [self._conf_loss.get(n) for n in neighbors]

        neighbor_shorter = conf._replace(t=conf.t // 2)
        results = self._result_set.get_conf_results(neighbor_shorter)
        if results is None or not results.runs:
            neighbor_scores.extend((None, None))
        else:
            steps = median(len(r.step_loss) for r in results.runs)
            pred_score = median(
                r.loss_trend.predict(2 * len(r.step_loss)) for r in results.runs
            )
            neighbor_scores.extend((math.log2(steps), pred_score))

        return neighbor_scores

    def get_dense_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_model_features(conf.model),
            dtype=np.float32,
        )

    def get_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_layer_features(conf.model)
            + self._get_neighbor_features(conf),
            dtype=np.float32,
        )

    def categorical(self) -> list:
        return [False] * 5 + [True, False, False, False] * 4 + [False] * 15

    def add_sample(self, conf_results, run) -> list:
        """Add a new run.

        Returns the list of confs that need to be updated.
        """

        self._conf_loss[conf_results.conf] = conf_results.median_score
        for neighbor in self._get_neighbors(conf_results.conf):
            if conf_is_valid(neighbor):
                yield neighbor


class _HistPredictor(object):
    def __init__(self, categorical_features: list[bool]):
        self.pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            max_leaf_nodes=63,
            max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            warm_start=False,
            early_stopping=False,
            categorical_features=categorical_features,
        )

    def _fit(self, xs, ys, sample_weight):
        self.pred.fit(xs, ys, sample_weight)

    def _predict(self, xs):
        return self.pred.predict(xs)

    def fit(self, xs, ys, sample_weight):
        with latency.timer("HistPredictor.fit"):
            self._fit(xs, ys, sample_weight)

    def predict(self, xs):
        with latency.timer("HistPredictor.predict"):
            return self._predict(xs)

    def predict_one(self, x):
        xs = x.reshape((1, -1))
        return self._predict(xs)[0]

    def loss(self, test_xs, test_ys, sample_weight):
        pred_ys = self.predict(test_xs)
        pred_ys = decode_loss(pred_ys)
        pred_ys = np.minimum(pred_ys, MAX_LOSS)
        pred_ys = np.log2(pred_ys)

        test_ys = decode_loss(test_ys)
        test_ys = np.minimum(test_ys, MAX_LOSS)
        test_ys = np.log2(test_ys)

        error = np.abs(pred_ys - test_ys) * sample_weight
        return np.sum(error) / np.sum(sample_weight)


class AveragePredictor(object):
    """Predictor based on the previous runs."""

    def __init__(self, result_set):
        self._result_set = result_set

    def train(self):
        pass

    def predict(self, confs):
        with latency.timer("MedianPredictor.predict"):
            losses = []
            for conf in confs:
                conf_results = self._result_set.find_conf_results(conf)
                if conf_results is None or not conf_results.runs:
                    losses.append(3)
                else:
                    # losses.append(median(r.loss for r in conf_results.runs))
                    l = [
                        math.log2(min(r.loss, MAX_LOSS))
                        for r in conf_results.runs
                    ]
                    losses.append(2 ** (sum(l) / len(l)))

            return losses


class Predictor3(object):
    """Predictor based on the previous runs."""

    def __init__(self, result_set):
        self._result_set = result_set

    def train(self):
        pass

    def predict(self, confs):
        with latency.timer("Predictor3.predict"):
            losses = []
            for conf in confs:
                losses.append(3)
            return losses


class Predictor(object):
    def __init__(self, result_set):
        self._result_set = result_set
        self._feature_provider = FeatureProvider(result_set)
        categorical_features = self._feature_provider.categorical()
        self._predictor = _HistPredictor(categorical_features)

    def add_sample(self, conf: Configuration, run: Run) -> List[Configuration]:
        return self._feature_provider.add_sample(conf, run)

    def train(self):
        with latency.timer("Predictor.train"):
            logging.info("Retraining loss prediction.")

            features = []
            sample_weight = []
            losses = []
            for conf, loss in self._result_set.all_conf_runs():
                features.append(self._feature_provider.get_features(conf))
                sample_weight.append(conf.t)
                assert loss is not None
                assert loss > 0.1
                losses.append(loss)

            features = np.array(features, dtype=np.float32)
            logging.info("Features:\n" + str(features))
            sample_weight = np.array(sample_weight, dtype=np.float32)
            losses = np.array(losses, dtype=np.float32)
            losses = encode_loss(losses)

            logging.info(f"Prepared training data: {features.shape}")
            self._predictor.fit(features, losses, sample_weight)

            logging.info("Evaluating")
            loss = self._predictor.loss(features, losses, sample_weight)

            logging.info(f"Loss on the training data: {loss}")

    def predict(self, confs: Iterable[Configuration]):
        with latency.timer("Predictor.predict"):
            if len(confs) > 10000:
                logging.info("Preparing features")
            features = []
            for conf in confs:
                features.append(self._feature_provider.get_features(conf))

            features = np.array(features, dtype=np.float32)

            if len(confs) > 10000:
                logging.info("Predicting")
            pred_losses = decode_loss(self._predictor.predict(features))

            if len(confs) > 10000:
                logging.info("Adjusting losses using existing runs")
            losses = []
            for conf, pred_loss in zip(confs, pred_losses):
                conf_results = self._result_set.get_conf_results(conf)
                if conf_results is not None:
                    losses.append(
                        median(
                            [run.loss for run in conf_results.runs]
                            + [pred_loss]
                        )
                    )
                else:
                    losses.append(pred_loss)

            return losses


# Predictor = AveragePredictor
