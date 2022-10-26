from collections import namedtuple
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from statistics import median
from typing import List

from texmo.configuration import (
    Configuration,
    next_number,
    prev_number,
)
from texmo.model import NCHAR
from texmo.spec import ModelSpec
from texmo import latency


def total_size(shape):
    prod = 1
    for dim in shape:
        prod *= dim
    return prod


LayerStat = namedtuple(
    "LayerStat", ["name", "input", "state", "output", "weights", "suffix"]
)


def get_layer_stats(spec):
    shape = (NCHAR,)

    layers = []

    for layer in spec._layers:
        name = layer.name
        input = total_size(shape)
        if name in ("gru", "mgru"):
            state = layer._size
        elif name == "lstm":
            state = 2 * layer._size
        else:
            state = 0
        weights = layer.weights(shape)
        shape = layer.output_shape(shape)
        output = total_size(shape)
        if name == "suffix":
            suffix = layer._size
        elif name == "attn":
            suffix = layer._length
        else:
            suffix = 1

        layers.append(LayerStat(name, input, state, output, weights, suffix))

    return layers


# We don't care about precisely predicting loss above this value
MAX_LOSS = 16
SCALE_LOSS = 1.2


def encode_loss(loss):
    loss = np.minimum(loss, 2**16)
    return np.tanh(np.log2(loss) / SCALE_LOSS)


ENC_MAX_LOSS = encode_loss(MAX_LOSS)


def decode_loss(loss):
    loss = np.minimum(loss, ENC_MAX_LOSS)
    return np.exp2(SCALE_LOSS * np.arctanh(loss))


class FeatureProvider(object):
    """Provides configuration features for loss prediction."""

    def __init__(self, conf_runs):
        # conf -> [losses]
        self._runs = {}
        # conf -> median loss
        self._scores = {}
        self._spec_features_cache = {}

        for conf, run_loss in conf_runs:
            conf = conf._replace(id=None)
            if conf not in self._runs:
                self._runs[conf] = []
            self._runs[conf].append(encode_loss(run_loss))
        for conf, s in self._runs.items():
            self._scores[conf] = median(s)

    @staticmethod
    def _count_layer(spec: ModelSpec, name: str) -> int:
        count = 0
        for layer in spec._layers:
            if layer.name == name:
                count += 1
        return count

    def _get_spec_features(self, spec: ModelSpec) -> list:
        res = self._spec_features_cache.get(spec)
        if res is not None:
            return res

        features = []
        features.append(math.log2(spec.weights()))
        features.append(len(spec._layers))
        features.append(self._count_layer(spec, "dense"))
        features.append(self._count_layer(spec, "rec"))
        features.append(self._count_layer(spec, "gru"))
        features.append(self._count_layer(spec, "mgru"))
        features.append(self._count_layer(spec, "lstm"))
        features.append(self._count_layer(spec, "suffix"))
        features.append(self._count_layer(spec, "attn"))

        stats = get_layer_stats(spec)

        features.append(math.log2(stats[0].input))
        features.append(math.log2(stats[-1].output))
        features.append(math.log2(min(s.input for s in stats)))
        features.append(math.log2(min(s.input + s.state for s in stats)))
        features.append(math.log2(1 + sum(s.state for s in stats)))
        features.append(math.log2(1 + sum(s.suffix - 1 for s in stats)))

        self._spec_features_cache[spec] = features

        return features

    @staticmethod
    def _get_metaparameter_features(conf) -> list:
        return [math.log2(conf.lr), math.log2(conf.batch), math.log2(conf.t)]

    @staticmethod
    def _get_neighbors(conf):
        return [
            conf._replace(t=conf.t // 2),
            conf._replace(t=conf.t * 2),
            conf._replace(lr=next_number(conf.lr)),
            conf._replace(lr=prev_number(conf.lr)),
            conf._replace(batch=conf.batch // 2),
            conf._replace(batch=conf.batch * 2),
            conf._replace(t=conf.t // 2, lr=next_number(conf.lr)),
            conf._replace(t=conf.t * 2, lr=prev_number(conf.lr)),
            conf._replace(t=conf.t // 2, batch=conf.batch // 2),
            conf._replace(t=conf.t * 2, batch=conf.batch * 2),
        ]

    def _get_neighbor_features(self, conf) -> list:
        conf = conf._replace(id=None)
        neighbors = self._get_neighbors(conf)
        neighbor_scores = [self._scores.get(n) for n in neighbors]
        if any(x is not None for x in neighbor_scores):
            neighbor_median = median(
                x for x in neighbor_scores if x is not None
            )
        else:
            neighbor_median = None

        return neighbor_scores + [neighbor_median]

    def get_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_spec_features(conf.spec),
            dtype=np.float32,
        )

    def get_sparse_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_spec_features(conf.spec)
            + self._get_neighbor_features(conf),
            dtype=np.float32,
        )

    def add_sample(self, conf, loss) -> list:
        """Add a new run.

        Returns the list of confs that need to be updated.
        """
        conf = conf._replace(id=None)
        if conf not in self._runs:
            self._runs[conf] = []
        self._runs[conf].append(encode_loss(loss))
        self._scores[conf] = median(self._runs[conf])

        return self._get_neighbors(conf)


class HistPredictor(object):
    def __init__(self):
        self.pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            max_leaf_nodes=63,
            max_iter=1000,
            n_iter_no_change=10,
            # learning_rate=0.1,
            warm_start=False,
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


class Predictor(object):
    def __init__(self, conf_runs):
        self._runs = {}
        conf_runs = list(conf_runs)
        for conf, loss in conf_runs:
            conf = conf._replace(id=None)
            if conf not in self._runs:
                self._runs[conf] = []
            self._runs[conf].append(loss)
        self._feature_provider = FeatureProvider(conf_runs)
        self._predictor = HistPredictor()

    def add_sample(self, conf, loss) -> List[Configuration]:
        if conf not in self._runs:
            self._runs[conf] = []
        self._runs[conf].append(loss)
        return self._feature_provider.add_sample(conf, loss)

    def train(self):
        with latency.timer("Predictor.train"):
            print("Retraining loss prediction.")

            features = []
            sample_weight = []
            losses = []
            for conf, conf_losses in self._runs.items():
                for loss in conf_losses:
                    features.append(
                        self._feature_provider.get_sparse_features(conf)
                    )
                    sample_weight.append(conf.t)
                    losses.append(loss)

            features = np.array(features, dtype=np.float32)
            sample_weight = np.array(sample_weight, dtype=np.float32)
            losses = np.array(losses, dtype=np.float32)
            losses = encode_loss(losses)

            print("Prepared training data:", features.shape)
            self._predictor.fit(features, losses, sample_weight)

            print("Evaluating")
            loss = self._predictor.loss(features, losses, sample_weight)

            print("Loss on the training data:", loss)

    def predict(self, confs):
        with latency.timer("Predictor.predict"):
            if len(confs) > 10000:
                print("Preparing features")
            features = []
            for conf in confs:
                features.append(
                    self._feature_provider.get_sparse_features(conf)
                )

            features = np.array(features, dtype=np.float32)

            if len(confs) > 10000:
                print("Predicting")
            pred_losses = decode_loss(self._predictor.predict(features))

            if len(confs) > 10000:
                print("Adjusting losses using existing runs")
            losses = []
            for conf, pred_loss in zip(confs, pred_losses):
                conf = conf._replace(id=None)
                runs = self._runs.get(conf)
                if runs is not None:
                    med_loss = median(runs)
                    alt_loss = median(runs + [pred_loss])
                    losses.append(min(pred_loss, alt_loss))
                else:
                    losses.append(pred_loss)

            return losses
