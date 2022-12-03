from collections import namedtuple
import math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from statistics import median
from typing import List

from texmo.configuration import Configuration, conf_is_valid
from texmo.common import NCHAR, total_size
from texmo.model2 import Model2
from texmo import latency


# We aren't differentiating losses above 10 bits per byte.
MIN_LOSS = 0.1
MAX_LOSS = 10


def prediction_score(true_losses, predicted_losses):
    n = len(true_losses)
    assert len(predicted_losses) == n

    all_losses = np.concatenate((true_losses, predicted_losses))
    all_losses = np.minimum(all_losses, MAX_LOSS)
    all_losses = np.maximum(all_losses, MIN_LOSS)
    all_losses = np.log2(all_losses)

    return np.average(np.abs(all_losses[:n] - all_losses[n:]))


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

    def _get_model_features(self, model: Model2) -> list:
        res = self._spec_features_cache.get(model)
        if res is not None:
            return res

        features = []
        # features.append(math.log2(spec.weights()))

        stats = get_layer_stats(model)

        # features.append(math.log2(stats[0].input))
        # features.append(math.log2(stats[-1].output))
        # features.append(math.log2(min(s.input for s in stats)))
        # features.append(math.log2(min(s.input + s.state for s in stats)))
        # features.append(math.log2(1 + sum(s.state for s in stats)))
        # features.append(math.log2(1 + sum(s.suffix - 1 for s in stats)))

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

        features.append(len(model.layers))

        self._spec_features_cache[model] = features

        return features

    @staticmethod
    def _get_metaparameter_features(conf) -> list:
        assert conf_is_valid(conf)
        return [
            math.log2(conf.lr),
            math.log2(conf.sample_len),
            math.log2(conf.batch),
            math.log2(conf.t),
        ]

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
        ]

    def _get_neighbor_features(self, conf) -> list:
        conf = conf._replace(id=None)
        neighbors = self._get_neighbors(conf)
        neighbor_scores = [self._scores.get(n) for n in neighbors]
        return neighbor_scores

    def get_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_model_features(conf.model),
            dtype=np.float32,
        )

    def get_sparse_features(self, conf) -> np.array:
        return np.array(
            self._get_model_features(conf.model)
            + self._get_neighbor_features(conf)
            + self._get_metaparameter_features(conf),
            dtype=np.float32,
        )

    def categorical(self) -> list:
        return [True, False, False, False] * 4 + [False] * 18

    def add_sample(self, conf, loss) -> list:
        """Add a new run.

        Returns the list of confs that need to be updated.
        """
        conf = conf._replace(id=None)
        if conf not in self._runs:
            self._runs[conf] = []
        self._runs[conf].append(encode_loss(loss))
        self._scores[conf] = median(self._runs[conf])

        for neighbor in self._get_neighbors(conf):
            if conf_is_valid(neighbor):
                yield neighbor


class HistPredictor(object):
    def __init__(self, feature_provider):
        self.pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            max_leaf_nodes=63,
            max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            warm_start=False,
            early_stopping=False,
            categorical_features=feature_provider.categorical(),
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
        self._predictor = HistPredictor(self._feature_provider)

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
                    losses.append(min(med_loss, alt_loss))
                else:
                    losses.append(pred_loss)

            return losses
