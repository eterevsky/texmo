import logging
import math
from collections import namedtuple
from collections.abc import Iterable
from statistics import median
from typing import List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import latency
from ..common import NCHAR, total_size
from ..configuration import Configuration, conf_is_valid, conf_tokens_name
from ..model2 import Model2
from .features import get_layer_cat_features, get_tokens_cat_features
from .predict_common import MAX_LOSS, encode_loss, decode_loss, prediction_score
from ..results import ResultSet, ConfResults
from ..run import Run
from ..tokens import get_tokenizer

class ResultsProvider(object):
    """An abstract class that returns results for a given configuration.

    Added to avoid a circular dependency.
    """

    def all_conf_results(self) -> Iterable[ConfResults]:
        raise NotImplementedError

    def get_conf_results(self, conf: Configuration) -> Optional[ConfResults]:
        raise NotImplementedError


class FeatureProvider(object):
    """Provides configuration features for loss prediction."""

    def __init__(self, result_set: ResultSet):
        assert isinstance(result_set, ResultSet)
        self._result_set = result_set
        # conf -> median loss
        self._conf_loss = {}
        self._spec_features_cache = {}

        for conf_results in self._result_set.all_conf_results():
            self._conf_loss[conf_results.conf] = encode_loss(
                conf_results.median_score
            )

    @staticmethod
    def _get_tokenset_features(conf) -> list:
        token_set = get_tokenizer(conf_tokens_name(conf)).token_set
        return get_tokens_cat_features(token_set)

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

        for i in (0, 1, 2, -1):
            if i >= len(model.layers):
                features.extend([None] * 7)
                continue
            features.extend(get_layer_cat_features(model.layers[i]))

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
            conf._replace(sample_len=conf.sample_len // 2),
            conf._replace(sample_len=conf.sample_len * 2),
            conf._replace(sample_len=conf.sample_len // 2, batch=conf.batch * 2),
            conf._replace(sample_len=conf.sample_len * 2, batch=conf.batch // 2),
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
        neighbors = self._get_neighbors(conf)
        neighbor_scores = [self._conf_loss.get(n) for n in neighbors]

        neighbor_shorter = conf._replace(t=conf.t // 2)
        results = self._result_set.get_conf_results(neighbor_shorter)
        if results is None or not results.runs:
            neighbor_scores.extend((None, None))
        else:
            steps = median(len(r.step_loss) for r in results.runs)
            pred_score = float(median(
                r.loss_trend.predict(2 * len(r.step_loss)) for r in results.runs
            ))
            neighbor_scores.extend((math.log2(steps), pred_score))

        return neighbor_scores

    def get_features(self, conf) -> np.array:
        return np.array(
            self._get_metaparameter_features(conf)
            + self._get_tokenset_features(conf)
            + self._get_layer_features(conf.model),
            # + self._get_neighbor_features(conf),
            dtype=np.float32,
        )

    def categorical(self) -> list:
        return [False] * 5 + [True, True, False, False, False] + [True, True, False, False, False, False, False] * 4  #+ [False] * 19

    def update_conf_results(self, conf_results: ConfResults) -> list:
        """Add a new run.

        Returns the list of confs that need to be updated.
        """
        assert isinstance(conf_results, ConfResults)
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
            # warm_start=False,
            # early_stopping=False,
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

    # def loss(self, test_xs, test_ys, sample_weight):
    #     pred_ys = self.predict(test_xs)
    #     pred_ys = decode_loss(pred_ys)
    #     pred_ys = np.minimum(pred_ys, MAX_LOSS)
    #     pred_ys = np.log2(pred_ys)

    #     test_ys = decode_loss(test_ys)
    #     test_ys = np.minimum(test_ys, MAX_LOSS)
    #     test_ys = np.log2(test_ys)

    #     error = np.abs(pred_ys - test_ys) * sample_weight
    #     return np.sum(error) / np.sum(sample_weight)

    def loss(self, test_xs, test_ys):
        pred_ys = self.predict(test_xs)
        return prediction_score(pred_ys, test_ys)


class LossPredictorV1(object):
    def __init__(self, result_set: ResultSet, split_test_set: bool = False):
        assert isinstance(result_set, ResultSet)
        self._result_set = result_set
        self._feature_provider = FeatureProvider(result_set)
        categorical_features = self._feature_provider.categorical()
        self._predictor = _HistPredictor(categorical_features)
        self._split_test_set = split_test_set

    def update_conf_results(self, conf_results: ConfResults) -> List[Configuration]:
        return self._feature_provider.update_conf_results(conf_results)
    
    def _prepare_data(self, result_set: ResultSet):
        features = []
        sample_weight = []
        losses = []
        for conf, run in result_set.all_conf_runs():
            features.append(self._feature_provider.get_features(conf))
            sample_weight.append(conf.t)
            loss = run.loss
            assert loss is not None
            assert loss > 0.1
            losses.append(loss)

        features = np.array(features, dtype=np.float32)
        logging.info("Features:\n" + str(features))
        sample_weight = np.array(sample_weight, dtype=np.float32)
        losses = np.array(losses, dtype=np.float32)
        losses = encode_loss(losses)

        return features, losses, sample_weight

    def train(self):
        with latency.timer("Predictor.train"):
            logging.info("Retraining loss prediction.")

            if self._split_test_set:
                logging.info("Splitting the data into training and test sets.")
                train_set, test_set = self._result_set.train_test_split()
                features, losses, sample_weight = self._prepare_data(train_set)
                test_features, test_losses, test_sample_weight = self._prepare_data(test_set)
            else:
                features, losses, sample_weight = self._prepare_data(self._result_set)

            logging.info(f"Prepared training data: {features.shape}")
            self._predictor.fit(features, losses, sample_weight)

            if self._split_test_set and test_features.shape[0] > 0:
                loss = self._predictor.loss(test_features, test_losses)
                logging.info(f"Loss on test set ({test_features.shape}): {loss}")
            else:
                loss = self._predictor.loss(features, losses)
                logging.info(f"Loss on training set: {loss}")

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
