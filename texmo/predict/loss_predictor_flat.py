import logging
import math
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..configuration import Configuration, Template, conf_is_valid, conf_tokens_name
from ..resultdb import ResultDB
from ..results import ResultSet
from ..run import Run
from ..model2 import Model2
from ..tokens import get_tokenizer
from .features import get_layer_cat_features, get_tokens_cat_features
from .predict_common import prediction_score, encode_loss, decode_loss


def make_metaparameter_features(conf: Configuration, run: Run) -> list[float]:
    assert conf_is_valid(conf)
    return [
        math.log2(run.steps),
        math.log2(conf.lr),
        math.log2(conf.sample_len),
        math.log2(conf.batch),
        len(conf.model.layers),
    ]


def make_tokenset_features(conf: Configuration) -> list[float]:
    token_set = get_tokenizer(conf_tokens_name(conf)).token_set
    return get_tokens_cat_features(token_set)


_spec_features_cache = {}


def make_model_features(model: Model2) -> list:
    res = _spec_features_cache.get(model)
    if res is not None:
        return res

    features = []

    for i in (0, 1, 2, -1):
        if i >= len(model.layers):
            features.extend([None] * 7)
            continue
        features.extend(get_layer_cat_features(model.layers[i]))

    _spec_features_cache[model] = features

    return features


def make_features(conf: Configuration, run: Run) -> list[float]:
    return np.array(
            make_metaparameter_features(conf, run)
            + make_tokenset_features(conf)
            + make_model_features(conf.model),
            dtype=np.float32,
        )

class LossPredictorFlat(object):
    def __init__(self, result_db: ResultDB):
        self._result_db = result_db
        self._result_set = ResultSet(
            result_db, template=Template(), populate_neighbors=False
        )
        self._pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            # max_leaf_nodes=63,
            # max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            warm_start=False,
            # early_stopping=False,
            categorical_features=[False] * 5 + [True, True, False, False, False] + [True, True, False, False, False, False, False] * 4,
        )

    def _prepare_data(self, result_set: ResultSet):
        features = []
        sample_weight = []
        losses = []
        for conf, run in result_set.all_conf_runs():
            features.append(make_features(conf, run))
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
        logging.info("Splitting into train and test sets")
        train_set = ResultSet(result_db=None, template=Template(), populate_neighbors=False)
        test_set = ResultSet(result_db=None, template=Template(), populate_neighbors=False)

        for _, conf, run in self._result_db.get_confs_runs():
            target_set = train_set if random.random() < 0.9 else test_set
            target_set.add_run_conf(conf, run)

        features, losses, sample_weight = self._prepare_data(train_set)
        test_features, test_losses, test_sample_weight = self._prepare_data(test_set)

        logging.info(f"Prepared training data: {features.shape}")
        self._pred.fit(features, losses, sample_weight)

        pred_losses = self._pred.predict(test_features)
        score = prediction_score(test_losses, pred_losses)
        logging.info(f"Loss on test set ({test_features.shape}): {score}")
