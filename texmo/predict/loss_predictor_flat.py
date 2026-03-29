import logging
import math
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import latency
from ..configuration import (Configuration, Template)
from ..model3 import Model3
from ..resultdb import ResultDB
from ..results import ResultSet
from ..run import Run
from ..tokens import get_tokenizer
from .features import get_layer_cat_features, get_tokens_cat_features
from .predict_common import decode_loss, encode_loss, prediction_score


def make_metaparameter_features(conf: Configuration, steps: int) -> list[float]:
    assert conf.is_valid()
    assert steps > 0
    return [
        math.log2(steps),
        math.log2(conf.lr),
        math.log2(conf.sample_len),
        math.log2(conf.batch),
        len(conf.model.layers),
    ]


def make_tokenset_features(conf: Configuration) -> list[float]:
    token_set = conf.model.input.tokenizer.token_set
    return get_tokens_cat_features(token_set)


_spec_features_cache = {}


def make_model_features(model: Model3) -> list:
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


def make_features(conf: Configuration, steps: int) -> list[float]:
    assert isinstance(steps, int)
    return np.array(
        make_metaparameter_features(conf, steps)
        + make_tokenset_features(conf)
        + make_model_features(conf.model),
        dtype=np.float32,
    )


class LossPredictorFlat(object):
    def __init__(self, result_db: ResultDB, extra_dbs: list[str]):
        self._result_db = result_db
        self._extra_dbs = []
        for path in extra_dbs:
            db = ResultDB(path)
            self._extra_dbs.append(db)
        self._pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            # max_leaf_nodes=63,
            # max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            warm_start=False,
            # early_stopping=False,
            categorical_features=[False] * 5
            + [True, True, False, False, False]
            + [True, True, False, False, False, False, False] * 4,
        )
        self._samples_till_next_train = 0

    def _prepare_data(self, result_set: ResultSet):
        features = []
        sample_weight = []
        losses = []
        for conf, run in result_set.all_conf_runs():
            features.append(make_features(conf, run.steps))
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

    def _get_all_confs_runs(self):
        for _, conf, run in self._result_db.get_confs_runs():
            yield conf, run
        for db in self._extra_dbs:
            for _, conf, run in db.get_confs_runs():
                yield conf, run

    def train(self):
        logging.info("Splitting into train and test sets")
        train_set = ResultSet(
            result_db=None, template=Template(), populate_neighbors=False
        )
        test_set = ResultSet(
            result_db=None, template=Template(), populate_neighbors=False
        )

        total_samples = 0

        for conf, run in self._get_all_confs_runs():
            target_set = train_set if random.random() < 0.9 else test_set
            target_set.add_run_conf(conf, run)
            total_samples += 1

        self._last_train_samples = total_samples
        self._samples_since_train = 0

        logging.info("Preparing loss model training data")
        features, losses, sample_weight = self._prepare_data(train_set)
        logging.info(f"Prepared training data: {features.shape}")
        logging.info("Preparing loss model test data")
        test_features, test_losses, test_sample_weight = self._prepare_data(
            test_set
        )

        self._pred.fit(features, losses, sample_weight)

        pred_losses = self._pred.predict(test_features)
        score = prediction_score(test_losses, pred_losses)
        logging.info(f"Loss on test set ({test_features.shape}): {score}")

    def predict(
        self, confs: list[Configuration], steps: list[int]
    ) -> list[float]:
        if not confs:
            return []
        with latency.timer("LossPredictionFlat.predict"):
            features = []
            for conf, s in zip(confs, steps):
                features.append(make_features(conf, s))
            features = np.array(features, dtype=np.float32)
            losses = self._pred.predict(features)
            return decode_loss(losses)
        
    def total_runs(self) -> int:
        runs = self._result_db.total_runs()
        for db in self._extra_dbs:
            runs += db.total_runs()
        return runs

    def maybe_train(self) -> bool:
        logging.info(f"_samples_till_next_train = {self._samples_till_next_train}")

        self._samples_till_next_train -= 1
        if self._samples_till_next_train > 0:
            return False

        self.train()

        total_runs = self.total_runs()
        self._samples_till_next_train = int(total_runs ** (1 / 3))

        logging.info(f"total_runs = {total_runs}")


        return True
