import argparse
from collections import namedtuple
import math
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from statistics import median
from itertools import islice

from texmo.predict import FeatureProvider, HistPredictor, encode_loss
from texmo.common import NCHAR
from texmo.predict import decode_loss, MAX_LOSS
from texmo.resultdb import ResultDB
from texmo import latency


class BasePredictor(object):
    supports_sparse = False

    def _fit(self, xs, ys, sample_weight):
        pass

    def _predict(self, xs):
        pass

    def fit(self, xs, ys, sample_weight):
        with latency.timer(self.__class__.__name__ + "-fit"):
            self._fit(xs, ys, sample_weight)

    def predict(self, xs):
        with latency.timer(self.__class__.__name__ + "-predict"):
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


class Const(BasePredictor):
    def __init__(self):
        self._value = math.log2(4)

    def _fit(self, xs, ys, sample_weight):
        self._value = median(ys)

    def _predict(self, xs):
        xs = np.array(xs)
        return np.ones(shape=(xs.shape[0],)) * self._value


class Linear(BasePredictor):
    def __init__(self):
        self.pred = LinearRegression()

    def _fit(self, xs, ys, sample_weight):
        self.pred.fit(xs, ys, sample_weight)

    def _predict(self, xs):
        return self.pred.predict(xs)


class LinearReg(BasePredictor):
    def __init__(self):
        self.pred = Ridge()

    def _fit(self, xs, ys, sample_weight):
        self.pred.fit(xs, ys, sample_weight)

    def _predict(self, xs):
        return self.pred.predict(xs)


class Svm(BasePredictor):
    def __init__(self):
        self.pred = LinearSVR(max_iter=1000)

    def _fit(self, xs, ys):
        self.pred.fit(xs, ys)

    def _predict(self, xs):
        return self.pred.predict(xs)


class Neighbors(BasePredictor):
    def __init__(self):
        self.pred = KNeighborsRegressor()

    def _fit(self, xs, ys, sample_weight):
        self.pred.fit(xs, ys)

    def _predict(self, xs):
        return self.pred.predict(xs)


class Decision(BasePredictor):
    def __init__(self):
        self.pred = DecisionTreeRegressor(
            criterion="absolute_error", max_depth=3
        )

    def _fit(self, xs, ys):
        self.pred.fit(xs, ys)

    def _predict(self, xs):
        return self.pred.predict(xs)


class AdaBoost(BasePredictor):
    def __init__(self):
        self.pred = AdaBoostRegressor(
            DecisionTreeRegressor(criterion="absolute_error", max_depth=3),
            loss="linear",
        )

    def _fit(self, xs, ys):
        self.pred.fit(xs, ys)

    def _predict(self, xs):
        return self.pred.predict(xs)


class Forest(BasePredictor):
    def __init__(self):
        self.pred = RandomForestRegressor(
            n_estimators=10, criterion="absolute_error", max_depth=3
        )

    def _fit(self, xs, ys):
        self.pred.fit(xs, ys)

    def _predict(self, xs):
        return self.pred.predict(xs)


class HistGradient(BasePredictor):
    supports_sparse = True

    def __init__(self):
        self.pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            max_leaf_nodes=65,
            max_iter=10000,
            n_iter_no_change=20,
            learning_rate=0.1,
        )

    def _fit(self, xs, ys, sample_weight):
        self.pred.fit(xs, ys, sample_weight)

    def _predict(self, xs):
        return self.pred.predict(xs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_db = ResultDB(args.db)

    conf_runs = list(record_db.get_confs_runs())

    feature_provider = FeatureProvider(conf_runs)
    conf_features = []
    sparse_features = []
    sample_weight = []
    losses = []
    for conf, loss in conf_runs:
        conf_features.append(feature_provider.get_features(conf))
        sparse_features.append(feature_provider.get_sparse_features(conf))
        sample_weight.append(conf.t)
        losses.append(loss)

    print("Read", len(conf_features), "runs from the DB")

    conf_features = np.array(conf_features, dtype=np.float32)
    sparse_features = np.array(sparse_features, dtype=np.float32)
    sample_weight = np.array(sample_weight, dtype=np.float32)
    losses = np.array(losses, dtype=np.float32)
    losses = encode_loss(losses)

    (
        train_features,
        test_features,
        train_sparse_features,
        test_sparse_features,
        train_sample_weight,
        test_sample_weight,
        train_losses,
        test_losses,
    ) = train_test_split(
        conf_features, sparse_features, sample_weight, losses, test_size=0.2
    )

    print(
        "Train data:",
        train_features.shape,
        train_sparse_features.shape,
        train_losses.shape,
    )
    print(train_features)
    print(train_sparse_features)
    print(train_losses)
    print()

    print("Test data:", test_features.shape, test_losses.shape)
    print(test_features)
    print(test_sparse_features)
    print(test_losses)
    print()

    pred = HistPredictor()
    pred.fit(
        train_sparse_features, train_losses, sample_weight=train_sample_weight
    )
    loss = pred.loss(test_sparse_features, test_losses, test_sample_weight)
    print("Loss:", loss)

    # for pred_cls in (
    #     Const,
    #     Linear,
    #     LinearReg,
    #     # Svm,
    #     Neighbors,
    #     # Decision,
    #     # AdaBoost,
    #     HistGradient,
    # ):
    #     pred = pred_cls()
    #     if pred.supports_sparse:
    #         pred.fit(train_sparse_features, train_losses, sample_weight=train_sample_weight)
    #         loss = pred.loss(test_sparse_features, test_losses, test_sample_weight)
    #         # if loss < 0.1:
    #         #     for features, loss in islice(zip(test_sparse_features, test_losses), 10):
    #         #         pred_loss = pred.predict_one(features)
    #         #         print(features, decode_loss(loss), decode_loss(pred_loss))
    #     else:
    #         pred.fit(train_features, train_losses, sample_weight=train_sample_weight)
    #         loss = pred.loss(test_features, test_losses, test_sample_weight)
    #     print(pred_cls.__name__, "loss:", loss)

    latency.report()
