import math

import logging
import numpy as np
from statistics import mean
import jax
from jax import numpy as jnp
import optax
from sklearn.ensemble import HistGradientBoostingRegressor

from .configuration import Configuration, conf_tokens_name
from .prng import Rng
from .tokens import get_tokenizer


_TYPE_IDX = {
    "bits1": 0,
    "bits2": 1,
    "bits4": 2,
    "all": 3,
}


def _predict(weights, input):
    return (jnp.dot(weights["w"], input) + weights["b"]).squeeze()


def _loss(weights, input, output):
    prediction = _predict(weights, input)
    return (jnp.log2(prediction) - jnp.log2(output)) ** 2


class SamplerModel(object):
    def __init__(self):
        self.pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=None,
            max_leaf_nodes=63,
            max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            warm_start=False,
            early_stopping=False,
            categorical_features=[True, True] + [False] * 5,
        )
        self.samples = []
        self.latencies = []

    def _build_features(
        self,
        token_type: str,
        token_processing: str,
        ntokens: int,
        batch: int,
        sample_len: int,
        bytes_per_token: float,
    ):
        features = [_TYPE_IDX[token_type]]
        features.append(1 if token_processing == "capwords" else 0)
        features.append(np.log2(ntokens))
        features.append(np.log2(batch))
        features.append(np.log2(sample_len))
        features.append(np.log2(bytes_per_token))
        features.append(np.log2(bytes_per_token * ntokens * batch))
        return np.array(features, dtype=np.float32)

    def predict(
        self,
        token_type,
        token_processing,
        ntokens,
        batch,
        sample_len,
        bytes_per_token,
    ):
        features = self._build_features(
            token_type,
            token_processing,
            ntokens,
            batch,
            sample_len,
            bytes_per_token,
        )
        xs = features.reshape(1, -1)
        return self.pred.predict(xs)

    def train(
        self,
        token_type,
        token_processing,
        ntokens,
        batch,
        sample_len,
        bytes_per_token,
        latencies,
    ):
        features = self._build_features(
            token_type,
            token_processing,
            ntokens,
            batch,
            sample_len,
            bytes_per_token,
        )
        avg_latency_ms = mean(latencies) / 1E6

        self.samples.append(features)
        self.latencies.append(math.log2(avg_latency_ms))

        logging.info(
            f"Features: {token_type} {token_processing} {ntokens} {batch} {sample_len} {bytes_per_token}"
        )
        if len(self.samples) > 2:
            logging.info(f"Input:\n{features}")
            pred = self.pred.predict(features.reshape(1, -1))
            pred_ms = 2 ** pred[0]
            logging.info(
                f"Predicted latency: {pred_ms} ms real latency: {avg_latency_ms} ms"
            )

        xs = np.array(self.samples, dtype=np.float32)
        ys = np.array(self.latencies, dtype=np.float32)

        logging.info(f"xs:\n{xs}")
        logging.info(f"ys:\n{ys}")

        self.pred.fit(xs, ys)


class TimingModel(object):
    def __init__(self):
        self._sampler_model = SamplerModel()
        self._sample_latency = {}

        self._confs = []
        self._first_step_latency = []
        self._step_latencies = []
        self._steps = []
        self._total_latency = []

        self._regression = None
        self._layer_to_feature = {}
        self._feature_to_layer = []
        self._conf_features = {}

    def register_sample_latency(
        self,
        token_set_name: str,
        sample_len: int,
        batch: int,
        latencies: list[int],
    ):
        assert isinstance(latencies, list)
        token_set = get_tokenizer(token_set_name).token_set

        self._sampler_model.train(
            token_set.token_type,
            token_set.processing,
            token_set.ntokens,
            batch,
            sample_len,
            token_set.bytes_per_token,
            latencies,
        )

    def generate_timing_key(self, conf: Configuration):
        key = len(self._confs)
        self._confs.append(conf)
        self._first_step_latency.append(None)
        self._step_latencies.append([])
        self._steps.append(None)
        self._total_latency.append(None)
        return key

    def register_step(self, key, first: bool, latency_s: float):
        if first:
            self._first_step_latency[key] = latency_s
        else:
            self._step_latencies[key].append(latency_s)

    def register_training_time(self, key, steps: int, latency_s: float):
        self._steps[key] = steps
        self._total_latency[key] = latency_s

    # def fit(self):
    #     for conf in self._confs:
    #         if conf in self._conf_features:
    #             continue
    #         batch = conf.batch
    #         sample_len = conf.sample_len
    #         ntokens = conf.model.ntokens
    #         layers = [f"input-i{ntokens}-b{batch}-l{sample_len}"]

    #         size = ntokens
    #         for layer in conf.model.layers:
    #             layers.append(f"{layer}-i{size}-b{batch}-l{sample_len}")
    #             size = 1
    #             for dim in layer.output_shape:
    #                 size *= dim

    #         layers.append(f"output{ntokens}-i{size}-b{batch}-l{sample_len}")

    #         conf_features = []

    #         for layer in layers:
    #             feature_id = self._layer_to_feature.get(layer)
    #             if feature_id is None:
    #                 feature_id = len(self._feature_to_layer)
    #                 self._feature_to_layer.append(layer)
    #                 self._layer_to_feature[layer] = feature_id

    #             conf_features.append(feature_id)

    #         self._conf_features[conf] = conf_features

    #     xs = []
    #     first_step = []
    #     step = []
    #     weights = []

    #     for conf, first_step_latency, step_latencies, steps in zip(
    #         self._confs,
    #         self._first_step_latency,
    #         self._step_latencies,
    #         self._steps,
    #     ):
    #         x = np.zeros(
    #             shape=(
    #                 len(
    #                     self._feature_to_layer,
    #                 )
    #             ),
    #             dtype=np.float32,
    #         )
    #         conf_features = self._conf_features[conf]
    #         for f in conf_features:
    #             x[f] += 1
    #         xs.append(x)
    #         first_step.append(first_step_latency)

    #         if step_latencies:
    #             step.append(mean(step_latencies))
    #         else:
    #             # assert steps == 1
    #             step.append(0)
    #         weights.append(len(step_latencies))

    #     self._step_regression = linear_model.LinearRegression(
    #         positive=True, fit_intercept=False
    #     )
    #     self._step_regression.fit(xs, step, weights)

    #     self._first_step_regression = linear_model.LinearRegression(
    #         positive=True, fit_intercept=False
    #     )
    #     self._first_step_regression.fit(xs, first_step)

    def report(self):
        for key in sorted(self._sample_latency.keys()):
            avg = mean(self._sample_latency[key]) * 1000
            n = len(self._sample_latency[key])
            print(f"{key}  {avg:.3f} ms ({n})")

        # self.fit()

        # print(
        #     self._first_step_regression.intercept_ * 1000,
        #     self._step_regression.intercept_ * 1000,
        # )

        # for f in sorted(self._feature_to_layer):
        #     i = self._layer_to_feature[f]
        #     print(
        #         f, " ",
        #         self._first_step_regression.coef_[i] * 1000, " ",
        #         self._step_regression.coef_[i] * 1000,
        #     )
