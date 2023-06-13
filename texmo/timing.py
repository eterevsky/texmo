import logging
import numpy as np
from statistics import mean
from sklearn import linear_model

from .configuration import Configuration, conf_tokens_name


class SamplerModel(object):
    def __init__(self):
        pass

    def predict(self, token_type, token_processing, ntokens, batch, sample_len, bytes_per_token):
        pass

    def train(self, token_type, token_processing, ntokens, batch, sample_len, bytes_per_token, latencies):
        pass



class Timing(object):
    def __init__(self):
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
        self, token_set_name: str, sample_len: int, batch: int, latency_s: float
    ):
        key = (token_set_name, sample_len, batch)

        l = self._sample_latency.get(key)
        if l is None:
            l = []
            self._sample_latency[key] = l

        l.append(latency_s)

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

    def fit(self):
        for conf in self._confs:
            if conf in self._conf_features:
                continue
            batch = conf.batch
            sample_len = conf.sample_len
            ntokens = conf.model.ntokens
            layers = [f"input-i{ntokens}-b{batch}-l{sample_len}"]

            size = ntokens
            for layer in conf.model.layers:
                layers.append(f"{layer}-i{size}-b{batch}-l{sample_len}")
                size = 1
                for dim in layer.output_shape:
                    size *= dim

            layers.append(f"output{ntokens}-i{size}-b{batch}-l{sample_len}")

            conf_features = []

            for layer in layers:
                feature_id = self._layer_to_feature.get(layer)
                if feature_id is None:
                    feature_id = len(self._feature_to_layer)
                    self._feature_to_layer.append(layer)
                    self._layer_to_feature[layer] = feature_id

                conf_features.append(feature_id)

            self._conf_features[conf] = conf_features

        xs = []
        first_step = []
        step = []
        weights = []

        for conf, first_step_latency, step_latencies, steps in zip(
            self._confs,
            self._first_step_latency,
            self._step_latencies,
            self._steps,
        ):
            x = np.zeros(
                shape=(
                    len(
                        self._feature_to_layer,
                    )
                ),
                dtype=np.float32,
            )
            conf_features = self._conf_features[conf]
            for f in conf_features:
                x[f] += 1
            xs.append(x)
            first_step.append(first_step_latency)

            if step_latencies:
                step.append(mean(step_latencies))
            else:
                # assert steps == 1
                step.append(0)
            weights.append(len(step_latencies))

        self._step_regression = linear_model.LinearRegression(positive=True, fit_intercept=False)
        self._step_regression.fit(xs, step, weights)

        self._first_step_regression = linear_model.LinearRegression(
            positive=True, fit_intercept=False
        )
        self._first_step_regression.fit(xs, first_step)

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

