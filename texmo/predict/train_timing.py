import json
import logging
import random
from collections import namedtuple
from statistics import mean
from typing import Iterable, Optional

import numpy as np
from scipy.sparse import csr_array
from scipy.optimize import nnls, lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .. import latency
from ..common import total_size, INF
from ..configuration import Configuration
from ..model2 import Model2, build_model

RunTiming = namedtuple(
    "RunTiming",
    [
        "spec",
        "ntokens",
        "sample_len",
        "batch",
        "first_step",
        "avg_step",
    ],
)


def conf_to_run_timing(
    conf: Configuration,
    first_step: Optional[float] = None,
    avg_step: Optional[float] = None,
) -> RunTiming:
    return RunTiming(
        spec=str(conf.model),
        ntokens=conf.ntokens,
        sample_len=conf.sample_len,
        batch=conf.batch,
        first_step=first_step,
        avg_step=avg_step,
    )


LayerTiming = namedtuple(
    "LayerTiming", ["layer", "input", "output", "sample_len", "batch"]
)


class TrainTiming(object):
    def __init__(self, jsonl_path=None):
        self._run_timings = []
        if jsonl_path is not None:
            try:
                with open(jsonl_path, "r") as f:
                    for line in f:
                        run = RunTiming(**json.loads(line))
                        self._run_timings.append(run)
            except FileNotFoundError:
                pass
            self._file = open(jsonl_path, "a", encoding="utf-8", newline="\n")
        else:
            self._file = None

        self._last_train = 0
        self._layer_types = {}
        self._layers = []
        self._layer_to_idx = {}
        self._layer_first_step = []
        self._layer_avg_step = []

        self._first_pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            # max_depth=None,
            max_leaf_nodes=63,
            max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            # warm_start=False,
            # early_stopping=False,
            categorical_features=[True, False, False, False, False],
            monotonic_cst=[0, 1, 1, 1, 1],
        )
        self._avg_pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            # max_depth=None,
            max_leaf_nodes=63,
            max_iter=100,
            # n_iter_no_change=20,
            # learning_rate=0.1,
            # warm_start=False,
            # early_stopping=False,
            categorical_features=[True, False, False, False, False],
            monotonic_cst=[0, 1, 1, 1, 1],
        )

    def _run_timing_to_layers(self, run: RunTiming) -> Iterable[LayerTiming]:
        model: Model2 = build_model(run.ntokens, run.spec)
        yield LayerTiming(
            layer="io",
            input=run.ntokens,
            output=run.ntokens,
            sample_len=run.sample_len,
            batch=run.batch,
        )
        for layer in model.layers:
            output_size = total_size(layer.output_shape)
            yield LayerTiming(
                layer=layer.name,
                input=layer.input_size,
                output=output_size,
                sample_len=run.sample_len,
                batch=run.batch,
            )
            if layer.name in ("dense", "rec"):
                yield LayerTiming(
                    layer=layer._activation_suffix,
                    input=output_size,
                    output=output_size,
                    sample_len=run.sample_len,
                    batch=run.batch,
                )
        layer = model.out_layer
        yield LayerTiming(
            layer=layer.name,
            input=layer.input_size,
            output=run.ntokens,
            sample_len=run.sample_len,
            batch=run.batch,
        )

    def add_step_latency(
        self, conf: Configuration, first_step: float, avg_step: float
    ):
        run = conf_to_run_timing(conf, first_step, avg_step)
        self._run_timings.append(run)
        if self._file is not None:
            print(json.dumps(run._asdict()), file=self._file)

        return first_step, avg_step

    def _get_features_for_layer(self, layer: LayerTiming) -> np.array:
        type_idx = self._layer_types.get(layer.layer)
        if type_idx is None:
            type_idx = len(self._layer_types)
            self._layer_types[layer.layer] = type_idx
        return np.log2(
            np.array(
                [
                    2**type_idx,
                    layer.input,
                    layer.output,
                    layer.sample_len,
                    layer.batch,
                ]
            )
        )

    def predict_layer(self, layer: LayerTiming) -> float:
        layer_idx = self._layer_to_idx.get(layer)
        if layer_idx is not None:
            return (
                self._layer_first_step[layer_idx],
                self._layer_avg_step[layer_idx],
            )
        features = self._get_features_for_layer(layer)
        first_log = self._first_pred.predict([features])
        avg_log = self._avg_pred.predict([features])
        return 2 ** first_log[0], 2 ** avg_log[0]

    def predict(self, conf: Configuration) -> tuple[float, float]:
        with latency.timer("TrainTiming.predict"):
            total_timings = len(self._run_timings)
            if total_timings == 0:
                # If we don't have any data at all, return 100 ms and 1 ms
                # as a default.
                return 0.1, 0.001
            samples_since_last_traing = total_timings - self._last_train
            if samples_since_last_traing**3 >= self._last_train:
                self.train()

            run = conf_to_run_timing(conf)
            total_first, total_avg = 0, 0

            for layer in self._run_timing_to_layers(run):
                first_step, avg_step = self.predict_layer(layer)
                total_first += first_step
                total_avg += avg_step

            return total_first, total_avg

    def _train_layer_timing(self) -> tuple[np.ndarray, np.ndarray]:
        """Run a linear regression to separate network timings into layer timings.

        Returns:
            A tuple of coefficients for first step timing and average step timing.
        """
        with latency.timer("TrainTiming._train_layer_timing.prepare"):
            avg_xs = []
            avg_rows = []
            avg_cols = []
            avg_ys = []
            first_xs = []
            first_rows = []
            first_cols = []
            first_ys = []

            avg_row = 0
            first_row = 0

            for run_timing in self._run_timings:
                if run_timing.first_step is None:
                    continue
                layer_counts = {}
                for layer in self._run_timing_to_layers(run_timing):
                    idx = self._layer_to_idx.get(layer)
                    if idx is None:
                        idx = len(self._layers)
                        self._layers.append(layer)
                        self._layer_to_idx[layer] = idx
                    layer_counts[idx] = layer_counts.get(idx, 0) + 1

                for col, value in layer_counts.items():
                    first_rows.append(first_row)
                    first_cols.append(col)
                    first_xs.append(value)
                first_ys.append(run_timing.first_step)
                first_row += 1

                if run_timing.avg_step is not None:
                    for col, value in layer_counts.items():
                        avg_rows.append(avg_row)
                        avg_cols.append(col)
                        avg_xs.append(value)
                    avg_ys.append(run_timing.avg_step)
                    avg_row += 1

            first_xs = csr_array(
                (first_xs, (first_rows, first_cols)),
                shape=(len(first_ys), len(self._layers)),
            )
            first_ys = np.array(first_ys)

            avg_xs = csr_array(
                (avg_xs, (avg_rows, avg_cols)),
                shape=(len(avg_ys), len(self._layers)),
            )
            avg_ys = np.array(avg_ys)

        with latency.timer("TrainTiming._train_layer_timing.fit"):
            logging.info(
                "Running linear regression for per-layer first_step timing"
            )
            n = first_xs.shape[0]
            nlayers = len(self._layers)
            logging.info(f"Using {n} samples / {nlayers} layers")
            # first_coef, _ = nnls(first_xs, first_ys)
            # first_coef = nnls_grad(first_xs, first_ys)
            result = lsq_linear(first_xs, first_ys, bounds=(0, INF))
            first_coef = result.x
            # first_regression = LinearRegression(positive=True, fit_intercept=False, n_jobs=4)
            # first_regression.fit(first_xs, first_ys)
            # print(first_coef.shape)
            # print(first_coef)

            logging.info(
                "Running linear regression for per-layer avg_step timing"
            )
            n = avg_xs.shape[0]
            logging.info(f"Using {n} samples / {nlayers} layers")

            # avg_coef, _ = nnls(avg_xs, avg_ys)
            # avg_coef = nnls_grad(avg_xs, avg_ys)
            result = lsq_linear(avg_xs, avg_ys, bounds=(0, INF))
            avg_coef = result.x
            # xtx = np.matmul(avg_xs.transpose(), avg_xs)
            # xty = np.matmul(avg_xs.transpose(), avg_ys)
            # avg_coef, _ = fnnls(xtx, xty)
            # print(avg_coef.shape)
            # print(avg_coef)

            # avg_regression = LinearRegression(positive=True, fit_intercept=False, n_jobs=4)
            # avg_regression.fit(avg_xs, avg_ys)

        return first_coef, avg_coef
        # return first_regression.coef_, avg_regression.coef_

    def train(self):
        with latency.timer("TrainTiming.train"):
            self._last_train = len(self._run_timings)

            logging.info("Training a model to predict layer latency")

            first_step, avg_step = self._train_layer_timing()
            self._layer_first_step = first_step
            self._layer_avg_step = avg_step

            first_step = np.maximum(first_step, 0.0001)
            avg_step = np.maximum(avg_step, 0.0001)
            first_step_log = np.log2(first_step)
            avg_step_log = np.log2(avg_step)

            features = []
            for layer in self._layers:
                features.append(self._get_features_for_layer(layer))

            features = np.array(features)

            self._first_pred.fit(features, first_step_log)
            self._avg_pred.fit(features, avg_step_log)
