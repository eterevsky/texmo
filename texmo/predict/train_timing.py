import argparse
import json
import logging
import math
import random
from collections import namedtuple
from statistics import mean
from typing import Iterable, Optional

import jax.numpy as jnp
import numpy as np
import scipy
from jax.experimental.sparse import BCOO
from jax.scipy.optimize import minimize
from scipy.optimize import lsq_linear, nnls
from scipy.sparse import csr_array
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .. import latency
from ..common import INF, total_size
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
        avg_log = self._avg_pred.predict([features])
        return 2 ** avg_log[0]

    @property
    def total_samples(self) -> int:
        return len(self._run_timings)

    def predict(self, conf: Configuration) -> tuple[float, float]:
        with latency.timer("TrainTiming.predict"):
            if self.total_samples == 0:
                # If we don't have any data at all, return 100 ms and 1 ms
                # as a default.
                return 0.1, 0.001

            run = conf_to_run_timing(conf)
            total_avg = 0

            for layer in self._run_timing_to_layers(run):
                total_avg += self.predict_layer(layer)

            return total_avg

    def _prepare_model_layer_data(self):
        """Build matrices for training splitting model latency into layer latency

        Produces two datasets, for the latency of the first training steps and
        the average step (after the first one). Each dataset contains:
          - a sparse matrix with models as rows and layers as columsn, with the number of time a given layer appears in the a given model;
          - latency of training the model
        """
        with latency.timer("TrainTiming._prepare_model_layer_data"):
            avg_xs = []
            avg_rows = []
            avg_cols = []
            avg_ys = []

            avg_row = 0

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

                if run_timing.avg_step is not None:
                    for col, value in layer_counts.items():
                        avg_rows.append(avg_row)
                        avg_cols.append(col)
                        avg_xs.append(value)
                    avg_ys.append(run_timing.avg_step)
                    avg_row += 1

            avg_xs = csr_array(
                (avg_xs, (avg_rows, avg_cols)),
                shape=(len(avg_ys), len(self._layers)),
            )
            # avg_xs = BCOO.from_scipy_sparse(avg_xs)
            avg_ys = np.array(avg_ys)

            return avg_xs, avg_ys

    def _optimize_layer_split_jax(
        self, model_layer_mat: BCOO, model_time: np.ndarray
    ) -> np.ndarray:
        """Given a sparse matrix with the model-layer correspondence and model timing, find layer timing."""

        assert isinstance(model_layer_mat, BCOO)
        nmodels, nlayers = model_layer_mat.shape

        model_time_log = jnp.log2(model_time)

        def loss(layer_time_log):
            layer_time = 2**layer_time_log
            pred_model_time = model_layer_mat @ layer_time
            pred_model_time_log = jnp.log2(pred_model_time)
            return jnp.sum(jnp.abs(pred_model_time_log - model_time_log))

        initial_guess = jnp.ones(shape=(nlayers,)) * math.log2(0.001)

        results = minimize(
            loss, initial_guess, method="BFGS", options={"maxiter": 40}
        )

        avg_loss = results.fun / nmodels
        loss_pct = (2**avg_loss - 1) * 100

        logging.info(
            f"nfev = {results.nfev} njev = {results.njev} nit = {results.nit}"
        )
        logging.info(f"loss = {results.fun} avg_loss = {avg_loss}, {loss_pct}%")

        layer_time = 2**results.x
        logging.info(f"Layer time: {layer_time}")

        return layer_time

    def _optimize_layer_split(
        self, model_layer_mat: csr_array, model_time: np.ndarray
    ) -> np.ndarray:
        assert isinstance(model_layer_mat, csr_array)
        nmodels, nlayers = model_layer_mat.shape
        result = lsq_linear(model_layer_mat, model_time, bounds=(0, INF))
        layer_time = result.x

        pred = model_layer_mat @ layer_time
        loss = np.sum(np.abs(np.log2(pred) - np.log2(model_time))) / nmodels
        loss_pct = (2**loss - 1) * 100
        logging.info(f"Loss on training data: {loss_pct:.1f}%")

        return layer_time

    def _train_layer_timing(self) -> tuple[np.ndarray, np.ndarray]:
        """Run a linear regression to separate network timings into layer timings.

        Returns:
            A tuple of coefficients for first step timing and average step timing.
        """
        avg_xs, avg_ys = self._prepare_model_layer_data()

        with latency.timer("TrainTiming._train_layer_timing.fit"):
            logging.info(
                "Running linear regression for per-layer avg_step timing"
            )
            layer_avg_step = self._optimize_layer_split(avg_xs, avg_ys)

        return layer_avg_step

    def train(self):
        logging.info("Training a model to predict training latency")

        avg_step = self._train_layer_timing()
        self._layer_avg_step = avg_step

        with latency.timer("TrainTiming.train.fit"):
            avg_step = np.maximum(avg_step, 0.0001)
            avg_step_log = np.log2(avg_step)

            features = []
            for layer in self._layers:
                features.append(self._get_features_for_layer(layer))

            features = np.array(features)

            self._avg_pred.fit(features, avg_step_log)
