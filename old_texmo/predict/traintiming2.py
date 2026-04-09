import argparse
import json
import logging
from collections import namedtuple
from typing import Optional
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.optimize import minimize

from .. import latency
from ..common import INF
from ..configuration import Configuration, conf_to_string
from ..model2 import Model2, build_model
from ..tokens import get_tokenizer, set_tokens_dir

RunTiming = namedtuple(
    "RunTiming",
    [
        "spec",
        "ntokens",
        "sample_len",
        "batch",
        "avg_step",
    ],
)


def conf_to_run_timing(
    conf: Configuration,
    avg_step: float,
) -> RunTiming:
    # assert isinstance(avg_step, float)
    return RunTiming(
        spec=str(conf.model),
        ntokens=conf.ntokens,
        sample_len=conf.sample_len,
        batch=conf.batch,
        avg_step=avg_step,
    )


MAX_LAYERS = 16
_LAYER_FEATURES = [
    ("input", [0] * 8),  # ntokens * sample_len * batch
    ("output", [0] * 16),  # input * ntokens * sample_len * batch
    ("suffix", [0] * 16),  # input * length * sample_len * batch
    ("dense.relu", [0] * 16),  # input * size * sample_len * batch
    ("dense.gelu", [0] * 16),  # input * size * sample_len * batch
    ("dense.tanh", [0] * 16),  # input * size * sample_len * batch
    ("rec.relu", [0] * 16),  # input * size * sample_len * batch
    ("rec.gelu", [0] * 16),  # input * size * sample_len * batch
    ("rec.tanh", [0] * 16),  # input * size * sample_len * batch
    ("gru", [0] * 16),  # input * size * sample_len * batch
    ("mgru", [0] * 16),  # input * size * sample_len * batch
    ("lstm", [0] * 16),  # input * size * sample_len * batch
    ("attn", [0] * 64),  # input * size * heads * length * sample_len * batch
    ("attnmq", [0] * 64),  # input * size * heads * length * sample_len * batch
]
_N_LAYER_FEATURES = sum(len(z) for _, z in _LAYER_FEATURES)
_ALL_ZEROS = [0] * _N_LAYER_FEATURES


def _make_layer_features(
    name: str, dims: list[int]
) -> tuple[list[int], list[bool]]:
    features = []

    for layer_name, zeros in _LAYER_FEATURES:
        if name == layer_name:
            for prod_idx in range(2 ** len(dims)):
                prod = 1
                for i in range(len(dims)):
                    if prod_idx & 1:
                        prod *= dims[i]
                    prod_idx //= 2
                features.append(prod)
        else:
            features.extend(zeros)
    return features


def _make_features(
    model: Model2, ntokens: int, sample_len: int, batch: int, max_layers: int
) -> np.ndarray:
    layer_features = []
    layer_features.append(
        _make_layer_features("input", [ntokens, sample_len, batch])
    )
    for layer in model.layers:
        name = layer.name
        if name in ("dense", "rec"):
            name += layer._activation_suffix
        dims = [layer.input_size, sample_len, batch]
        if layer.name != "suffix":
            dims.append(layer.size)
        if layer.name in ("suffix", "attn", "attnmq"):
            dims.append(layer.length)
        if layer.name in ("attn", "attnmq"):
            dims.append(layer.heads)

        layer_features.append(_make_layer_features(name, dims))

    while len(layer_features) < max_layers:
        layer_features.append(_ALL_ZEROS)

    return np.array(layer_features)


def make_conf_features(conf: Configuration, max_layers=0) -> np.ndarray:
    assert isinstance(conf, Configuration)
    return _make_features(conf.model, conf.ntokens, conf.sample_len, conf.batch, max_layers)


def _make_run_timing_features(run_timing: RunTiming) -> np.ndarray:
    return _make_features(
        build_model(run_timing.ntokens, run_timing.spec),
        run_timing.ntokens,
        run_timing.sample_len,
        run_timing.batch,
        max_layers=MAX_LAYERS,
    )


# def _predict_one(weights, features):
#     assert features.shape == (MAX_LAYERS, _N_LAYER_FEATURES)

#     coef = weights[:_N_LAYER_FEATURES]
#     bias = weights[_N_LAYER_FEATURES:]
#     assert coef.shape == (_N_LAYER_FEATURES,)
#     assert bias.shape == (_N_LAYER_FEATURES,)
#     coef = np.expand_dims(coef, axis=0)
#     bias = np.expand_dims(bias, axis=0)
#     # component_time = np.maximum(coef * features + bias, 0)
#     # component_time = jax.nn.relu(coef * features + bias)
#     # component_time = jax.nn.elu(coef * features + bias) + 1
#     component_time = coef * features + bias
#     return np.sum(component_time)


def _predict_batch(weights, features_batch):
    assert len(features_batch.shape) == 3
    assert features_batch.shape[1] == MAX_LAYERS
    assert features_batch.shape[2] == _N_LAYER_FEATURES

    # pos_weights = jnp.exp(weights)

    coef = weights[:_N_LAYER_FEATURES]
    # bias = jnp.exp(weights[_N_LAYER_FEATURES:])

    coef = jnp.expand_dims(coef, axis=(0, 1))
    # bias = jnp.expand_dims(bias, axis=(0, 1))

    # To force coefficients to be positive
    coef = coef * coef
    # bias = bias * bias

    # component_time = jax.nn.relu(coef * features_batch - bias)
    component_time = jnp.where(
        features_batch > 0, coef * features_batch, 0
    )
    # component_time = jax.nn.relu(component_time) + jnp.tanh(component_time) * 0.001

    return jnp.sum(component_time, axis=(1, 2))
    # component_time = jax.nn.relu(coef * features_batch + bias)
    # component_time = coef * features_batch + bias
    # component_time = 0.001 * (jax.nn.elu(coef * features_batch + bias) + 1)
    # s = jnp.sum(component_time, axis=(1, 2))
    # return (jax.nn.elu(s) + 1) * 0.001


# def _predict_one_by_batch(weights: np.ndarray, features: list[float]) -> float:
#     features = jnp.array([features], dtype=jnp.float64)
#     p = _predict_batch(weights, features)
#     return p[0]


def _predict_one(weights: np.ndarray, features: list[float]) -> float:
    assert isinstance(weights, np.ndarray)
    # features = np.array(features, dtype=np.float64)
    features = np.array(features)
    coef = weights[:_N_LAYER_FEATURES]
    coef = np.expand_dims(coef, axis=(0, 1))
    coef = coef * coef
    component_time = np.where(features > 0, coef * features, 0)
    return np.sum(component_time)


def _loss(weights: dict, features_batch, times):
    pred_times = _predict_batch(weights, features_batch)
    return jnp.mean(jnp.abs(jnp.log2(pred_times) - jnp.log2(times)))
    # return jnp.mean((pred_times - times)**2)


def _loss_log(weights: dict, features_batch, times):
    pred_times = _predict_batch(weights, features_batch)
    pred_times = jnp.maximum(pred_times, 0.001)
    return jnp.mean(jnp.abs(jnp.log2(pred_times) - jnp.log2(times)))
    # return jnp.mean((pred_times - times)**2)


def _loss_norm(weights: dict, features_batch, times):
    return _loss(weights, features_batch, times) + 1e-4 * jnp.sum(weights**2)


def _show_loss(loss: float) -> str:
    loss_pct = (2**loss - 1) * 100
    return f"{loss_pct:.2f}%"


def _draw_train_graph(step_loss: list[float]):
    fig, ax = plt.subplots()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(range(len(step_loss)), step_loss)

    plt.show()


def _train(
    features, times, weights: Optional[dict], steps: int, lr: float = 0.00001
) -> dict:
    if weights is None:
        weights = 0.0001 * jnp.ones((_N_LAYER_FEATURES,))
        # weights = jnp.concatenate(
        #     0.0001 * jnp.ones((_N_LAYER_FEATURES,), dtype=jnp.float64),
        #     jnp.zeros((_N_LAYER_FEATURES,), dtype=jnp.float64),
        # )

    if not features:
        return jax.device_get(weights)

    # features = jnp.array(features, dtype=jnp.float64)
    # times = jnp.array(times, dtype=jnp.float64)
    features = jnp.array(features)
    times = jnp.array(times)

    loss = lambda weights: _loss(weights, features, times)
    loss_grad = jax.jit(jax.value_and_grad(loss))

    optimizer = optax.adamw(lr, weight_decay=0.01)
    opt_state = optimizer.init(weights)
    step_loss = []
    best_weights = weights
    best_loss = INF

    logging.info(f"Training timing model for {steps} steps")

    for _ in range(steps):
        loss, grads = loss_grad(weights)

        if loss < best_loss:
            best_weights = weights
            best_loss = loss

        step_loss.append(loss)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

    # results = minimize(loss, weights, method="BFGS", options={"gtol": 1E-7})  #, options={"disp": True, })
    # best_weights = results.x

    show_weights = best_weights**2

    pure_loss = _loss_log(best_weights, features, times)

    logging.info(f"final loss: {pure_loss}")
    # _draw_train_graph(step_loss)

    return jax.device_get(best_weights)


class TrainTiming2(object):
    def __init__(self, jsonl_path=None):
        self._avg_features = []
        self._avg_times = []
        self._avg_test_features = []
        self._avg_test_times = []

        self._avg_weights = None

        if jsonl_path is not None:
            try:
                with open(jsonl_path, "r") as f:
                    for line in f:
                        run_json = json.loads(line)
                        if "first_step" in run_json:
                            del run_json["first_step"]
                        run = RunTiming(**run_json)
                        features = _make_run_timing_features(run)
                        self._add_sample(features, run.avg_step)
            except FileNotFoundError:
                pass
            self._file = open(jsonl_path, "a", encoding="utf-8", newline="\n")
        else:
            self._file = None

    def _add_sample(
        self, features: np.ndarray, avg_step: Optional[float]
    ):
        assert isinstance(features, np.ndarray)

        if random.random() < 0.9:
            if avg_step is not None:
                self._avg_features.append(features)
                self._avg_times.append(avg_step)
        else:
            if avg_step is not None:
                self._avg_test_features.append(features)
                self._avg_test_times.append(avg_step)

    def add_step_latency(
        self, conf: Configuration, avg_step: float
    ):
        if self._file is not None:
            run = conf_to_run_timing(conf, avg_step)
            print(json.dumps(run._asdict()), file=self._file)

        self._add_sample(make_conf_features(conf, max_layers=MAX_LAYERS), avg_step)

    @property
    def total_samples(self) -> int:
        return len(self._avg_features)

    def predict(self, conf: Configuration) -> float:
        assert self._avg_weights is not None
        features = make_conf_features(conf)
        avg_pred = _predict_one(self._avg_weights, features)
        return avg_pred

    def train(self):
        logging.info("Updating train timing model")
        self._avg_weights = _train(
            features=self._avg_features,
            times=self._avg_times,
            weights=self._avg_weights,
            steps=500 if self._avg_weights is None else 50,
            lr=0.000005,
        )

        if not self._avg_times:
            return

        train_loss = _loss_log(
            self._avg_weights,
            jnp.array(self._avg_features),
            jnp.array(self._avg_times),
        )

        if not self._avg_test_times:
            return

        test_loss = _loss_log(
            self._avg_weights,
            jnp.array(self._avg_test_features),
            jnp.array(self._avg_test_times),
        )
        logging.info(
            f"Average step model: train loss {_show_loss(train_loss)} "
            + f"test loss {_show_loss(test_loss)}"
        )


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)

    logging.info("Creating new training time predictor")
    predictor = TrainTiming2(args.train_timing)

    logging.info("Training")
    predictor.train()

    tokenizer = get_tokenizer(args.token_set)
    conf = Configuration(
        build_model(tokenizer.token_set.ntokens, args.spec),
        ntokens=tokenizer.token_set.ntokens,
        token_type=tokenizer.token_set.token_type,
        token_processing=tokenizer.token_set.processing,
        lr=0.0625,
        sample_len=args.sample_len,
        batch=args.batch,
        t=1,
    )
    logging.info("Predicting step time for the model: " + conf_to_string(conf))

    avg_step = predictor.predict(conf)
    logging.info(f"Time: {1000 * avg_step} ms")

    latency.report()


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default="tokens",
        help="directory with token sets",
    )
    parser.add_argument(
        "--train-timing",
        type=str,
        default="results/train-timing.jsonl",
        help="a file with measured train timings for each trainined configuration",
    )

    parser.add_argument(
        "-s",
        "--spec",
        default=None,
        help="layer-by-layer model specification",
    )
    parser.add_argument(
        "--token-set", required=True, type=str, help="token set name"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        metavar="N",
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training",
    )

    parser.set_defaults(func=main)
