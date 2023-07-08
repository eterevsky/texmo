import json
import logging
import math
from collections import namedtuple
from statistics import median
from typing import Optional

import numpy as np
from jax import numpy as jnp
from sklearn.ensemble import HistGradientBoostingRegressor

from ..configuration import Configuration, conf_tokens_name
from ..prng import Rng
from ..tokens import get_tokenizer
from .. import latency

SampleLatency = namedtuple(
    "SampleLatency",
    [
        "type",
        "processing",
        "ntokens",
        "batch",
        "sample_len",
        "latency",
    ],
)


def build_sample_latency(
    token_set_name: str,
    sample_len: int,
    batch: int,
    latency: Optional[float] = None,
) -> SampleLatency:
    token_set = get_tokenizer(token_set_name).token_set

    return SampleLatency(
        type=token_set.token_type,
        processing=token_set.processing,
        ntokens=token_set.ntokens,
        batch=batch,
        sample_len=sample_len,
        latency=latency,
    )


class SampleTiming(object):
    def __init__(self, jsonl_path=None):
        self._types: list[str] = []
        self._processing_types: list[str] = []
        self._sample_timings: dict[SampleLatency, list[float]] = {}

        if jsonl_path is not None:
            try:
                with open(jsonl_path, "r") as f:
                    for line in f:
                        sample = SampleLatency(**json.loads(line))
                        self._add_sample(sample)
            except FileNotFoundError:
                pass
            self._file = open(jsonl_path, "a", encoding="utf-8", newline="\n")
        else:
            self._file = None

        self._last_train = 0
        self._pred = HistGradientBoostingRegressor(
            loss="absolute_error",
            categorical_features=[True, True, False, False, False],
            monotonic_cst=[0, 0, 0, 1, 1],
        )

    def _add_sample(self, sample: SampleLatency):
        latency = sample.latency
        sample = sample._replace(latency=None)
        if sample not in self._sample_timings:
            self._sample_timings[sample] = []
        self._sample_timings[sample].append(latency)
        if sample.type not in self._types:
            self._types.append(sample.type)
        if sample.processing not in self._processing_types:
            self._processing_types.append(sample.processing)

    def _get_features(self, sample: SampleLatency) -> np.ndarray:
        try:
            type_idx = self._types.index(sample.type)
        except ValueError:
            type_idx = len(self._types)
            self._types.append(sample.type)

        try:
            processing_idx = self._processing_types.index(sample.processing)
        except ValueError:
            processing_idx = len(self._processing_types)
            self._processing_types.append(sample.processing)

        return np.array(
            [
                type_idx,
                processing_idx,
                math.log2(sample.ntokens),
                math.log2(sample.batch),
                math.log2(sample.sample_len),
            ]
        )

    def _train(self):
        with latency.timer("SampleTiming._train.prepare"):
            features = []
            log_latencies = []

            for sample, latencies in self._sample_timings.items():
                f = self._get_features(sample)
                for l in latencies:
                    features.append(f)
                    log_latencies.append(math.log2(l))

            features = np.array(features)
            log_latencies = np.array(log_latencies)

        with latency.timer("SampleTiming._train.fit"):
            self._pred.fit(features, log_latencies)

    def predict(
        self, token_set_name: str, sample_len: int, batch: int
    ) -> float:
        """Predict the latency of sample generation."""
        with latency.timer("SampleTiming.predict"):
            sample = build_sample_latency(token_set_name, sample_len, batch)
            latencies = self._sample_timings.get(sample)
            if latencies is not None:
                # logging.info(f"Predicting sampling latency from historical data: {latencies}")
                return median(latencies)

            total_samples = sum(
                len(v) for v in self._sample_timings.values()
            )

            if total_samples == 0:
                # Return 1 ms by default if we don't have any data at all.
                return 0.001

            samples_since_last_train = total_samples - self._last_train
            if samples_since_last_train**3 >= self._last_train:
                self._train()
            self._last_train = total_samples

            features = self._get_features(sample)
            logging.info(f"Predicting sampling latency by the model ({features})")
            latency_log = self._pred.predict([features])
            return 2**latency_log[0]

    def add_sample_latency(
        self,
        token_set_name: str,
        sample_len: int,
        batch: int,
        latency: float,
    ):
        assert isinstance(latency, float)

        sample = build_sample_latency(
            token_set_name, sample_len, batch, latency
        )
        self._add_sample(sample)
        if self._file is not None:
            print(json.dumps(sample._asdict()), file=self._file)
