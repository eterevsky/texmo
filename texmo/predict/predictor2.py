import logging
import math
import numpy as np
from typing import Optional, Iterable

from ..configuration import conf_tokens_name, Configuration
from ..record import TrainingRecord
from ..resultdb import ResultDB
from .loss_predictor_flat import LossPredictorFlat
from .sample_timing import SampleTiming
from .train_timing import TrainTiming


class Predictor2(object):
    """Predict number of training steps in a given time.

    Umbrella class over SampleTiming, TrainTiming and the loss model.
    """

    def __init__(
        self,
        sample_timing_path: Optional[str],
        train_timing_path: Optional[str],
        result_db: ResultDB,
    ):
        self._sample_timing = SampleTiming(sample_timing_path)
        self._train_timing = TrainTiming(train_timing_path)
        self._loss_predictor = LossPredictorFlat(result_db)

    def add_record(self, record: TrainingRecord):
        conf = record.conf
        token_set_name = conf_tokens_name(conf)
        sample_pred = self._sample_timing.predict(
            token_set_name, conf.sample_len, conf.batch
        )
        first_pred, avg_pred = self._train_timing.predict(conf)

        if first_pred > conf.t:
            steps = None
        else:
            avg_step = max(avg_pred, sample_pred)

            steps = math.ceil((conf.t - first_pred) / avg_step) + 1

        sample_pred *= 1000
        first_pred *= 1000
        avg_pred *= 1000

        logging.info(
            f"Prediction: steps {steps}, sample {sample_pred:.2f} ms, step {first_pred:.2f}|{avg_pred:.2f} ms"
        )

        self._sample_timing.add_sample_latency(
            token_set_name, conf.sample_len, conf.batch, record.avg_sample_time
        )
        self._train_timing.add_step_latency(
            conf, record.first_step_time, record.avg_step_time
        )
        self._loss_predictor.count_record()

    def predict_steps(self, conf: Configuration, debug: bool = False) -> Optional[int]:
        token_set_name = conf_tokens_name(conf)
        sample_pred = self._sample_timing.predict(
            token_set_name, conf.sample_len, conf.batch
        )
        first_pred, avg_pred = self._train_timing.predict(conf)

        if first_pred > conf.t:
            steps = None
        else:
            avg_step = max(avg_pred, sample_pred)

            steps = math.ceil((conf.t - first_pred) / avg_step) + 1

        if debug:
            sample_pred *= 1000
            first_pred *= 1000
            avg_pred *= 1000

            logging.info(
                f"Prediction: steps {steps}, sample {sample_pred:.2f} ms, step {first_pred:.2f}|{avg_pred:.2f} ms"
            )
        
        return steps

    def predict_loss(self, confs: Iterable[Configuration]) -> Iterable[tuple[Configuration, float]]:
        if not isinstance(confs, list):
            confs = list(confs)
        steps = [self.predict_steps(conf) for conf in confs]

        good_confs = []
        good_steps = []

        for conf, steps in zip(confs, steps):
            if steps is None:
                yield conf, None
            else:
                good_confs.append(conf)
                good_steps.append(steps)

        losses = self._loss_predictor.predict(good_confs, good_steps)
