import argparse
import logging
import math
from typing import Iterable, Optional

import numpy as np

from .. import latency
from ..configuration import Configuration, conf_to_string, conf_tokens_name
from ..model2 import build_model
from ..record import TrainingRecord
from ..resultdb import ResultDB
from ..tokens import get_tokenizer, set_tokens_dir
from .loss_predictor_flat import LossPredictorFlat
from .sample_timing import SampleTiming
from .traintiming2 import TrainTiming2


class Predictor2(object):
    """Predict number of training steps in a given time.

    Umbrella class over SampleTiming, TrainTiming and the loss model.
    """

    def __init__(
        self,
        sample_timing_path: Optional[str],
        train_timing_path: Optional[str],
        result_db: ResultDB,
        extra_dbs: list[str],
    ):
        self._sample_timing = SampleTiming(sample_timing_path)
        self._train_timing = TrainTiming2(train_timing_path)
        # Applies to timing models.
        self._samples_till_next_train = 0
        self._loss_predictor = LossPredictorFlat(result_db, extra_dbs)

    def add_record(self, record: TrainingRecord):
        conf = record.conf
        token_set_name = conf_tokens_name(conf)

        self._show_predictions(conf, token_set_name)

        self._sample_timing.add_sample_latency(
            token_set_name, conf.sample_len, conf.batch, record.avg_sample_time
        )
        self._train_timing.add_step_latency(
            conf, record.avg_step_time
        )

    def _show_predictions(self, conf: Configuration, token_set_name: str):
        sample_pred = self._sample_timing.predict(
            token_set_name, conf.sample_len, conf.batch
        )
        avg_pred = self._train_timing.predict(conf)

        avg_step = max(avg_pred, sample_pred)
        steps = math.ceil(conf.t / avg_step)

        sample_pred *= 1000
        avg_pred *= 1000

        logging.info(
            f"Prediction: steps {steps}, sample {sample_pred:.2f} ms, step {avg_pred:.2f} ms"
        )

    def predict(
        self, confs: list[Configuration], verbose: bool = False
    ) -> list[float]:
        """Predicting losses for configurations"""
        steps = []
        for conf in confs:
            token_set_name = conf_tokens_name(conf)
            sample = self._sample_timing.predict(
                token_set_name, conf.sample_len, conf.batch
            )
            avg_step = self._train_timing.predict(conf)
            avg_step = max(avg_step, sample)
            assert avg_step > 0
            s = math.ceil(conf.t / avg_step) + 1
            steps.append(s)

            if verbose:
                sample_ms = sample * 1000
                first_step_ms = first_step * 1000
                avg_step_ms = avg_step * 1000
                logging.info(
                    "Predicting steps for configuration " + conf_to_string(conf)
                )
                logging.info(f"Sampling time: {sample_ms} ms")
                logging.info(
                    f"Training time: {first_step_ms} ms | {avg_step_ms} ms"
                )
                logging.info(f"Steps: {s}")

        losses = self._loss_predictor.predict(confs, steps)

        if verbose:
            logging.info(f"Predicted losses: {losses}")

        return losses

    def maybe_train(self) -> bool:
        trained_timing = False

        self._samples_till_next_train -= 1
        if self._samples_till_next_train <= 0:
            self._sample_timing.train()
            self._train_timing.train()
            total_samples = min(
                self._sample_timing.total_samples,
                self._train_timing.total_samples,
            )
            self._samples_till_next_train = int(total_samples ** (1 / 3))
            trained_timing = True

        trained_loss = self._loss_predictor.maybe_train()
        return trained_timing or trained_loss


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)

    logging.info(f"Loading results from {args.db}")
    result_db = ResultDB(args.db)

    if args.extra_db:
        extra_dbs = [args.extra_db]
    else:
        extra_dbs = []

    logging.info("Creating new predictor")
    predictor = Predictor2(
        args.sample_timing, args.train_timing, result_db, extra_dbs
    )

    logging.info("Training")
    predictor.maybe_train()

    tokenizer = get_tokenizer(args.token_set)
    conf = Configuration(
        build_model(tokenizer.token_set.ntokens, args.spec),
        ntokens=tokenizer.token_set.ntokens,
        token_type=tokenizer.token_set.token_type,
        token_processing=tokenizer.token_set.processing,
        lr=args.lr,
        sample_len=args.sample_len,
        batch=args.batch,
        t=args.time,
    )
    logging.info("Predicting loss for the model: " + conf_to_string(conf))

    predictor.predict([conf], verbose=True)

    latency.report()


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default="tokens",
        help="directory with token sets",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "--extra-db",
        type=str,
        default=None,
        help="path to additional result DBs from other machines",
    )
    parser.add_argument(
        "--train-timing",
        type=str,
        default="results/train-timing.jsonl",
        help="a file with measured train timings for each trainined configuration",
    )
    parser.add_argument(
        "--sample-timing",
        type=str,
        default="results/sample-timing.jsonl",
        help="a file with measured timings of sample preparation",
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
        "-l",
        "--lr",
        type=float,
        default="0.0625",
        help="learning rate, could be written as a float or as 2^-10",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="NTOKENS",
        help="length in tokens of text fragments used for training",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        metavar="SECONDS",
        help="time limit for training",
        default=None,
    )

    parser.set_defaults(func=main)
