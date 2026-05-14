"""Background thread that fits timing & loss models and writes estimates.

Opens its own `DbReader` and `DbWriter` connections (separate from
SearchThread's) so transactions on the two threads don't interleave
on a shared SQLite connection. Owns the writer side of the shared
`TrainTimingModel` and `LossModelHolder` references (passed in at
construction) and consumes `TrainMessage`s from its input queue:

    RunAdded(system, precision)
        Notification that a new run has been written to the DB. The
        thread tracks per-(system, precision) and global run counts and
        triggers refits when the corresponding threshold is crossed.

    LossRefit()
        Force a loss-model refit now (used at server startup when no
        persisted loss model is on disk).

    Stop()
        Exit the loop.

Median-backed estimates are also maintained by `DbWriter._add_run`, so
the thread can skip over confs with medians without touching their
estimate rows.

A full bootstrap — median backfill + fit all pairs + predictions for
confs without runs — is exposed as a synchronous `bootstrap()` function
so startup can block on it before serving requests.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue

from .. import latency
from ..configuration import Configuration
from ..db import DbReader, DbWriter
from ..precision import Precision
from . import loss_rnn
from .loss_rnn import LossModelHolder
from .timing import TrainTimingModel


# Threshold of new runs per (system, precision) pair that triggers a
# timing-model refit. Picked so refits are frequent enough to absorb
# genuinely new points but not so frequent that fit cost dominates.
_REFIT_EVERY = 100

# Threshold of total new runs (any system/precision) that triggers a
# retrain of the loss-prediction model.
_LOSS_REFIT_EVERY = 200


@dataclass
class RunAdded:
    """A new run was written to the DB. Producer: request handlers via
    SearchThread; consumer: ModelThread (counter bump + maybe refit)."""
    system: str
    precision: Precision


@dataclass
class LossRefit:
    """Force a loss-model refit now (used at startup)."""


@dataclass
class Stop:
    """Stop the model thread."""


TrainMessage = RunAdded | LossRefit | Stop


def _refresh_estimates(
    reader: DbReader,
    writer: DbWriter,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
):
    with latency.timer('timing._refresh_estimates.median_ids'):
        median_ids = reader.get_conf_ids_with_median_time(system, precision)
    with latency.timer('timing._refresh_estimates.iter_confs'):
        to_predict: list[tuple[int, Configuration]] = [
            (conf_id, conf)
            for conf_id, conf in reader.iter_confs_by_precision(precision)
            if conf_id not in median_ids
        ]
    if not to_predict:
        return
    with latency.timer('timing._refresh_estimates.predict_batch'):
        preds = model.predict_batch(system, [c for _, c in to_predict])
    if preds is None:
        # Below MIN_SAMPLES — no model yet, nothing to predict with.
        return
    with latency.timer('timing._refresh_estimates.build_rows'):
        # `conf_time_estimate.time_s` stores total train_time, which is
        # what the model predicts directly.
        rows = [
            (conf_id, system, float(preds[i]))
            for i, (conf_id, conf) in enumerate(to_predict)
        ]
    with latency.timer('timing._refresh_estimates.upsert'):
        # Predicted-only upsert: a 'median' row written between our
        # `get_conf_ids_with_median_time` snapshot and now would
        # otherwise be silently overwritten by REPLACE.
        writer.upsert_predicted_time_estimates(rows)
    logging.info(
        f"refreshed {len(rows)} predicted estimates "
        f"for ({system!r}, {precision})"
    )


def _refit_pair(
    reader: DbReader,
    writer: DbWriter,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
):
    with latency.timer('timing._refit_pair.get_runs'):
        runs = list(reader.get_runs_for_timing(system, precision))
    if not runs:
        return
    with latency.timer('timing._refit_pair.fit'):
        model.fit(runs, verbose=False)
    _refresh_estimates(reader, writer, model, system, precision)


def bootstrap(
    reader: DbReader, writer: DbWriter, timing_model: TrainTimingModel,
) -> None:
    """Bring the DB into a consistent state before starting the search.

    1. Backfill medians: recompute median_score and median_time (-> the
       'median' rows of conf_time_estimate) for every (conf, system) pair
       with runs. This matters after a schema migration — new estimate
       rows are written per-run by `add_run`, but pre-existing data only
       shows up here.
    2. Fit the timing model for every (system, precision) pair with
       enough runs (mutates `timing_model` in place).
    3. Write 'predicted' estimates for every conf without a median.
    """
    logging.info("Bootstrap: backfilling medians")
    writer.update_all_scores()

    logging.info("Bootstrap: fitting timing models")
    for system in reader.get_systems():
        for precision in Precision:
            _refit_pair(reader, writer, timing_model, system, precision)
    writer.save_model('timing', timing_model.snapshot())


class ModelThread(threading.Thread):
    def __init__(
        self,
        db_path: str | None,
        queue: Queue,
        timing_model: TrainTimingModel,
        loss_model: LossModelHolder,
    ):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        # Shared references with the Search threads. Per-(system,
        # precision) weight inserts on `timing_model` are atomic;
        # `loss_model.set_model(...)` is a single pointer swap.
        self._timing_model = timing_model
        self._loss_model = loss_model
        # Per-(system, precision) and total run counters. Private to
        # this thread, so no locking needed.
        self._run_counter: dict[tuple[str, Precision], int] = defaultdict(int)
        self._total_run_counter = 0

    def run(self):
        reader = DbReader(self._db_path)
        writer = DbWriter(self._db_path)
        logging.info("Started model thread")
        try:
            while True:
                m = self._queue.get()
                match m:
                    case RunAdded(system=s, precision=p):
                        self._on_run_added(reader, writer, s, p)
                    case LossRefit():
                        self._refit_loss(reader, writer)
                    case Stop():
                        logging.info("Stopping model thread")
                        break
                    case _:
                        assert False, f"Unknown message: {m!r}"
        finally:
            reader.close()
            writer.close()

    def _on_run_added(
        self, reader: DbReader, writer: DbWriter,
        system: str, precision: Precision,
    ):
        key = (system, precision)
        self._run_counter[key] += 1
        if self._run_counter[key] >= _REFIT_EVERY:
            self._run_counter[key] = 0
            _refit_pair(reader, writer, self._timing_model, system, precision)
            writer.save_model('timing', self._timing_model.snapshot())
        self._total_run_counter += 1
        if self._total_run_counter >= _LOSS_REFIT_EVERY:
            self._total_run_counter = 0
            self._refit_loss(reader, writer)

    def _refit_loss(self, reader: DbReader, writer: DbWriter):
        loss_model = loss_rnn.train_loss_model(reader)
        if loss_model is not None:
            writer.save_model('loss', loss_model)
            self._loss_model.set_model(loss_model)
            logging.info(
                f"Published new loss model "
                f"(max_layers={loss_model.max_layers}, "
                f"{len(loss_model.simple_types)} simple types)"
            )
