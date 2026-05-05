"""Background thread that fits timing & loss models and writes estimates.

Opens its own `ResultDB` connection (separate from the SearchThread's)
so transactions on the two threads don't interleave on a shared SQLite
connection. Owns an in-memory `TrainTimingModel` and handles these
messages:

    ("refit", (system, precision))
        Reload runs for that (system, precision), refit just that pair's
        timing model, then overwrite `conf_time_estimate` for every
        conf of that precision on that system with either the median
        (if runs exist) or a prediction. After refitting, push a
        snapshot of the current timing model to the search queue as
        ("timing_weights", snapshot).

    ("loss_refit", None)
        Retrain the loss-prediction RNN on all labeled runs and push
        the new LossModel back to the search thread's queue as
        ("loss_weights", loss_model).

    ("stop", None)
        Exit the loop.

Median-backed estimates are also maintained by `ResultDB._add_run`, so
the thread can skip over confs with medians without touching their
estimate rows.

A full bootstrap — median backfill + fit all pairs + predictions for
confs without runs — is exposed as a synchronous `bootstrap()` function
so startup can block on it before serving requests.
"""

import logging
import threading
from queue import Queue
from typing import Optional

from . import latency
from .configuration import Configuration
from .precision import Precision
from .predict import loss_rnn
from .predict.timing import TrainTimingModel
from .resultdb import ResultDB


def _refresh_estimates(
    db: ResultDB,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
):
    with latency.timer('timing._refresh_estimates.median_ids'):
        median_ids = db.get_conf_ids_with_median_time(system, precision)
    with latency.timer('timing._refresh_estimates.iter_confs'):
        to_predict: list[tuple[int, Configuration]] = [
            (conf_id, conf)
            for conf_id, conf in db.iter_confs_by_precision(precision)
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
        db.upsert_predicted_time_estimates(rows)
    logging.info(
        f"refreshed {len(rows)} predicted estimates "
        f"for ({system!r}, {precision})"
    )


def _refit_pair(
    db: ResultDB,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
):
    with latency.timer('timing._refit_pair.get_runs'):
        runs = list(db.get_runs_for_timing(system, precision))
    if not runs:
        return
    with latency.timer('timing._refit_pair.fit'):
        model.fit(runs, verbose=False)
    _refresh_estimates(db, model, system, precision)


def bootstrap(db: ResultDB) -> TrainTimingModel:
    """Bring the DB into a consistent state before starting the search.

    1. Backfill medians: recompute median_score and median_time (-> the
       'median' rows of conf_time_estimate) for every (conf, system) pair
       with runs. This matters after a schema migration — new estimate
       rows are written per-run by `add_run`, but pre-existing data only
       shows up here.
    2. Fit the timing model for every (system, precision) pair with
       enough runs.
    3. Write 'predicted' estimates for every conf without a median.

    Returns the fitted timing model so callers can install it on a
    live search without waiting for the first async refit to arrive.
    """
    logging.info("Bootstrap: backfilling medians")
    db.update_all_scores()

    logging.info("Bootstrap: fitting timing models")
    model = TrainTimingModel()
    for system in db.get_systems():
        for precision in Precision:
            _refit_pair(db, model, system, precision)
    db.save_model('timing', model.snapshot())
    return model


class ModelTrainingThread(threading.Thread):
    def __init__(
        self,
        db_path: str | None,
        queue: Queue,
        search_queue: Optional[Queue] = None,
    ):
        """`search_queue` is used to send trained model snapshots back
        to the search thread as ('loss_weights', LossModel) and
        ('timing_weights', TrainTimingModel snapshot)."""
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        self._search_queue = search_queue
        self._model = TrainTimingModel()

    def run(self):
        db = ResultDB(self._db_path)
        logging.info("Started model-training thread")
        try:
            while True:
                command, args = self._queue.get()
                if command == "refit":
                    system, precision = args
                    _refit_pair(db, self._model, system, precision)
                    snap = self._model.snapshot()
                    db.save_model('timing', snap)
                    if self._search_queue:
                        self._search_queue.put(("timing_weights", snap))
                elif command == "loss_refit":
                    loss_model = loss_rnn.train_loss_model(db)
                    if loss_model is not None:
                        db.save_model('loss', loss_model)
                        if self._search_queue:
                            self._search_queue.put(
                                ("loss_weights", loss_model))
                elif command == "stop":
                    logging.info("Stopping model-training thread")
                    break
                else:
                    assert False, f"Unknown command: {command}"
        finally:
            db.close()
