"""Background thread that fits timing models and writes time estimates.

Opens its own `ResultDB` connection (separate from the SearchThread's)
so transactions on the two threads don't interleave on a shared SQLite
connection. Owns an in-memory `TrainTimingModel`. Receives messages on
a queue:

    ("refit", (system, precision))
        Reload runs for that (system, precision), refit just that pair's
        model, then overwrite `conf_time_estimate` for every conf of that
        precision on that system with either the median (if runs exist)
        or a prediction.

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

from .precision import Precision
from .predict.timing import TrainTimingModel
from .resultdb import ResultDB


def _refresh_estimates(
    db: ResultDB,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
):
    if (system, precision) not in model.keys():
        # Below MIN_SAMPLES — no model yet, nothing to predict with.
        return
    median_ids = db.get_conf_ids_with_median_time(system, precision)
    rows = []
    for conf_id, conf in db.iter_confs_by_precision(precision):
        if conf_id in median_ids:
            # Median estimate is kept up to date by add_run.
            continue
        pred = model.predict(system, conf, conf.batch, conf.length)
        rows.append((conf_id, system, pred, 'predicted'))
    if rows:
        db.upsert_time_estimates(rows)
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
    runs = list(db.get_runs_for_timing(system, precision))
    if not runs:
        return
    model.fit(runs, verbose=False)
    _refresh_estimates(db, model, system, precision)


def bootstrap(db: ResultDB):
    """Bring the DB into a consistent state before starting the search.

    1. Backfill medians: recompute median_score and median_time (-> the
       'median' rows of conf_time_estimate) for every (conf, system) pair
       with runs. This matters after a schema migration — new estimate
       rows are written per-run by `add_run`, but pre-existing data only
       shows up here.
    2. Fit the timing model for every (system, precision) pair with
       enough runs.
    3. Write 'predicted' estimates for every conf without a median.
    """
    logging.info("Bootstrap: backfilling medians")
    db.update_all_scores()

    logging.info("Bootstrap: fitting timing models")
    model = TrainTimingModel()
    for system in db.get_systems():
        for precision in Precision:
            _refit_pair(db, model, system, precision)


class TimingThread(threading.Thread):
    def __init__(self, db_path: str | None, queue: Queue):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        self._model = TrainTimingModel()

    def run(self):
        db = ResultDB(self._db_path)
        logging.info("Started timing thread")
        try:
            while True:
                command, args = self._queue.get()
                if command == "refit":
                    system, precision = args
                    _refit_pair(db, self._model, system, precision)
                elif command == "stop":
                    logging.info("Stopping timing thread")
                    break
                else:
                    assert False, f"Unknown timing command: {command}"
        finally:
            db.close()
