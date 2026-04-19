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

    ("bootstrap", None)
        Refit every (system, precision) pair with enough data and
        refresh estimates for all of them.

    ("stop", None)
        Exit the loop.

Median-backed estimates are also maintained by `ResultDB._add_run`, so
the thread can skip over confs with medians without touching their
estimate rows.

For an in-memory DB (path is None or ':memory:') the thread opens a
fresh, empty in-memory DB — writes never become visible to the search
thread. That's intentional and acceptable for tests.
"""

import logging
import threading
from queue import Queue

from .precision import Precision
from .predict.timing import TrainTimingModel
from .resultdb import ResultDB


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
                    self._refit_pair(db, system, precision)
                elif command == "bootstrap":
                    self._bootstrap_all(db)
                elif command == "stop":
                    logging.info("Stopping timing thread")
                    break
                else:
                    assert False, f"Unknown timing command: {command}"
        finally:
            db.close()

    def _refit_pair(self, db: ResultDB, system: str, precision: Precision):
        runs = list(db.get_runs_for_timing(system, precision))
        if not runs:
            return
        self._model.fit(runs, verbose=False)
        self._refresh_estimates(db, system, precision)

    def _bootstrap_all(self, db: ResultDB):
        logging.info("Bootstrapping timing models")
        for system in db.get_systems():
            for precision in Precision:
                self._refit_pair(db, system, precision)

    def _refresh_estimates(
        self, db: ResultDB, system: str, precision: Precision
    ):
        if (system, precision) not in self._model.keys():
            # Below MIN_SAMPLES — no model yet, nothing to predict with.
            return
        median_ids = db.get_conf_ids_with_median_time(system, precision)
        rows = []
        for conf_id, conf in db.iter_confs_by_precision(precision):
            if conf_id in median_ids:
                # Median estimate is kept up to date by add_run.
                continue
            pred = self._model.predict(system, conf, conf.batch, conf.length)
            rows.append((conf_id, system, pred, 'predicted'))
        if rows:
            db.upsert_time_estimates(rows)
        logging.info(
            f"refreshed {len(rows)} predicted estimates "
            f"for ({system!r}, {precision})"
        )
