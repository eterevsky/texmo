"""Background thread that fits timing & loss models and writes estimates.

Opens its own `DbReader` (separate from SearchThread's) so reads
don't compete with the search loop. Posts every write through a
shared `DbWriterProxy` that lands the request on the `WriterThread`'s
queue. Owns the writer side of the shared `TrainTimingModel` and
`LossModelHolder` references (passed in at construction) and
consumes `TrainMessage`s from its input queue:

    RunAdded(system, precision)
        Notification that a new labeled run has been written to the
        DB. The thread tracks per-(system, precision) and global run
        counts and triggers a timing refit when the corresponding
        threshold is crossed -- and a loss refit too, but only with
        `loss_refit_local` (otherwise the server hands that fit to a
        worker; see server.LOSS_REFIT_MODES).

    LossRefit()
        Force a loss-model refit now (used at server startup when no
        persisted loss model is on disk). Honoured in both refit
        modes -- it predates the distributed flow and is the only way
        a mode-'workers' server gets a first model without waiting
        for a grant.

    Stop()
        Exit the loop.

Median-backed estimates are also maintained by `DbWriter._add_run`, so
the thread can skip over confs with medians without touching their
estimate rows.

A full bootstrap — median backfill + fit all pairs + predictions for
confs without runs — is exposed as a synchronous `bootstrap()` function
so startup can block on it before serving requests. Bootstrap takes a
plain `DbWriter` (called from the main thread before `WriterThread`
starts) rather than a queue.
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue

from .. import latency
from ..configuration import Configuration
from ..db import DbReader, DbWriter
from ..db.writer import DbWriterProxy
from ..precision import Precision
from . import loss_tree
from .loss_rnn import LossModelHolder
from .persist import save_predictor
from .timing import TrainTimingModel


# Threshold of new runs per (system, precision) pair that triggers a
# timing-model refit. At 100 the fit machinery (get_runs + fit alone,
# before any estimate refresh) consumed ~18% of server wall in the
# 2026-07-23 latency report; the model barely moves per 100 runs, so
# refit rarely. A pair with no fitted model yet (a new system) uses
# the faster _FIRST_FIT_EVERY so it gets predictions promptly.
_REFIT_EVERY = 400
_FIRST_FIT_EVERY = 100

# Ordinary refits refresh estimates only for confs that have none yet
# (a few hundred rows, sub-second); every Nth refit per pair re-predicts
# everything, bounding the drift of old predictions against the updated
# model. The full sweep parses + featurizes every conf of the precision
# (~40 s of pure Python at 350k confs, 2026-07-23) and starves the
# GIL, so it must stay rare -- during a backfill it used to run on
# every refit and stalled all request handling for minutes.
_FULL_REFRESH_EVERY = 5

# Every this-many new labeled runs the loss predictor is refit from
# scratch on all labeled runs. The fit blocks this thread for ~2 min
# at 1.2M runs (dedup load + typed-step tree on a GPU platform),
# which is why it stays this rare: timing refits and estimate
# refreshes queue behind it.
#
# The same cadence drives the distributed flow, where the server
# grants one client job per this many runs instead (server.py imports
# this constant; one definition, two triggers). Which trigger is live
# is the `loss_refit_local` flag below.
LOSS_REFIT_EVERY = 500


@dataclass
class RunAdded:
    """A new labeled run was written to the DB. Producer: request
    handlers via SearchThread (`/add` drops runs without a loss);
    consumer: ModelThread (counter bump + maybe refit)."""
    system: str
    precision: Precision


@dataclass
class LossRefit:
    """Force a loss-model refit now (used at startup when there's no
    persisted loss model on disk)."""


@dataclass
class BootstrapTiming:
    """Run the full timing-model bootstrap on the Model thread: median
    backfill, per-(system, precision) refits, predicted estimates for
    every conf without a median. Used at startup when there's no
    persisted timing model on disk."""


@dataclass
class Stop:
    """Stop the model thread."""


TrainMessage = RunAdded | LossRefit | BootstrapTiming | Stop


def _refresh_estimates(
    reader: DbReader,
    writer: DbWriter | DbWriterProxy,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
    full: bool = True,
):
    if full:
        with latency.timer('timing._refresh_estimates.median_ids'):
            median_ids = reader.get_conf_ids_with_median_time(
                system, precision)
        with latency.timer('timing._refresh_estimates.iter_confs'):
            to_predict: list[tuple[int, Configuration]] = [
                (conf_id, conf)
                for conf_id, conf in reader.iter_confs_by_precision(precision)
                if conf_id not in median_ids
            ]
    else:
        # Incremental: only confs with no estimate row for this
        # system yet. Their predictions come from the just-updated
        # model; existing rows keep their (slightly stale)
        # predictions until the next full sweep.
        with latency.timer('timing._refresh_estimates.iter_missing'):
            to_predict = list(
                reader.iter_confs_missing_estimate(system, precision))
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
        f"for ({system!r}, {precision}, {'full' if full else 'incremental'})"
    )


def _refit_pair(
    reader: DbReader,
    writer: DbWriter | DbWriterProxy,
    model: TrainTimingModel,
    system: str,
    precision: Precision,
    full_refresh: bool = True,
):
    with latency.timer('timing._refit_pair.get_runs'):
        runs = list(reader.get_runs_for_timing(
            system, precision, limit=TrainTimingModel.FIT_RUNS_CAP))
    if not runs:
        return
    with latency.timer('timing._refit_pair.fit'):
        model.fit(runs, verbose=False)
    _refresh_estimates(
        reader, writer, model, system, precision, full=full_refresh)


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
    save_predictor(reader.path, 'timing', timing_model.snapshot())


class ModelThread(threading.Thread):
    def __init__(
        self,
        db_path: str | None,
        queue: Queue,
        write_queue: Queue,
        timing_model: TrainTimingModel,
        loss_model: LossModelHolder,
        loss_refit_local: bool = False,
    ):
        super().__init__(daemon=True)
        self._db_path = db_path
        self._queue = queue
        # Posts every write through `WriterThread`; constructed inside
        # `run()` together with the reader so neither connection
        # crosses a thread boundary.
        self._write_queue = write_queue
        # Shared references with the Search threads. Per-(system,
        # precision) weight inserts on `timing_model` are atomic;
        # `loss_model.set_model(...)` is a single pointer swap.
        self._timing_model = timing_model
        self._loss_model = loss_model
        # Per-(system, precision) run counters for timing refits.
        # Private to this thread, so no locking needed.
        self._run_counter: dict[tuple[str, Precision], int] = defaultdict(int)
        # Refit counters driving the incremental/full refresh cadence.
        self._refit_counter: dict[tuple[str, Precision], int] = defaultdict(int)
        # Labeled runs since the last loss-model refit (global, not
        # per pair -- the loss predictor is fit on every labeled run).
        # Only consulted when this thread owns the refit trigger; with
        # `loss_refit_local` false the server grants the fit to a
        # worker instead and nothing here counts down to it.
        self._loss_refit_local = loss_refit_local
        self._loss_run_counter = 0

    def run(self):
        reader = DbReader(self._db_path)
        writer = DbWriterProxy(self._write_queue)
        logging.info("Started model thread")
        try:
            while True:
                m = self._queue.get()
                match m:
                    case RunAdded(system=s, precision=p):
                        self._on_run_added(reader, writer, s, p)
                    case LossRefit():
                        self._refit_loss(reader, writer)
                    case BootstrapTiming():
                        bootstrap(reader, writer, self._timing_model)
                    case Stop():
                        logging.info("Stopping model thread")
                        break
                    case _:
                        assert False, f"Unknown message: {m!r}"
        finally:
            reader.close()

    def _on_run_added(
        self, reader: DbReader, writer: DbWriterProxy,
        system: str, precision: Precision,
    ):
        if self._loss_refit_local:
            self._loss_run_counter += 1
            if self._loss_run_counter >= LOSS_REFIT_EVERY:
                self._loss_run_counter = 0
                self._refit_loss(reader, writer)

        key = (system, precision)
        self._run_counter[key] += 1
        threshold = (
            _REFIT_EVERY
            if self._timing_model.has_weights(system, precision)
            else _FIRST_FIT_EVERY)
        if self._run_counter[key] >= threshold:
            self._run_counter[key] = 0
            self._refit_counter[key] += 1
            full = self._refit_counter[key] >= _FULL_REFRESH_EVERY
            if full:
                self._refit_counter[key] = 0
            _refit_pair(
                reader, writer, self._timing_model, system, precision,
                full_refresh=full)
            save_predictor(
                self._db_path, 'timing', self._timing_model.snapshot())

    def _refit_loss(self, reader: DbReader, writer: DbWriterProxy):
        loss_model = loss_tree.train_loss_model(reader)
        if loss_model is not None:
            # Predictors persist as files next to the DB, not through
            # the writer queue (see predict/persist.py).
            save_predictor(self._db_path, 'loss', loss_model)
            self._loss_model.set_model(loss_model)
            logging.info(
                f"Published new tree loss model "
                f"({len(loss_model.simple_types)} simple types, "
                f"typed_step={loss_model.per_type_fwd})"
            )
