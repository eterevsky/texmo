import base64
import hmac
import io
import logging
import os
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from flask import (
    Flask,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.serving import make_server

from .common import INF, ttoa3
from .configuration import (
    Bounds,
    Configuration,
    DecayType,
    Precision,
    Template,
    default_from_template,
    format_lr,
)
from .db import ConfScore, DbReader, DbWriter
from .db.writer import DbWriterProxy, Stop as WriterStop, WriterThread
from .latency import get_report, report, timer
from .predict.loss_rnn import LossModelHolder
from .predict.persist import load_predictor
from .predict.model_thread import (
    BootstrapTiming,
    LossRefit,
    ModelThread,
    RunAdded,
    Stop as ModelStop,
)
from .predict.timing import TrainTimingModel
from .report import (
    build_fastest_loss_graph,
    build_fastest_time_graph,
    build_throughput_graph,
    fastest_near_best,
    per_system_throughput,
)
from .run import Run
from .search import (
    Select,
    SearchThread,
    SetTemplate,
    Stop as SearchStop,
)


# Paths reachable on the external (auth-required) port. Anything else
# 404s — in particular the index page, which is heavy to render and a
# DDoS target.
_EXTERNAL_PATHS = frozenset({'/select', '/add'})

# Internal port (LAN-trusted, no auth required).
_INTERNAL_PORT = 5000
# External port (requires Bearer auth).
_EXTERNAL_PORT = 5001

# How often (seconds) to append a latency-counter snapshot to the dump
# file, so we have a time series of where wall-clock goes (and a record
# that survives a hung shutdown).
_LATENCY_DUMP_INTERVAL = 120


def build_graph(confs: list[Configuration]) -> bytes:
    plt.ioff()
    plt.clf()

    _fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    xs = []
    ys = []

    prev_x = None
    prev_y = None

    for conf_score in confs:
        if prev_y is not None:
            xs.append(conf_score.conf.model.num_weights - 0.1)
            ys.append(prev_y)

        xs.append(conf_score.conf.model.num_weights)
        ys.append(conf_score.median_score)

        prev_x = conf_score.conf.model.num_weights
        prev_y = conf_score.median_score

    if prev_x is not None:
        xs.append(prev_x * 2)
        ys.append(prev_y)

    plt.plot(xs, ys)
    plt.xlabel('weights')
    plt.ylabel('enthropy, b/B')
    f = io.BytesIO()
    plt.savefig(f, format='png')
    return f.getvalue()


def _envelope(confs) -> list[tuple[int, float]]:
    """Loss envelope for a list of ConfScore: a list of (weights, score)
    points with strictly decreasing score in weights ascending. Same as
    what `top_confs_global` already yields, but explicitly typed."""
    pts = sorted(
        ((c.conf.num_weights, c.median_score) for c in confs),
        key=lambda p: p[0],
    )
    out: list[tuple[int, float]] = []
    best = float('inf')
    for w, s in pts:
        if s < best:
            best = s
            out.append((w, best))
    return out


def _eval_envelope(env: list[tuple[int, float]], w: int) -> Optional[float]:
    """Step-function value at `w`: the most recent envelope score for
    a Pareto point with weights <= w. None if `w` precedes the envelope."""
    last: Optional[float] = None
    for w_i, s_i in env:
        if w_i > w:
            break
        last = s_i
    return last


def build_diff_graph(
    env1: list[tuple[int, float]],
    env2: list[tuple[int, float]],
    max_weights: int,
) -> bytes:
    """Step plot of (L2 - L1) / min(L1, L2) * 100 over the union of
    weight points in both envelopes. Positive means template 1 is
    better at that weight count."""
    plt.ioff()
    plt.clf()
    _fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # x-grid is the union of weight points from both envelopes that are
    # within the budget; both envelopes must have a value at x for the
    # diff to be defined, so we start at max(min_w1, min_w2).
    ws = sorted({w for w, _ in env1} | {w for w, _ in env2})
    ws = [w for w in ws if w <= max_weights]
    xs: list[float] = []
    ys: list[float] = []
    prev_y: Optional[float] = None
    for w in ws:
        l1 = _eval_envelope(env1, w)
        l2 = _eval_envelope(env2, w)
        if l1 is None or l2 is None:
            continue
        y = (l2 - l1) / min(l1, l2) * 100.0
        if prev_y is not None:
            xs.append(w - 0.1)
            ys.append(prev_y)
        xs.append(w)
        ys.append(y)
        prev_y = y
    if prev_y is not None:
        xs.append(max_weights)
        ys.append(prev_y)

    ax.axhline(0.0, color='gray', linewidth=0.8)
    if xs:
        ax.plot(xs, ys)
    ax.set_xlabel('weights')
    ax.set_ylabel('(L2 - L1) / min(L1, L2), %  '
                  '(positive = T1 better)')

    f = io.BytesIO()
    plt.savefig(f, format='png')
    return f.getvalue()


def _maybe_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _strip_prefix(form: dict, prefix: str) -> dict:
    """Return a sub-dict with `prefix_` stripped from matching keys."""
    head = prefix + '_'
    return {k[len(head):]: v for k, v in form.items() if k.startswith(head)}


def _template_from_compare_form(form: dict, prefix: str) -> Template:
    """Build a Template from prefixed compare-form fields, plumbing the
    shared `weights` value through unchanged."""
    sub = _strip_prefix(form, prefix)
    sub['weights'] = form.get('weights', '')
    return Template.from_form(sub)


def _compare_defaults(live: Template) -> dict:
    """Initial values for the compare form. Both templates start as an
    unfiltered match; weights inherits the live template's upper bound
    so the comparison covers the same range the search is sweeping."""
    weights_default = (
        '' if live.max_weights.max == INF else str(int(live.max_weights.max))
    )
    out = {'weights': weights_default}
    for prefix in ('t1', 't2'):
        out[f'{prefix}_spec'] = ''
        out[f'{prefix}_lr'] = ''
        out[f'{prefix}_length'] = ''
        out[f'{prefix}_batch'] = ''
        out[f'{prefix}_steps'] = ''
        out[f'{prefix}_time'] = ''
        for p in (Precision.FP32, Precision.FP16, Precision.BF16):
            out[f'{prefix}_{p}'] = True
        out[f'{prefix}_{Precision.FP64}'] = False
        for d in DecayType:
            out[f'{prefix}_decay_{d}'] = True
    return out


def _conf_row(conf_score) -> dict:
    """Build a template row dict for a ConfScore."""
    conf = conf_score.conf
    # --decay and --cosine are mutually exclusive; only emit the
    # flag that's actually in use.
    schedule_flag = (
        ' --cosine' if conf.cosine
        else f' --decay {format_lr(conf.decay)}'
    )
    cmd = (
        f'uv run texmo.py train'
        f" -s '{conf.model}'"
        f' -p {conf.precision}'
        f' -b {conf.batch}'
        f' --lr {format_lr(conf.lr)}'
        f'{schedule_flag}'
        f' -l {conf.length}'
        f' --steps {conf.steps}'
    )
    return {
        'spec': str(conf.model),
        'weights': conf.model.num_weights,
        'precision': str(conf.precision),
        'data': f'{conf.batch}×{conf.length}',
        'lr': conf.learning_str,
        'steps': conf.steps,
        'score': f'{conf_score.median_score:.3f} ({conf_score.num_runs})',
        'time': f'{ttoa3(conf_score.median_time)} on {conf_score.system}',
        'cmd': cmd,
    }


def _render_bounds(b: Bounds):
    if b.min == b.max:
        return str(b.min)
    else:
        return f'{b.min}-{b.max}'


def _probe_timing(model: TrainTimingModel, conf: Configuration) -> None:
    """Exercise a loaded timing model's feature path on one conf, to
    surface a feature-schema mismatch. predict_batch returns early
    (without featurizing) unless a fitted pair matches the conf's
    precision, so probe against such a pair; if none exists there's
    nothing to exercise."""
    for system, precision in model.keys():
        if precision == conf.precision:
            model.predict_batch(system, [conf])
            return


def _load_predictor(reader, name: str, probe):
    """Load the persisted predictor file for `name` (stored next to
    the DB; see predict/persist.py), returning it only if it both
    deserializes and survives `probe` (a trivial use).

    A feature-schema change (e.g. the loss model's new split slot) makes
    an old model unpickle fine but raise a shape mismatch on the first
    predict -- so we don't just guard deserialization, we run it once.
    On any failure we log and return None, leaving the caller to refit.
    """
    try:
        model = load_predictor(reader.path, name)
        if model is None:
            return None
        probe(model)
        return model
    except Exception as e:
        logging.warning(
            f"Ignoring persisted {name!r} model "
            f"(incompatible, will refit): {e!r}")
        return None


class SearchServer(object):
    def __init__(
        self,
        path: str,
        template: Template,
        train_time: tuple[float, float],
        default_spec: str,
    ):
        # `path` is also the read-only handle factory for per-request
        # reads in `index` / `compare` / `throughput`; opening a fresh
        # `DbReader` per request avoids contending with the writer.
        self.path: str = path
        self.template: Template = template
        self.train_time: tuple[float, float] = train_time
        # Stored for `update()` so a template change with no literal
        # spec can still fall back to the CLI default.
        self._default_spec = default_spec
        self.default = default_from_template(template, spec=default_spec)
        logging.info(f"Default configuration: {self.default}")

        self.requests_queue = Queue()
        self.confs_by_system: dict[str, Queue] = {}
        self.confs_by_system_lock = threading.Lock()

        # Single shared instances. The Model thread mutates them
        # (per-(system, precision) writes on the timing model are atomic
        # in CPython; the loss holder swaps its underlying model with a
        # single attribute write); the Search thread only reads.
        self.timing_model = TrainTimingModel()
        self.loss_model = LossModelHolder()

        # Apply the schema if this is a fresh DB so the read below
        # doesn't trip on a not-yet-created file — `open_connection`
        # only runs the schema bootstrap on a read-write open. The
        # temporary writer lives only on this thread.
        with DbWriter(self.path):
            pass

        # Whatever's missing here gets trained async on the Model
        # thread once the queues are wired up below.
        need_timing_bootstrap = True
        need_loss_refit = True
        with DbReader(self.path) as reader:
            # Load persisted predictors, but only adopt them if they
            # still work with the current feature schema -- an old model
            # from before a schema change deserializes fine yet raises on
            # the first predict. _load_predictor probes each one and
            # returns None (-> refit) on any incompatibility.
            loaded_timing = _load_predictor(
                reader, 'timing',
                lambda m: _probe_timing(m, self.default))
            if loaded_timing is not None:
                # Adopt the persisted instance as the shared one.
                # Safe to swap here: no threads have been started yet.
                self.timing_model = loaded_timing
                need_timing_bootstrap = False
                logging.info(
                    "Loaded persisted timing model from DB "
                    f"({len(loaded_timing.keys())} pairs)")
            loaded_loss_model = _load_predictor(
                reader, 'loss',
                lambda m: m.predict([self.default]))
            if loaded_loss_model is not None:
                self.loss_model.set_model(loaded_loss_model)
                need_loss_refit = False
                logging.info(
                    "Loaded persisted loss model from DB "
                    f"(max_layers={loaded_loss_model.max_layers})")

        # Writer thread first: producers below post to its queue and
        # we want a consumer ready before that happens.
        self.write_queue: Queue = Queue()
        self.writer_thread = WriterThread(self.path, self.write_queue)
        self.writer_thread.start()

        # Server-side handle the request handlers and `add_run` use to
        # enqueue writes without touching SearchThread.
        self._writer = DbWriterProxy(self.write_queue)

        self.train_queue: Queue = Queue()
        self.model_thread = ModelThread(
            self.path, self.train_queue,
            write_queue=self.write_queue,
            timing_model=self.timing_model,
            loss_model=self.loss_model,
        )
        self.model_thread.start()
        # Whatever wasn't on disk: train it async. Each request gets a
        # falls-through strategy until the Model thread publishes.
        if need_timing_bootstrap:
            self.train_queue.put(BootstrapTiming())
        if need_loss_refit:
            self.train_queue.put(LossRefit())

        # SearchThread opens its own persistent DbReader inside `run()`
        # (so the reader's connection lives on its own thread). Writes
        # happen on the request handler thread directly into
        # `write_queue` — SearchThread is read-only now.
        self.search_thread = SearchThread(
            path=self.path,
            template=template,
            train_time=train_time,
            default=self.default,
            requests_queue=self.requests_queue,
            confs_by_system=self.confs_by_system,
            confs_by_system_lock=self.confs_by_system_lock,
            timing_model=self.timing_model,
            loss_model=self.loss_model,
        )
        self.search_thread.start()

    def __del__(self):
        self.requests_queue.put(SearchStop())
        self.train_queue.put(ModelStop())
        self.write_queue.put(WriterStop())

    def index(self, selected_system: Optional[str] = None):
        return self._render_index(
            template=self.template,
            default_spec=self._default_spec or '',
            train_time=self.train_time,
            selected_system=selected_system,
        )

    def _render_index(
        self,
        *,
        template: Template,
        default_spec: str,
        train_time: tuple[float, float],
        selected_system: Optional[str] = None,
        error: Optional[str] = None,
    ):
        if template.spec:
            pattern = template.spec
        elif template.regex:
            pattern = template.regex.pattern
        else:
            pattern = ''

        precision = {}
        for p in Precision:
            precision[p] = p in template.precision

        decay_types = {
            t.value: t in template.decay_types for t in DecayType
        }

        tmin, tmax = train_time
        train_time_str = f'{tmin}-{tmax}'

        # Use a dedicated read-only connection to avoid contending with
        # the writer thread for the main connection.
        # `max_time=tmax` limits the top list to confs that at least
        # one system can train within the user's time budget -- confs
        # whose smallest cross-system median_time exceeds tmax aren't
        # actionable from this template.
        with DbReader(self.path) as ro_db:
            systems = ro_db.get_systems()
            top_confs = list(
                ro_db.top_confs_global(
                    template, system=selected_system,
                    max_time=tmax))

        graph = build_graph(top_confs)

        top = [_conf_row(tc) for tc in top_confs]

        return render_template(
            "index.html",
            spec=pattern,
            default_spec=default_spec,
            weights=_render_bounds(template.max_weights),
            num_layers=_render_bounds(template.num_layers),
            length=_render_bounds(template.length),
            batch=_render_bounds(template.batch),
            precision=precision,
            lr=_render_bounds(template.lr),
            decay_types=decay_types,
            steps=_render_bounds(template.steps),
            time=train_time_str,
            top=top,
            graph=base64.b64encode(graph).decode('ascii'),
            systems=systems,
            selected_system=selected_system,
            error=error,
        )

    def update(self, params):
        # The form's default_spec field is authoritative: empty means
        # "no override, let the auto-finder pick". Replaces any prior
        # value (CLI --default-spec on startup, or previous form value).
        new_default_spec = params.get('default_spec', '').strip()
        new_template = None
        new_default = None
        new_train_time = self.train_time
        try:
            new_template = Template.from_form(params)
            spec = new_template.spec or new_default_spec or None
            new_default = default_from_template(new_template, spec=spec)
            time_str = params.get("time", "")
            if time_str:
                tmin, tmax = map(float, time_str.split("-"))
                new_train_time = (tmin, tmax)
        except Exception as exc:
            # Re-render the index with the (partially) submitted form
            # values and an error banner; leave the live template
            # untouched so the search keeps running on the last known-
            # good configuration.
            logging.warning(f"update() rejected form: {exc}")
            return self._render_index(
                template=new_template or self.template,
                default_spec=new_default_spec or (
                    self._default_spec or ''),
                train_time=new_train_time,
                error=str(exc),
            )

        # Mutate the search thread's state via a queue message so the
        # update applies atomically between two select_conf calls
        # rather than racing the loop.
        self.requests_queue.put(SetTemplate(
            template=new_template,
            init_conf=new_default,
            train_time=new_train_time,
        ))
        # SearchServer-side copies are read by Flask handlers (index
        # form rendering, /compare); the search thread's copies may
        # lag by one queue step but converge on the next select.
        self.template = new_template
        self.default = new_default
        self.train_time = new_train_time
        self._default_spec = new_default_spec
        logging.info(f'New default configuration: {new_default}')
        logging.info(f'New template: {new_template}')
        logging.info(f'New train time: {new_train_time}')
        return redirect("/")

    def throughput(self, args):
        """Render the per-system throughput chart for the live template.

        For each weight bucket on the global Pareto front, fixes the
        model spec and finds each system's best (batch, length, lr,
        steps) for that spec. Plots `mults * batch * length * steps /
        median_time` per system.

        `args` carries the `system_<name>` checkboxes; on the first
        visit (no args) every system is selected.
        """
        max_time = self.train_time[1]
        with DbReader(self.path) as ro_db:
            all_systems = ro_db.get_systems()
            if args:
                # Submitted form: unchecked boxes are absent. Treat
                # each system as opt-in by `system_<name>` membership.
                selected = [
                    s for s in all_systems if f'system_{s}' in args
                ]
            else:
                selected = list(all_systems)
            per_system = per_system_throughput(
                ro_db, self.template, max_time, systems=selected)
        graph = base64.b64encode(
            build_throughput_graph(per_system)
        ).decode('ascii')
        selected_set = set(selected)
        return render_template(
            "throughput.html",
            graph=graph,
            max_time=max_time,
            num_systems=len(per_system),
            systems=all_systems,
            selected_systems=selected_set,
        )

    def fastest(self, args):
        """Render the cross-system fastest-near-best report.

        For each weight bucket on the global Pareto front, finds the
        (conf, system) pair that reaches within 1% of the best loss
        in the lowest time across all systems. Renders two graphs
        (loss and time per weight) plus a table.
        """
        with DbReader(self.path) as ro_db:
            segments = fastest_near_best(ro_db, self.template)

        # Deduplicate by weight count for the table — two confs at the
        # same weight count shouldn't both appear; keep the faster one.
        by_weights: dict[int, ConfScore] = {}
        for _, _, cs in segments:
            nw = cs.conf.model.num_weights
            existing = by_weights.get(nw)
            if existing is None or cs.median_time < existing.median_time:
                by_weights[nw] = cs
        confs = [
            _conf_row(cs)
            for cs in sorted(
                by_weights.values(),
                key=lambda c: c.conf.model.num_weights,
            )
        ]

        loss_graph = None
        time_graph = None
        if segments:
            max_w = max(cs.conf.model.num_weights for _, _, cs in segments)
            loss_graph = base64.b64encode(
                build_fastest_loss_graph(segments, max_w)
            ).decode('ascii')
            time_graph = base64.b64encode(
                build_fastest_time_graph(segments, max_w)
            ).decode('ascii')

        return render_template(
            "fastest.html",
            confs=confs,
            loss_graph=loss_graph,
            time_graph=time_graph,
        )

    def compare(self, args):
        """Render the template-comparison page.

        With no params -> just the form (plus reasonable defaults).
        With params -> form + diff curve graph + per-template top
        tables.

        Form layout: shared `weights`, then two namespaced columns
        (`t1_*` / `t2_*`) for spec, lr, length, batch, steps, decay
        types, precision, and per-template `max_time`.
        """
        defaults = _compare_defaults(self.template)
        # If the form was submitted, unchecked checkboxes are absent
        # from `args` — falling back to defaults would re-check them.
        # So only inherit defaults when nothing was submitted.
        submitted = bool(args)

        form: dict = {}
        for key, default in defaults.items():
            if isinstance(default, bool):
                form[key] = bool(args.get(key)) if submitted else default
            else:
                form[key] = args.get(key, default)

        graph = None
        top1: list[dict] = []
        top2: list[dict] = []
        max_weights: Optional[int] = None

        if args:
            try:
                max_weights = int(form['weights'])
            except (TypeError, ValueError):
                max_weights = None
            if max_weights is not None and max_weights > 0:
                t1 = _template_from_compare_form(form, 't1')
                t2 = _template_from_compare_form(form, 't2')
                t1_max_time = _maybe_float(form.get('t1_time'))
                t2_max_time = _maybe_float(form.get('t2_time'))
                with DbReader(self.path) as ro_db:
                    confs1 = list(ro_db.top_confs_global(
                        t1, max_weights=max_weights, max_time=t1_max_time))
                    confs2 = list(ro_db.top_confs_global(
                        t2, max_weights=max_weights, max_time=t2_max_time))
                env1 = _envelope(confs1)
                env2 = _envelope(confs2)
                graph = base64.b64encode(
                    build_diff_graph(env1, env2, max_weights)
                ).decode('ascii')
                top1 = [_conf_row(c) for c in confs1]
                top2 = [_conf_row(c) for c in confs2]

        return render_template(
            "compare.html",
            form=form,
            graph=graph,
            top1=top1,
            top2=top2,
        )

    def select(self, args):
        system = args["system"]

        with self.confs_by_system_lock:
            if system not in self.confs_by_system:
                self.confs_by_system[system] = Queue()
                self.requests_queue.put(Select(system=system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(Select(system=system))

        result = response_queue.get()
        if result is None:
            logging.info(f"No conf for system {system}")
            return {"system": system, "conf": None, "strategy": None}
        logging.info(
            f"Sending conf for system {result.system}: {result.conf} "
            f"(strategy={result.strategy})")
        return result.to_dict()

    def add_run(self, params):
        run = Run.from_dict(params["run"])
        if run.loss is None:
            logging.info('run.loss is None!')
            return
        conf = Configuration.from_dict(params["conf"])
        strategy = params.get("strategy")
        logging.info(f"Adding run: {conf} - {run} (strategy={strategy})")
        # Both queues are posted from this same request-handler thread
        # in order; SQLite WAL guarantees that any subsequent refit on
        # ModelThread reading after `AddRun` commits sees the new row.
        self._writer.add_run(
            conf, run, strategy=strategy, track_winner_change=True)
        self.train_queue.put(
            RunAdded(system=run.system, precision=conf.precision))

    def join(self):
        # Producers first so no more writes / refits land on the
        # writer queue, then the writer drains and exits.
        self.requests_queue.put(SearchStop())
        self.search_thread.join()
        logging.info("Search thread joined")
        self.train_queue.put(ModelStop())
        self.model_thread.join()
        logging.info("Model thread joined")
        self.write_queue.put(WriterStop())
        self.writer_thread.join()
        logging.info("Writer thread joined")

    def _dump_latency_loop(self, path: str):
        """Append a timestamped latency snapshot to `path` every
        `_LATENCY_DUMP_INTERVAL` seconds. Daemon loop, so it never blocks
        shutdown -- and the file survives a hung exit."""
        while True:
            time.sleep(_LATENCY_DUMP_INTERVAL)
            try:
                ts = datetime.now().isoformat(timespec='seconds')
                # Queue state: pending Selects on the (serial) search
                # thread, and per-system ready-but-unfetched responses.
                with self.confs_by_system_lock:
                    per_system = {
                        s: q.qsize() for s, q in self.confs_by_system.items()
                    }
                queues = (
                    f"requests_queue: {self.requests_queue.qsize()}  "
                    f"responses: {per_system}\n"
                )
                # utf-8 explicitly: the report has 'us' rendered as the
                # micro sign, which Windows' default cp1252 can't encode.
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(f"\n===== {ts} =====\n{queues}{get_report()}")
                    f.flush()
            except Exception:
                logging.exception("latency dump failed")

    def serve(self, api_key: str):
        """Run the Flask app on the LAN-internal and external ports.

        Blocks until both werkzeug listeners shut down (either via
        SIGINT or the `/stop` route), then joins the search/model
        threads and prints the latency report.
        """
        # Persist latency counters periodically: /latency shows the live
        # snapshot (and answers even while /select is stalled, since it
        # doesn't touch the SearchThread); this gives the time series and
        # a record that outlives a hung shutdown.
        latency_log = os.path.join(
            os.path.dirname(os.path.abspath(self.path)), 'latency.log')
        logging.info(
            f"Appending latency snapshots to {latency_log} every "
            f"{_LATENCY_DUMP_INTERVAL}s")
        threading.Thread(
            target=self._dump_latency_loop, args=(latency_log,),
            daemon=True).start()

        app = Flask("texmo")

        @app.before_request
        def _gate():
            # We bind two sockets: one on the internal port (LAN, no auth),
            # one on the external port (auth required, restricted path
            # set). The handler can tell which socket received the request
            # from SERVER_PORT.
            port = int(request.environ.get('SERVER_PORT', 0))
            if port != _EXTERNAL_PORT:
                return  # internal: no gating
            # 404 first -- gives less info to unauthenticated probes than
            # leaking which paths exist.
            if request.path not in _EXTERNAL_PATHS:
                return ('', 404)
            if not api_key:
                # External port is enabled but no key configured -- refuse
                # rather than serve unauthenticated externally.
                return ('', 401)
            provided = request.headers.get('Authorization', '')
            if not provided.startswith('Bearer '):
                return ('', 401)
            if not hmac.compare_digest(provided[7:], api_key):
                return ('', 401)

        @app.route("/", methods=["GET"])
        def _index():
            selected_system = request.args.get("system") or None
            return self.index(selected_system=selected_system)

        @app.route("/update", methods=["POST"])
        def _update():
            with timer("SearchServer.update"):
                return self.update(request.form)

        @app.route("/compare", methods=["GET"])
        def _compare():
            with timer("SearchServer.compare"):
                return self.compare(request.args)

        @app.route("/throughput", methods=["GET"])
        def _throughput():
            with timer("SearchServer.throughput"):
                return self.throughput(request.args)

        @app.route("/fastest", methods=["GET"])
        def _fastest():
            with timer("SearchServer.fastest"):
                return self.fastest(request.args)

        @app.route("/select", methods=["GET"])
        def _select():
            with timer("SearchServer.select"):
                return self.select(request.args)

        @app.route("/add", methods=["POST"])
        def _add():
            with timer("SearchServer.add_run"):
                self.add_run(request.json)
                return "", 200

        @app.route("/latency", methods=["GET"])
        def _latency():
            response = make_response(get_report(), 200)
            response.mimetype = "text/plain"
            return response

        @app.route('/favicon.ico')
        def _favicon():
            return send_from_directory(
                os.path.join(app.root_path, 'static'),
                'favicon.ico',
                mimetype='image/vnd.microsoft.icon'
            )

        # Bind two listeners on the same WSGI app:
        #   internal: 0.0.0.0:5000 -- LAN-accessible, no auth.
        #   external: 127.0.0.1:5001 -- bound to loopback so only a
        #     local reverse proxy (terminating TLS) can reach it;
        #     auth required.
        internal_srv = make_server(
            '0.0.0.0', _INTERNAL_PORT, app, threaded=True)
        external_srv = make_server(
            '127.0.0.1', _EXTERNAL_PORT, app, threaded=True)

        @app.route("/stop", methods=["POST"])
        def _stop():
            # `serve_forever` returns when `shutdown()` is called. Doing
            # that from inside a request thread would deadlock, so spawn
            # a one-shot thread that runs after the response is delivered.
            # The external port `before_request` gate already 404s /stop,
            # so this only fires from the internal (LAN-trusted) listener.
            logging.info("Stop requested via web UI")
            def _do_shutdown():
                internal_srv.shutdown()
                external_srv.shutdown()
            threading.Thread(target=_do_shutdown, daemon=True).start()
            return ("Stopping. Search threads will join and the process "
                    "will exit shortly.", 200)
        logging.info(
            f"Serving internal (no auth) on 0.0.0.0:{_INTERNAL_PORT}, "
            f"external (auth) on 127.0.0.1:{_EXTERNAL_PORT}")
        threads = [
            threading.Thread(target=s.serve_forever, daemon=True)
            for s in (internal_srv, external_srv)
        ]
        for t in threads:
            t.start()
        try:
            # Block forever; KeyboardInterrupt falls through to cleanup.
            for t in threads:
                t.join()
        finally:
            internal_srv.shutdown()
            external_srv.shutdown()
            self.join()
            report()

