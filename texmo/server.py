import argparse
import base64
import io
import logging
import os
import threading
from collections import defaultdict
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

from .common import INF, ttoa3
from .configuration import (
    Bounds,
    Configuration,
    DecayType,
    Precision,
    Template,
    default_from_template,
)
from .latency import get_report, timer, report
from .model_training_thread import ModelTrainingThread, bootstrap
from .predict import loss_rnn
from .resultdb import ResultDB
from .run import Run
from .search2 import Search
from .tokens import set_tokens_dir

# Threshold of new runs per (system, precision) pair that triggers a
# timing-model refit. Picked so refits are frequent enough to absorb
# genuinely new points but not so frequent that fit cost dominates.
_REFIT_EVERY = 100

# Threshold of total new runs (any system/precision) that triggers a
# retrain of the loss-prediction model.
_LOSS_REFIT_EVERY = 200


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


def build_scaling_graph(
    segments: list[tuple[int, int, float]], system: str
) -> bytes:
    """Plot weight budget -> fastest time to reach near-best loss.

    `segments` is a list of (w_low, w_high, time) tuples. Each segment
    is flat: for any w in [w_low, w_high), the fastest time is `time`.
    Draws a step function with a single plot() call so matplotlib
    connects adjacent segments as one line.
    """
    plt.ioff()
    plt.clf()
    _fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    y_ticks = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 60, 180, 600]
    y_labels = [
        '3 ms', '10 ms', '30 ms', '100 ms', '300 ms', '1 s', '3 s', '10 s', '30 s', '1 m', '3 m', '10 m']
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_yticks([], minor=True)

    xs: list[float] = []
    ys: list[float] = []
    for w_low, w_high, t in sorted(segments, key=lambda s: s[0]):
        xs.append(w_low)
        ys.append(t)
        xs.append(w_high - 0.1)
        ys.append(t)
    ax.plot(xs, ys)

    ax.set_xlabel('weights')
    ax.set_ylabel(f'fastest time to near-best loss on {system}')
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
    cosine_flag = ' --cosine' if conf.cosine else ''
    cmd = (
        f'uv run texmo.py train'
        f" -s '{conf.model}'"
        f' -p {conf.precision}'
        f' -b {conf.batch}'
        f' --lr {conf.lr}'
        f' --decay {conf.decay}'
        f'{cosine_flag}'
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


class SearchThread(threading.Thread):
    def __init__(
        self,
        db: ResultDB,
        template: Template,
        train_time: tuple[float, float],
        default: Configuration,
        requests_queue: Queue,
        confs_by_system: dict,
        confs_by_system_lock: threading.Lock,
        timing_queue: Optional[Queue] = None,
    ):
        super().__init__()
        self.db = db
        self.template = template
        self.search = Search(
            db=db,
            template=template,
            init_conf=default,
            train_time=train_time,
        )
        self.requests_queue = requests_queue
        self.confs_by_system = confs_by_system
        self.confs_by_system_lock = confs_by_system_lock
        _, self.max_time = train_time
        self._timing_queue = timing_queue
        # Per-(system, precision) counter used to throttle timing-model refits.
        self._run_counter: dict[tuple[str, Precision], int] = defaultdict(int)
        # Total-run counter for loss-model refits.
        self._total_run_counter = 0

    def run(self):
        logging.info("Started search thread")
        while True:
            command, args = self.requests_queue.get()
            if command == "select":
                system = args
                if system is None:
                    break
                result = self.search.select_conf(system)
                with self.confs_by_system_lock:
                    self.confs_by_system[system].put(result)
            elif command == "add":
                conf, run, strategy = args
                self.search.add_run(conf, run, strategy=strategy)
                self._maybe_trigger_time_refit(run.system, conf.precision)
                self._maybe_trigger_loss_refit()
            elif command == "loss_weights":
                self.search.loss_model = args
                logging.info(
                    f"Search thread: received new loss model "
                    f"(max_layers={args.max_layers}, "
                    f"{len(args.simple_types)} simple types)"
                )
            elif command == "timing_weights":
                self.search.timing_model = args
            elif command == "stop":
                logging.info("Stopping search thread")
                break
            else:
                assert False, f"Unknown command: {command}"

    def _maybe_trigger_time_refit(self, system: str, precision: Precision):
        if self._timing_queue is None:
            return
        key = (system, precision)
        self._run_counter[key] += 1
        if self._run_counter[key] >= _REFIT_EVERY:
            self._run_counter[key] = 0
            self._timing_queue.put(("refit", (system, precision)))

    def _maybe_trigger_loss_refit(self):
        if self._timing_queue is None:
            return
        self._total_run_counter += 1
        if self._total_run_counter >= _LOSS_REFIT_EVERY:
            self._total_run_counter = 0
            self._timing_queue.put(("loss_refit", None))


def _render_bounds(b: Bounds):
    if b.min == b.max:
        return str(b.min)
    else:
        return f'{b.min}-{b.max}'


class SearchServer(object):
    def __init__(
        self,
        db: ResultDB,
        template: Template,
        train_time: tuple[float, float],
        default_spec: str,
        bootstrap_models: bool = False,
    ):
        self.db: ResultDB = db
        self.template: Template = template
        self.train_time: tuple[float, float] = train_time
        self.default = default_from_template(template, spec=default_spec)
        logging.info(f"Default configuration: {self.default}")

        self.requests_queue = Queue()
        self.confs_by_system: dict[str, Queue] = {}
        self.confs_by_system_lock = threading.Lock()

        # Load any persisted models from the DB. With --bootstrap-models
        # we retrain from scratch (overwriting what's persisted).
        # Otherwise we use whatever was saved; if nothing is there we
        # fall through to the async refit at the end of init.
        initial_timing_model = None
        initial_loss_model = None
        if bootstrap_models:
            logging.info("Bootstrap: fitting timing model synchronously")
            initial_timing_model = bootstrap(db)
            logging.info("Bootstrap: training loss model synchronously")
            initial_loss_model = loss_rnn.train_loss_model(db)
            if initial_loss_model is not None:
                db.save_model('loss', initial_loss_model)
        else:
            initial_timing_model = db.load_model('timing')
            if initial_timing_model is not None:
                logging.info(
                    "Loaded persisted timing model from DB "
                    f"({len(initial_timing_model.keys())} pairs)")
            initial_loss_model = db.load_model('loss')
            if initial_loss_model is not None:
                logging.info(
                    "Loaded persisted loss model from DB "
                    f"(max_layers={initial_loss_model.max_layers})")

        self.timing_queue: Queue = Queue()
        self.timing_thread = ModelTrainingThread(
            db.path, self.timing_queue,
            search_queue=self.requests_queue,
        )
        self.timing_thread.start()
        if initial_loss_model is None and not bootstrap_models:
            # No persisted loss model — train one async against
            # whatever data is in the DB. Before this completes the
            # predicted-best strategy falls through.
            self.timing_queue.put(("loss_refit", None))

        self.search_thread = SearchThread(
            db,
            template,
            train_time=train_time,
            default=self.default,
            requests_queue=self.requests_queue,
            confs_by_system=self.confs_by_system,
            confs_by_system_lock=self.confs_by_system_lock,
            timing_queue=self.timing_queue,
        )
        # Install pre-fitted models before the thread starts processing
        # select requests so the predicted-best strategy is usable from
        # the first call.
        if initial_timing_model is not None:
            self.search_thread.search.timing_model = initial_timing_model
        if initial_loss_model is not None:
            self.search_thread.search.loss_model = initial_loss_model
        self.search_thread.start()

    def __del__(self):
        self.requests_queue.put(("stop", None))
        self.timing_queue.put(("stop", None))

    def index(self, selected_system: Optional[str] = None):
        if self.template.spec:
            pattern = self.template.spec
        elif self.template.regex:
            pattern = self.template.regex.pattern
        else:
            pattern = ''

        precision = {}
        for p in Precision:
            precision[p] = p in self.template.precision

        decay_types = {
            t.value: t in self.template.decay_types for t in DecayType
        }

        # Use a dedicated read-only connection to avoid contending with
        # the writer thread for the main connection.
        scaling_graph = None
        fastest = []
        with self.db.open_readonly() as ro_db:
            systems = ro_db.get_systems()
            top_confs = list(
                ro_db.top_confs_global(
                    self.template, system=selected_system))

            # Compute-scaling analysis: only makes sense for a specific
            # system because training times vary wildly across hardware.
            if selected_system is not None and top_confs:
                segments = ro_db.fastest_near_best_segments(
                    self.template,
                    system=selected_system,
                    pareto=top_confs,
                )
                # Deduplicate by number of weights: two configs with the
                # same weight count shouldn't both appear — keep the one
                # with the lower median_time.
                by_weights = {}
                for w_low, w_high, cs in segments:
                    nw = cs.conf.model.num_weights
                    existing = by_weights.get(nw)
                    if (existing is None
                            or cs.median_time < existing.median_time):
                        by_weights[nw] = cs
                for cs in sorted(
                    by_weights.values(),
                    key=lambda c: c.conf.model.num_weights,
                ):
                    fastest.append(_conf_row(cs))
                # The last segment has w_high = None (extends to +infty).
                # Cap it at 2x the max Pareto weight for plotting.
                max_pareto_w = max(
                    cs.conf.model.num_weights for cs in top_confs)
                plot_segments = []
                for w_low, w_high, cs in segments:
                    if cs.median_time is None:
                        continue
                    if w_high is None:
                        w_high = max_pareto_w * 2
                    plot_segments.append((w_low, w_high, cs.median_time))
                if plot_segments:
                    scaling_graph = base64.b64encode(
                        build_scaling_graph(plot_segments, selected_system)
                    ).decode('ascii')

        graph = build_graph(top_confs)

        top = [_conf_row(tc) for tc in top_confs]

        tmin, tmax = self.train_time
        train_time = f'{tmin}-{tmax}'

        return render_template(
            "index.html",
            spec=pattern,
            weights=_render_bounds(self.template.max_weights),
            length=_render_bounds(self.template.length),
            batch=_render_bounds(self.template.batch),
            precision=precision,
            lr=_render_bounds(self.template.lr),
            decay_types=decay_types,
            steps=_render_bounds(self.template.steps),
            time=train_time,
            top=top,
            graph=base64.b64encode(graph).decode('ascii'),
            fastest=fastest,
            scaling_graph=scaling_graph,
            systems=systems,
            selected_system=selected_system,
        )

    def update(self, params):
        self.template = Template.from_form(params)
        self.search_thread.search.template = self.template
        time_str = params.get("time", "")
        if time_str:
            tmin, tmax = map(float, time_str.split("-"))
            self.train_time = (tmin, tmax)
            self.search_thread.search.train_time = (tmin, tmax)
            self.search_thread.max_time = tmax
        logging.info(f'New template: {self.template}')
        logging.info(f'New train time: {self.train_time}')
        return redirect("/")

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
                with self.db.open_readonly() as ro_db:
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
                self.requests_queue.put(("select", system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(("select", system))

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
        self.requests_queue.put(("add", (conf, run, strategy)))

    def join(self):
        self.requests_queue.put(("stop", None))
        self.search_thread.join()
        logging.info("Search thread joined")
        self.timing_queue.put(("stop", None))
        self.timing_thread.join()
        logging.info("Timing thread joined")


def main(args: argparse.Namespace):
    # Server renders graphs to bytes for the web UI, never to a display.
    matplotlib.use('Agg')
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    logging.info(f"Template: {template}")

    db = ResultDB.from_args(args.db)
    train_time = tuple(map(float, args.train_time.split("-")))
    logging.info(f"T ∈ {train_time} s")
    server = SearchServer(
        db, template, train_time, args.default_spec,
        bootstrap_models=args.bootstrap_models,
    )

    app = Flask("texmo")

    @app.route("/", methods=["GET"])
    def _index():
        selected_system = request.args.get("system") or None
        return server.index(selected_system=selected_system)

    @app.route("/update", methods=["POST"])
    def _update():
        with timer("SearchServer.update"):
            return server.update(request.form)

    @app.route("/compare", methods=["GET"])
    def _compare():
        with timer("SearchServer.compare"):
            return server.compare(request.args)

    @app.route("/select", methods=["GET"])
    def _select():
        with timer("SearchServer.select"):
            return server.select(request.args)

    @app.route("/add", methods=["POST"])
    def _add():
        with timer("SearchServer.add_run"):
            server.add_run(request.json)
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

    try:
        # use_reloader=False: the Flask auto-reloader spawns a second
        # process that re-runs main() — which would re-bootstrap the
        # models (minutes of wasted work) and create a second pair of
        # search/timing threads.
        app.run(host="0.0.0.0", debug=True, use_reloader=False)
    finally:
        server.join()
        report()


def init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results, or a URL for a PostgreSQL database",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )

    # Template args
    parser.add_argument(
        "-s",
        "--spec",
        type=str,
        default=None,
        help="regex covering the acceptable specs or an exact spec",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp32,fp16,bf16",
        help="precision"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256"',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates",
    )
    parser.add_argument(
        '--decay-types',
        type=str,
        default="none,exp,cosine",
        help="comma-separated subset of LR-schedule types: "
             "none, exp, cosine (default: all)",
    )
    parser.add_argument(
        "--length",
        type=str,
        default=None,
        help="range of acceptable training sample lengths",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="range for the number of training steps; lower bound >= 2",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="1024-4294967296",
        help="range for the _maximal_ number of weights in the model",
    )
    parser.add_argument(
        "-t",
        "--train-time",
        default="1-16",
        help="range for the training time in seconds",
    )
    parser.add_argument("--default-spec", type=str, default=None, help="default model")
    parser.add_argument(
        "--bootstrap-models",
        action="store_true",
        default=False,
        help=(
            "Before accepting requests, synchronously fit the timing "
            "model (same as `db-bootstrap-estimates`) and train the "
            "loss-prediction RNN. Lets the predicted-best strategy run "
            "immediately. Takes minutes on a populated DB."
        ),
    )

    parser.set_defaults(func=main)
