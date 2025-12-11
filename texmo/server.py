import argparse
import io
import logging
import os
import threading
from queue import Queue
import base64

import matplotlib
import matplotlib.pyplot as plt
from flask import (Flask, make_response, redirect, render_template, request,
                   send_from_directory)

from .common import ttoa3
from .configuration2 import (Bounds, Configuration2, Optimizer, Precision,
                             Template, default_from_template)
from .latency import get_report, timer
from .report import generate_report_by_weight
from .resultdb import ResultDB
from .run import Run
from .search2 import Search
from .tokens import set_tokens_dir

matplotlib.use('Agg')


def build_graph(confs: list[Configuration2]) -> bytes:
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
            xs.append(conf_score.conf.model.weights - 0.1)
            ys.append(prev_y)

        xs.append(conf_score.conf.model.weights)
        ys.append(conf_score.median_score)

        prev_x = conf_score.conf.model.weights
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


class SearchThread(threading.Thread):
    def __init__(
        self,
        db: ResultDB,
        template: Template,
        train_time: tuple[float, float],
        default: Configuration2,
        requests_queue: Queue,
        confs_by_system: dict,
        confs_by_system_lock: threading.Lock,
        report_queue: Queue,
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
        self.report_queue = report_queue
        _, self.max_time = train_time

    def run(self):
        logging.info("Started search thread")
        while True:
            command, args = self.requests_queue.get()
            if command == "select":
                system = args
                if system is None:
                    break
                conf, soft_tl = self.search.select_conf(system)
                with self.confs_by_system_lock:
                    self.confs_by_system[system].put((conf, soft_tl))
            elif command == "add":
                conf, run = args
                self.search.add_run(conf, run)
            elif command == "report":
                system = args
                report = generate_report_by_weight(self.db, self.template, system, max_time=self.max_time)
                self.report_queue.put((system, report))
            elif command == "stop":
                logging.info("Stopping search thread")
                break
            else:
                assert False, f"Unknown command: {command}"


def _render_bounds(b: Bounds):
    if b.min == b.max:
        return str(b.min)
    else:
        return f'{b.min}-{b.max}'


def _get_flags(enum, params: dict) -> list:
    vals = []
    for val in enum:
        if params.get(str(val)):
            vals.append(val)
    return vals


class SearchServer(object):
    def __init__(self, db: ResultDB, template: Template, train_time: tuple[float, float], default_spec: str):
        self.db: ResultDB = db
        self.template: Template = template
        self.train_time: tuple[float, float] = train_time
        self.default = default_from_template(template, spec=default_spec)
        logging.info(f"Default configuration: {self.default}")

        self.requests_queue = Queue()
        self.confs_by_system: dict[str, Queue] = {}
        self.confs_by_system_lock = threading.Lock()
        self.report_queue = Queue()

        self.search_thread = SearchThread(
            db,
            template,
            train_time=train_time,
            default=self.default,
            requests_queue=self.requests_queue,
            confs_by_system=self.confs_by_system,
            confs_by_system_lock=self.confs_by_system_lock,
            report_queue=self.report_queue,
        )
        self.search_thread.start()

    def __del__(self):
        self.requests_queue.put(("stop", None))

    def index(self):
        pattern = self.template.regex.pattern if self.template.regex else ""

        precision = {}
        for p in Precision:
            precision[p] = p in self.template.precision

        optimizer = {}
        for o in Optimizer:
            optimizer[o] = o in self.template.optimizer

        top_confs = list(self.db.top_confs_global(self.template))

        graph = build_graph(top_confs)

        top = []
        for conf_score in top_confs:
            top.append({
                'spec': str(conf_score.conf.model),
                'weights': conf_score.conf.model.weights,
                'precision': str(conf_score.conf.precision),
                'length': conf_score.conf.length,
                'batch': conf_score.conf.batch,
                'learning': conf_score.conf.learning_str,
                'steps': conf_score.conf.steps,
                'score': f'{conf_score.median_score:.3f} ({conf_score.num_runs})',
                'time': f'{ttoa3(conf_score.median_time)} on {conf_score.system}',
            })

        tmin, tmax = self.train_time
        train_time = f'{tmin}-{tmax}'

        return render_template(
            "index.html",
            spec=pattern,
            weights=_render_bounds(self.template.max_weights),
            length=_render_bounds(self.template.length),
            batch=_render_bounds(self.template.batch),
            precision=precision,
            optimizer=optimizer,
            lr=_render_bounds(self.template.lr),
            decay=_render_bounds(self.template.decay),
            steps=_render_bounds(self.template.steps),
            time=train_time,
            top=top,
            graph=base64.b64encode(graph).decode('ascii')
        )

    def update(self, params):
        self.template.update_regex(params["spec"])
        self.template.update_weights(params['weights'])
        self.template.update_length(params['length'])
        self.template.update_batch(params['batch'])
        self.template.update_precision(_get_flags(Precision, params))
        self.template.update_optimizer(_get_flags(Optimizer, params))
        self.template.update_lr(params['lr'])
        self.template.update_decay(params['decay'])
        self.template.update_steps(params['steps'])
        return redirect("/")

    def select(self, args):
        system = args["system"]

        with self.confs_by_system_lock:
            if system not in self.confs_by_system:
                self.confs_by_system[system] = Queue()
                self.requests_queue.put(("select", system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(("select", system))

        conf, soft_tl = response_queue.get()
        logging.info(f"Sending conf for system {system}: {conf}")
        result = {
            "system": system,
            "conf": conf.to_dict(),
            "soft_tl": soft_tl
        }
        return result

    def add_run(self, params):
        run = Run.from_dict(params["run"])
        if run.loss is None:
            logging.info('run.loss is None!')
            return
        conf = Configuration2.from_dict(params["conf"])
        logging.info(f"Adding run: {conf} - {run}")
        self.requests_queue.put(("add", (conf, run)))

    def report(self, args):
        system = args["system"]
        logging.info(f"Generating report for system {system}")

        self.requests_queue.put(("report", system))
        sys, report = self.report_queue.get()
        assert sys == system

        return report

    def join(self):
        self.requests_queue.put(("stop", None))
        self.search_thread.join()
        logging.info("Search thread joined")


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    logging.info(f"Template: {template}")

    db = ResultDB.from_args(args.db)
    train_time = tuple(map(float, args.train_time.split("-")))
    logging.info(f"T ∈ {train_time} s")
    server = SearchServer(db, template, train_time, args.default_spec)

    app = Flask("texmo")

    @app.route("/", methods=["GET"])
    def _index():
        return server.index()

    @app.route("/update", methods=["POST"])
    def _update():
        with timer("SearchServer.update"):
            return server.update(request.form)

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

    @app.route("/report", methods=["GET"])
    def _report():
        with timer("SearchServer.report"):
            response = make_response(server.report(request.args), 200)
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
        app.run(host="0.0.0.0", debug=True)
    finally:
        server.join()


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
        "--spec-regex",
        type=str,
        default=None,
        help="regex covering the acceptable specs",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp64,fp32,fp16,bf16",
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
        "--optimizer",
        type=str,
        metavar="O",
        default="adam,fromage",
        help="the optimizer algorithm",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates",
    )
    parser.add_argument(
        '--decay',
        type=str,
        default="0-1",
        help="decay of the learning rate over the course of training, i.e. (LR at the last step) / (LR at the first step). Incompatible with saving intermediate results. (default: 1)",
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

    parser.set_defaults(func=main)
