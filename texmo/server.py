import argparse
import base64
import io
import logging
import os
import threading
from queue import Queue
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from flask import (Flask, make_response, redirect, render_template, request,
                   send_from_directory)

from .common import ttoa3
from .configuration import (Bounds, Configuration, Precision,
                             Template, default_from_template)
from .latency import get_report, timer
from .resultdb import ResultDB
from .run import Run
from .search2 import Search
from .tokens import set_tokens_dir

matplotlib.use('Agg')


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

    def run(self):
        logging.info("Started search thread")
        while True:
            command, args = self.requests_queue.get()
            if command == "select":
                system = args
                if system is None:
                    break
                conf = self.search.select_conf(system)
                with self.confs_by_system_lock:
                    self.confs_by_system[system].put(conf)
            elif command == "add":
                conf, run = args
                self.search.add_run(conf, run)
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

        self.search_thread = SearchThread(
            db,
            template,
            train_time=train_time,
            default=self.default,
            requests_queue=self.requests_queue,
            confs_by_system=self.confs_by_system,
            confs_by_system_lock=self.confs_by_system_lock,
        )
        self.search_thread.start()

    def __del__(self):
        self.requests_queue.put(("stop", None))

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

        # Use a dedicated read-only connection to avoid contending with
        # the writer thread for the main connection.
        with self.db.open_readonly() as ro_db:
            systems = ro_db.get_systems()
            top_confs = list(
                ro_db.top_confs_global(
                    self.template, system=selected_system))

        graph = build_graph(top_confs)

        top = []
        for conf_score in top_confs:
            conf = conf_score.conf
            cmd = (
                f'uv run texmo.py train'
                f" -s '{conf.model}'"
                f' -p {conf.precision}'
                f' -b {conf.batch}'
                f' --lr {conf.lr}'
                f' --decay {conf.decay}'
                f' -l {conf.length}'
                f' --steps {conf.steps}'
            )
            top.append({
                'spec': str(conf.model),
                'weights': conf.model.num_weights,
                'precision': str(conf.precision),
                'data': f'{conf.batch}×{conf.length}',
                'lr': conf.learning_str,
                'steps': conf.steps,
                'score': f'{conf_score.median_score:.3f} ({conf_score.num_runs})',
                'time': f'{ttoa3(conf_score.median_time)} on {conf_score.system}',
                'cmd': cmd,
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
            lr=_render_bounds(self.template.lr),
            decay=_render_bounds(self.template.decay),
            steps=_render_bounds(self.template.steps),
            time=train_time,
            top=top,
            graph=base64.b64encode(graph).decode('ascii'),
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

    def select(self, args):
        system = args["system"]

        with self.confs_by_system_lock:
            if system not in self.confs_by_system:
                self.confs_by_system[system] = Queue()
                self.requests_queue.put(("select", system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(("select", system))

        conf = response_queue.get()
        logging.info(f"Sending conf for system {system}: {conf}")
        return {
            "system": system,
            "conf": conf.to_dict(),
        }

    def add_run(self, params):
        run = Run.from_dict(params["run"])
        if run.loss is None:
            logging.info('run.loss is None!')
            return
        conf = Configuration.from_dict(params["conf"])
        logging.info(f"Adding run: {conf} - {run}")
        self.requests_queue.put(("add", (conf, run)))

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
        selected_system = request.args.get("system") or None
        return server.index(selected_system=selected_system)

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
