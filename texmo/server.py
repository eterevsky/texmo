import argparse
from flask import Flask, render_template, request, redirect, make_response
import logging
import threading
from queue import Queue

from flask import Flask, redirect, render_template, request

from .configuration2 import Configuration2, Template, default_from_template
from .latency import get_report, timer
from .resultdb import ResultDB
from .run import Run
from .search2 import Search
from .tokens import set_tokens_dir
from .run import Run
from .report import generate_report_by_weight


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
            elif command == "report":
                system = args
                report = generate_report_by_weight(self.db, self.template, system)
                self.report_queue.put((system, report))
            elif command == "stop":
                logging.info("Stopping search thread")
                break
            else:
                assert False, f"Unknown command: {command}"



class SearchServer(object):
    def __init__(self, db: ResultDB, template: Template, train_time: tuple[float, float], default_spec: str):
        self.db: ResultDB = db
        self.template: Template = template
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
        return render_template("index.html", spec=pattern)

    def update(self, params):
        self.template.update_regex(params["spec"])
        return redirect("/")

    def select(self, args):
        system = args["system"]
        logging.info(f"Generating conf for system {system}")

        with self.confs_by_system_lock:
            if system not in self.confs_by_system:
                self.confs_by_system[system] = Queue()
                self.requests_queue.put(("select", system))
            response_queue = self.confs_by_system[system]

        self.requests_queue.put(("select", system))

        conf = response_queue.get()
        logging.info(f"Generated conf for system {system}: {conf}")
        result = {
            "system": system,
            "conf": conf.to_dict()
        }
        return result

    def add_run(self, params):
        run = Run.from_dict(params["run"])
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
        default="32-4294967296",
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
