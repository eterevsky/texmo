import argparse
from flask import Flask, render_template, request, redirect
import logging

from .configuration2 import Configuration2, Template, default_from_template
from .resultdb import ResultDB
from .search import Search
from .tokens import set_tokens_dir


class SearchServer(object):
    def __init__(self, db: ResultDB, template: Template, default_spec: str):
        self.db: ResultDB = db
        self.template: Template = template
        self.default = default_from_template(template, spec=default_spec)
        logging.info(f"Default configuration: {self.default}")
        self.searches = {}

    def index(self):
        pattern = self.template.regex.pattern if self.template.regex else ""
        return render_template("index.html", spec=pattern)

    def update(self, params):
        self.template.update_regex(params["spec"])
        return redirect("/")

    def select(self, args):
        system = args["system"]

        search = self.searches.get(system)
        if search is None:
            search = Search(
                system,
                self.db,
                self.template,
                self.default,
                predictor=None,
                train_time=(1.0, 16.0),
            )
            self.searches[system] = search

        result = search.select_conf().to_dict()
        result["system"] = system
        return result


def main(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)
    template = Template.from_args(args)
    logging.info(f"Template: {template}")

    db = ResultDB.from_args(args.db)
    server = SearchServer(db, template, args.default_spec)

    app = Flask("texmo")

    @app.route("/", methods=["GET"])
    def _index():
        return server.index()

    @app.route("/update", methods=["POST"])
    def _update():
        return server.update(request.form)

    @app.route("/select", methods=["GET"])
    def _select():
        return server.select(request.args)

    app.run(debug=True)


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
