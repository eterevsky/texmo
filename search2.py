import argparse
import logging

import numpy as np

import config
from texmo import latency
from texmo.common import INF
from texmo.configuration import (Configuration, Template, conf_to_string,
                                 default_from_template)
from texmo.dataset import build_dataset
from texmo.manager import Manager
from texmo.model2 import build_model
from texmo.report import (draw_weight_loss_graph, generate_max_report,
                          generate_param_report, generate_report_by_weight)
from texmo.resultdb import ResultDB
from texmo.results import ResultSet
from texmo.search import Search

# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6


def pick_default_value(default, range):
    if default is not None:
        assert range is None or range[0] <= default <= range[1]
        return default
    assert range is not None
    return range[0]


def warmup(dataset):
    conf = Configuration(
        build_model("suffix.4-rec.32.relu"),
        config.DEFAULT_LR,
        128,
        config.DEFAULT_BATCH,
        8,
    )
    manager = Manager(conf)
    manager.init(quiet=True)
    manager.train_and_eval(
        steps=None,
        time_limit=8,
        train_set=dataset,
        temp_steps=None,
        temp_dir=None,
        output_dir=None,
        log=None,
        quiet=True,
    )


def generate_report(result_set, template, min_max_weights):
    if template.max_weights is None:
        print(generate_report_by_weight(result_set, template, min_max_weights))
        print()
    print(generate_max_report(result_set, template))
    print("\nLearning Rate")
    print(generate_param_report(result_set, template, lambda conf: conf.lr))
    if template.sample_len[0] < template.sample_len[1]:
        print("\nSample Length")
        print(
            generate_param_report(
                result_set,
                template,
                lambda conf: conf.sample_len,
                is_float=False,
            )
        )
    if template.batch is None or template.batch[0] < template.batch[1]:
        print("\nBatch")
        print(
            generate_param_report(
                result_set, template, lambda conf: conf.batch, is_float=False
            )
        )
    draw_weight_loss_graph(result_set, template)


def search_loop(dataset, search: Search):
    logging.info("Warming up training")
    warmup(dataset)

    logging.info("Starting search")
    while True:
        conf = search.select_conf()
        logging.info(conf_to_string(conf))

        manager = Manager(conf)
        manager.init(quiet=True)
        record, run = manager.train_and_eval(
            steps=None,
            time_limit=conf.t,
            train_set=dataset,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=args.log,
            quiet=True,
        )

        search.add_record(record, run)
        print()


def main(args, dataset, template):
    template = Template.from_args(args)
    default = default_from_template(template)
    default = default._replace(model=build_model("dense.1.relu"))
    assert template.match_conf(default)
    logging.info("Default configuration: " + conf_to_string(default))

    print(f"Creating ResultDB from {args.db}")
    result_db = ResultDB(args.db)

    if args.only_report:
        result_set = ResultSet(result_db, template, populate_neighbors=False)
    else:
        search = Search(result_db, template, default, args.min_max_weights)
        result_set = search._result_set
        try:
            search_loop(dataset, search)
        except KeyboardInterrupt:
            print("\nInterrupted\n")
            latency.report()

    generate_report(result_set, template, args.min_max_weights)


def add_template_args(parser: argparse.ArgumentParser):
    """Add command-line arguments describing a template."""
    parser.add_argument(
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex convering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default="1-8192",
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default="0.000001-10",
        help="range of acceptable learning rates (default: 0.001-10)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="128",
        help="range of acceptable sample lens (default: 128)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined (default: unrestricted)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_template_args(parser)

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=config.DATA,
        help="a file with training data",
    )
    parser.add_argument(
        "--log",
        default=config.LOG,
        metavar="LOG",
        help="path to a CSV file for logging",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=config.DB,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "--min-max-weights",
        type=int,
        default=2048,
        help="minimum max-weights value in search",
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="don't run a search, only generate a report",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    np.set_printoptions(linewidth=100, edgeitems=6, precision=3)
    args = parse_args()
    dataset = build_dataset(args.data)
    template = Template.from_args(args)
    main(args, dataset=dataset, template=template)
    dataset.join()
    latency.report()
