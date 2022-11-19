import argparse
import logging

from texmo.configuration import (
    Configuration,
    Template,
    add_template_args,
    parse_conf,
    default_from_template,
    conf_to_string,
)
from texmo.results import ResultSet
from texmo.resultdb import ResultDB
from texmo.dataset import build_dataset
from texmo import latency
from texmo.manager import Manager
from texmo.model2 import build_model
from texmo.search import Search
from texmo.common import INF
from texmo.report import (
    draw_weight_loss_graph,
    generate_report_by_weight,
    generate_max_report,
    generate_param_report,
)


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
        None, build_model("suffix.4-rec.32.relu"), 0.25, 128, 256, 0.125, 1.0, 8
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
    if (
        template.regularization is None
        or template.regularization[0] < template.regularization[1]
    ):
        print("\nRegularization")
        print(
            generate_param_report(
                result_set, template, lambda conf: conf.regularization
            )
        )
    if (
        template.init_scale is None
        or template.init_scale[0] < template.init_scale[1]
    ):
        print("\nInit Scale")
        print(
            generate_param_report(
                result_set, template, lambda conf: conf.init_scale
            )
        )
    draw_weight_loss_graph(result_set, template)


def search_loop(dataset, search):
    print("Warming up training")
    warmup(dataset)

    print("Starting search")
    while True:
        conf = search.select_conf()

        weights = conf.model.weights
        extras = ""
        if conf.sample_len != 128:
            extras += f"  LEN {conf.sample_len}"
        if conf.regularization != 0.125:
            extras += f"  R {conf.regularization}"
        if conf.init_scale != 1.0:
            extras += f"  I {conf.init_scale}"
        print(
            f"T = {conf.t:3}     LR{conf.lr:6.3f}  B{conf.batch:4}  "
            + f"{conf.model} ({weights}){extras}"
        )

        manager = Manager(conf)
        manager.init(quiet=True)
        record = manager.train_and_eval(
            steps=None,
            time_limit=conf.t,
            train_set=dataset,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=args.log,
            quiet=True,
        )

        if record.time_round is None:
            print("Bad training time, skipping")
            record.time_round = conf.t
            record.loss = INF

        search.add_record(record)
        print()


def main(args, dataset, template):
    template = Template.from_args(args)
    default = parse_conf(args.default, default_from_template(template))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    add_template_args(parser)

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="a file with training data",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="LOG",
        help="path to a CSV file for logging",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "--min-max-weights",
        type=int,
        default=2048,
        help="minimum max-weights value in search",
    )
    parser.add_argument(
        "--default",
        type=str,
        default="dense.1.relu",
        help="default configuration (default: 'dense.1.relu LR0.125 LEN128 B64 R0.125 I1.0')",
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="don't run a search, only generate a report",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    dataset = build_dataset(args.data)
    template = Template.from_args(args)
    main(args, dataset=dataset, template=template)
    dataset.join()
    latency.report()
