import argparse
import logging

from texmo.resultdb import ResultDB
from texmo.configuration import Configuration, Template, add_template_args, parse_conf, default_from_template
from texmo.dataset import build_dataset
from texmo import latency
from texmo.manager import Manager
from texmo.model2 import build_model
from texmo.search import Search
from texmo.common import INF


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
    model = build_model("suffix.4-rec.32.relu")
    manager = Manager(
        model,
        0.25,
        regularization=0.125,
        init_scale=1.0,
        use_model2=True,
    )
    manager.init(quiet=True)
    manager.train_and_eval(
        steps=None,
        time_limit=8,
        train_set=dataset,
        sample_len=128,
        batch_size=64,
        temp_steps=None,
        temp_dir=None,
        output_dir=None,
        log=None,
        quiet=True,
    )


def main(args, dataset, template):
    template = Template.from_args(args)
    default = parse_conf(args.default, default_from_template(template))
    assert template.match_conf(default)
    print("Default configuration:", default)

    print(f"Creating ResultDB from {args.db}")
    result_db = ResultDB(args.db)
    search = Search(result_db, template, default, args.min_max_weights)

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
            f"T = {conf.t:3}     LR{conf.lr:6.3f}  B{conf.batch:4}  {conf.model} ({weights}){extras}"
        )

        manager = Manager(
            conf.model,
            conf.lr,
            regularization=conf.regularization,
            init_scale=conf.init_scale,
            use_model2=True,
        )
        manager.init(quiet=True)
        record = manager.train_and_eval(
            steps=None,
            time_limit=conf.t,
            train_set=dataset,
            sample_len=conf.sample_len,
            batch_size=conf.batch,
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
        default=1024,
        help="minimum max-weights value in search",
    )
    parser.add_argument(
        "--default",
        type=str,
        default="dense.1.relu",
        help="default configuration (default: 'dense.1.relu LR0.125 LEN128 B64 R0.125 I1.0')"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    try:
        dataset = build_dataset(args.data)
        template = Template.from_args(args)
        main(args, dataset=dataset, template=template)
    except KeyboardInterrupt:
        print("\nInterrupted\n")
        latency.report()
    finally:
        dataset.join()
