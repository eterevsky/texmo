import argparse
import math
import random

from texmo.resultdb import ResultDB
from texmo.configuration import Configuration, Template
from texmo.dataset import build_dataset
from texmo import latency
from texmo.layered import LayeredModel2
from texmo.manager import Manager
from texmo.search import Search
from texmo.spec import ModelSpec
from texmo.common import INF


# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6


def parse_interval_int(arg: str):
    if arg is None:
        return None
    comps = arg.split("-")
    assert len(comps) in (1, 2)
    if len(comps) == 1:
        v = int(comps[0])
        return (v, v)
    else:
        return int(comps[0]), int(comps[1])


def parse_interval_float(arg: str):
    if arg is None:
        return None
    comps = arg.split("-")
    assert len(comps) in (1, 2)
    if len(comps) == 1:
        v = float(comps[0])
        return (v, v)
    else:
        return float(comps[0]), float(comps[1])


def pick_default_value(default, range):
    if default is not None:
        assert range is None or range[0] <= default <= range[1]
        return default
    assert range is not None
    return range[0]


def warmup(dataset):
    model = LayeredModel2.parse("suffix.4-rec.32.relu")
    manager = Manager(
        model,
        0.2,
        regularization=0.1,
        init_scale=1.0,
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


def main(
    data,
    dataset,
    log,
    db,
    spec_regex,
    spec_default,
    batch,
    batch_default,
    lr,
    lr_default,
    sample_len,
    sample_len_default,
    regularization,
    regularization_default,
    init_scale,
    init_scale_default,
    time,
    max_weights,
    min_max_weights,
):
    template = Template(
        spec_regex=spec_regex,
        batch=parse_interval_int(batch),
        lr=parse_interval_float(lr),
        sample_len=parse_interval_int(sample_len),
        regularization=parse_interval_float(regularization),
        init_scale=parse_interval_float(init_scale),
        t=parse_interval_int(time),
    )
    init_conf = Configuration(
        None,
        ModelSpec.parse(spec_default),
        lr=pick_default_value(lr_default, template.lr),
        sample_len=pick_default_value(sample_len_default, template.sample_len),
        batch=pick_default_value(batch_default, template.batch),
        regularization=pick_default_value(
            regularization_default, template.regularization
        ),
        init_scale=pick_default_value(init_scale_default, template.init_scale),
        t=template.t[0],
    )
    print("Initial configuration:", init_conf)

    print(f"Creating ResultDB from {db}")
    result_db = ResultDB(db)
    search = Search(
        result_db, template, init_conf, max_weights, min_max_weights
    )

    print("Warming up training")
    warmup(dataset)

    print("Starting search")
    while True:
        conf = search.select_conf()

        weights = conf.spec.weights()
        extras = ""
        if conf.sample_len != 128:
            extras += f"  LEN {conf.sample_len}"
        if conf.regularization != 0.1:
            extras += f"  R {conf.regularization}"
        if conf.init_scale != 1.0:
            extras += f"  I {conf.init_scale}"
        print(
            f"T = {conf.t:3}     LR{conf.lr:5}  B{conf.batch:4}  {conf.spec} ({weights}){extras}"
        )

        model = LayeredModel2.parse(str(conf.spec))
        manager = Manager(
            model,
            conf.lr,
            regularization=conf.regularization,
            init_scale=conf.init_scale,
        )
        manager.init(quiet=True)
        assert model.total_weights(manager.weights) == weights
        record = manager.train_and_eval(
            steps=None,
            time_limit=conf.t,
            train_set=dataset,
            sample_len=conf.sample_len,
            batch_size=conf.batch,
            temp_steps=None,
            temp_dir=None,
            output_dir=None,
            log=log,
            quiet=True,
        )

        if record.time_round is None:
            print("Bad training time, skipping")
            record.time_round = conf.t
            record.loss = INF

        search.add_record(record)
        print()


def parse_args():
    parser = argparse.ArgumentParser()

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
        "-s",
        "--spec-regex",
        type=str,
        default=None,
        help="regex convering the acceptable specs (default: unrestricted)",
    )
    parser.add_argument(
        "--spec-default",
        type=str,
        default="dense.1.relu",
        help="initial spec (default: dense.1.relu)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "--batch-default",
        type=int,
        default=64,
        help="default batch size. Should agree with limits from -b and be a power of 2. (default: 64)",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates (default: unrestricted)",
    )
    parser.add_argument(
        "--lr-default",
        type=float,
        default=0.1,
        help="default learning rate. (default: 0.1)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="128",
        help="range of acceptable sample lens (default: 128-128)",
    )
    parser.add_argument(
        "--sample-len-default",
        type=int,
        default=None,
        help="default sample length (default: taken from --sample-len)",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=str,
        default="0.1",
        help="range of values for regularization coefficient (default: 0.1-0.1)",
    )
    parser.add_argument(
        "--regularization-default",
        type=float,
        default=None,
        help="default value for regularization coefficient (default: taken from -r)",
    )
    parser.add_argument(
        "-i",
        "--init-scale",
        type=str,
        default="1.0",
        help="range of values of the coefficient for the initial weights (default: 0.1-0.1)",
    )
    parser.add_argument(
        "--init-scale-default",
        type=float,
        default=None,
        help="default value for init scaling coefficient (default: taken from --init-scale)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )
    parser.add_argument(
        "--min-max-weights",
        type=int,
        default=1024,
        help="minimum max-weights value in search",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined",
    )

    return parser.parse_args()


if __name__ == "__main__":
    print("TexMo parameter search")
    args = parse_args()
    try:
        dataset = build_dataset(args.data)
        main(dataset=dataset, **vars(args))
    except KeyboardInterrupt:
        print("\nInterrupted\n")
        latency.report()
    finally:
        dataset.join()
