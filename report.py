import argparse
import math
import matplotlib.pyplot as plt
import sqlite3

from texmo.results import ResultSet
from texmo.configuration import Configuration, Template
from texmo.spec import ModelSpec
from texmo.resultdb import ResultDB

def max_points(result_set, t, maxx):
    x = []
    y = []
    min_loss = 5

    for conf, results in result_set.all_results_by_weights():
        if conf.t != t or not results.score:
            continue
        if results.score > min_loss:
            continue

        weights = conf.spec.weights()

        if x:
            if x[-1] < weights - 1:
                x.append(weights - 1)
                y.append(min_loss)
            elif x[-1] == weights:
                x.pop()
                y.pop()

        x.append(weights)
        y.append(results.score)
        min_loss = results.score

    x.append(maxx)
    y.append(min_loss)

    return x, y


def get_top_confs(result_set):
    top_confs = {}  # (weights_limit, time_round) -> (conf, score)
    count = {}  # (weights_limit, time_round) -> count
    for conf, results in result_set.all_results_by_weights():
        weights = conf.spec.weights()
        weights_bucket = 2 ** math.ceil(math.log2(weights))
        try:
            count[(weights_bucket, conf.t)] += results.num_runs
        except KeyError:
            count[(weights_bucket, conf.t)] = results.num_runs

        while weights_bucket < 2**33:
            key = (weights_bucket, conf.t)
            if key in top_confs:
                top_conf, top_score = top_confs[key]
                if results.score < top_score:
                    top_confs[key] = (conf, results.score)
            else:
                top_confs[key] = (conf, results.score)
            weights_bucket *= 2

    return top_confs, count


def print_top_confs(top_confs, run_count, filename):
    spec_len = 0
    for conf, score in top_confs.values():
        l = len(str(conf.spec))
        if l > spec_len:
            spec_len = l
    with open(filename, "w") as out:
        for log_weights in range(10, 25):
            weights = 2**log_weights
            first_weight_line = True
            for log_t in range(0, 9):
                t = 2**log_t
                if (weights, t) not in top_confs:
                    continue
                conf, score = top_confs[(weights, t)]
                count = run_count.get((weights, t), 0)
                if (
                    count == 0
                    or log_weights > 10
                    and (weights // 2, t) in top_confs
                    and top_confs[(weights // 2, t)][0] == conf
                ):
                    continue
                if first_weight_line:
                    print(
                        "| ------- | ---- | ---- |",
                        "-" * spec_len,
                        "| ------ | --------- |",
                        file=out,
                    )
                    print(f"| {weights:>7} | ", file=out, end="")
                    first_weight_line = False
                else:
                    print("|         | ", file=out, end="")
                print(f"{t:>4} | {count:>4} | ", file=out, end="")
                print(
                    conf.spec,
                    " " * (spec_len - len(str(conf.spec))),
                    file=out,
                    end="",
                )
                print(f"| {score:.4f} | ", file=out, end="")
                print(f"B{conf.batch} LR{conf.lr}", file=out, end="")
                if conf.sample_len != 128:
                    print(f" LEN{conf.sample_len}", file=out, end="")
                if conf.regularization != 0.1:
                    print(f" R{conf.regularization}", file=out, end="")
                if conf.init_scale != 1.0:
                    print(f" i{conf.init_scale}", file=out, end="")

                print(" |", file=out)


def parse_interval_int(arg: str):
    if arg is None: return None
    comps = arg.split("-")
    assert len(comps) in (1, 2)
    if len(comps) == 1:
        v = int(comps[0])
        return (v, v)
    else:
        return int(comps[0]), int(comps[1])


def parse_interval_float(arg: str):
    if arg is None: return None
    comps = arg.split("-")
    assert len(comps) in (1, 2)
    if len(comps) == 1:
        v = float(comps[0])
        return (v, v)
    else:
        return float(comps[0]), float(comps[1])

def main(
    db,
    spec_regex,
    batch,
    lr,
    sample_len,
    regularization,
    init_scale,
    time,
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

    print(f"Creating ResultDB from {db}")
    result_db = ResultDB(db)

    print(f"Creaing ResultSet.")
    result_set = ResultSet(result_db, template, populate_neighbors=False)

    top_confs, run_count = get_top_confs(result_set)
    print("prepared data for report.txt")
    print_top_confs(top_confs, run_count, "results/report.txt")
    print("wrote report.txt")

    plt.xscale("log")
    plt.yscale("log")

    for t in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        x, y = max_points(result_set, t, 1e8)
        plt.plot(x, y, label=str(t))

    plt.savefig("results/graph.png")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

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
        "-b",
        "--batch",
        type=str,
        default=None,
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=str,
        default=None,
        help="range of acceptable learning rates (default: unrestricted)",
    )
    parser.add_argument(
        "--sample-len",
        type=str,
        default="128",
        help="range of acceptable sample lens (default: 128-128)",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=str,
        default="0.1",
        help="range of values for regularization coefficient (default: 0.1-0.1)",
    )
    parser.add_argument(
        "-i",
        "--init-scale",
        type=str,
        default="1.0",
        help="range of values of the coefficient for the initial weights (default: 0.1-0.1)",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=str,
        default="1-256",
        help="range of training time (default: 1-256)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
