import argparse
import math
import matplotlib.pyplot as plt
import sqlite3

from results import ResultSet
from configuration import Configuration
from spec import ModelSpec
from resultdb import ResultDB

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
            for log_t in range(0, 16):
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


def main(
    db,
    vary,
    spec,
    batch,
    lr,
    sample_len,
    regularization,
    init_scale,
    time,
):
    init_conf = Configuration(
        None,
        ModelSpec.parse(spec),
        lr=lr,
        sample_len=sample_len,
        batch=batch,
        regularization=regularization,
        init_scale=init_scale,
        t=time,
    )

    print("Initial configuration:", init_conf)
    vary = vary.split(",")
    print("Variables:", vary)
    t = None if "time" in vary else time

    print(f"Creating ResultDB from {db}")
    result_db = ResultDB(db)

    print(f"Creaing ResultSet.")
    result_set = ResultSet(result_db, init_conf, vary, populate_neighbors=False)

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
        "-v",
        "--vary",
        type=str,
        default="layer,type,size,suffix,batch,lr,time",
        help="model parameters that can be varied with search. "
        + "A comma-separated list of struct, size, act, lr, batch, len.",
    )
    parser.add_argument(
        "-s", "--spec", type=str, default="dense.1.relu", help="initial spec"
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=256,
        help="batch size, has to be a power of two",
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=0.5,
        help="initial learning rate, has to be 0.1, 0.2 or 0.5 multiplied by a power of 10",
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        help="length of the training samples, power of two",
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        default=0.1,
        help="regularization coefficient, has to be 0.1, 0.2 or 0.5 multiplied by a power of 10",
    )
    parser.add_argument(
        "-i",
        "--init-scale",
        type=float,
        default=1.0,
        help="coefficient for the initial weights, has to be 0.1, 0.2 or 0.5 multiplied by a power of 10",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=None,
        help="training time (power of 2)",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
