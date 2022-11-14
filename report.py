import argparse
import logging
import math
import matplotlib.pyplot as plt

from texmo.results import ResultSet
from texmo.configuration import Configuration, Template, add_template_args
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

        weights = conf.model.weights

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
        weights = conf.model.weights
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
        l = len(str(conf.model))
        if l > spec_len:
            spec_len = l
    with open(filename, "w") as out:
        for log_weights in range(10, 25):
            weights = 2**log_weights
            first_weight_line = True
            for log_t in range(0, 10):
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
                    conf.model,
                    " " * (spec_len - len(str(conf.model))),
                    file=out,
                    end="",
                )
                print(f"| {score:.4f} | ", file=out, end="")
                print(f"B{conf.batch} LR{conf.lr}", file=out, end="")
                if conf.sample_len != 128:
                    print(f" LEN{conf.sample_len}", file=out, end="")
                if conf.regularization != 0.125:
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

def main(db, output, template):
    print(f"Creating ResultDB from {db}")
    result_db = ResultDB(db)

    print(f"Creaing ResultSet")
    result_set = ResultSet(result_db, template, populate_neighbors=False)

    top_confs, run_count = get_top_confs(result_set)
    print("prepared data for the report")
    print_top_confs(top_confs, run_count, output)
    print(f"wrote {output}")

    plt.xscale("log")
    plt.yscale("log")

    for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        x, y = max_points(result_set, t, 1e8)
        plt.plot(x, y, label=str(t))

    plt.savefig("results/graph.png")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    add_template_args(parser)

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="path to the SQLite database with the results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results/report.txt",
        help="path to the created report file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    template = Template.from_args(args)
    main(args.db, args.output, template)
