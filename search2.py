import argparse
from itertools import islice
import math
import random
from resultdb import ResultDB

import results
from results import ResultSet, Configuration
from dataset import DataSet
import latency
from layered import LayeredModel2
from manager import Manager
from spec import ModelSpec
from train import train_and_eval


# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6


def select_time(result_set, max_time):
    with latency.timer("select_time"):
        runs_count = result_set.runs_count_per_t()
        total_runs = 0
        t = 1
        i = 0
        while t <= max_time and t in runs_count:
            total_runs += runs_count.get(t, 0)
            i += 1
            t *= 2

        if total_runs == 0:
            return 1

        # total = x + x * re + x * re^2 + ... + x * re^(i-1) = x * (1 - re^i) / (1 - re)

        # Number of expected runs for t
        t = 1
        expected_runs = total_runs * (1 - RUNS_EXP) / (1 - RUNS_EXP**i)

        best_t = 1
        most_lacking_runs = -100

        print(f"Total runs: {total_runs}")

        while expected_runs >= 1 and (max_time is None or t <= max_time):
            complete_runs = runs_count.get(t, 0)
            gap = expected_runs - complete_runs
            print(
                f"t = {t}  complete = {complete_runs}  "
                + f"expected = {expected_runs:.2f}  gap = {gap:.2f}"
            )
            if expected_runs - complete_runs > most_lacking_runs:
                most_lacking_runs = expected_runs - complete_runs
                best_t = t
            t *= 2
            expected_runs *= RUNS_EXP

        return best_t


def select_max_weights(result_set, t):
    with latency.timer("select_max_weights"):
        top_conf = result_set.top_conf(t)
        if top_conf is None:
            return 1024
        maxw = top_conf.spec.weights()
        l = random.uniform(10, math.log2(maxw) + 2)
        return int(2**l)


def print_top_confs(result_set, t, max_weights):
    with latency.timer("print_top_confs"):
        print(f"\nT = {t}  W ≤ {max_weights}\n")

        for i, (conf, results) in enumerate(
            result_set.top_cluster_confs(t, max_weights, limit=20)
        ):
            if i >= 20:
                break
            score = f"{results.score:.4f}" if results.score else "      "
            print(
                f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
                + f"B{conf.batch:4}  R{conf.regularization:4}  "
                + f"I{conf.init_scale:4}  {score} ({results.num_runs})"
            )
        print()


def select_conf_prev_time(result_set, t, max_weights, fixed_weights):
    with latency.timer("select_conf_prev_time"):
        t2 = t // 2
        with latency.timer("select_conf_prev_time-confs_count"):
            nconfs = result_set.confs_count(t2)

        if fixed_weights:
            n_top_confs = nconfs // 8
        else:
            actual_max_weights = max_weights
            for top_conf, _ in result_set.top_cluster_confs(
                t, max_weights, limit=1
            ):
                actual_max_weights = top_conf.spec.weights()
            n_top_confs = round(nconfs / (8 * (math.log2(actual_max_weights) - 9)))

        print(
            f"Checking {n_top_confs} out of {nconfs} confs for "
            + f"T = {t2}, W ≤ {max_weights}"
        )
        for i, conf in enumerate(result_set.top_confs(t2, max_weights)):
            if i >= n_top_confs:
                break
            conf_t = conf._replace(id=None, t=t)
            conf_t, score = result_set.find(conf_t)
            if score is None:
                print(f"Picked conf #{i}")
                return conf_t
        return None


def select_conf_neighbors(result_set, t, max_weights):
    with latency.timer("select_conf_neighbors"):
        nruns = result_set.runs_count(t, max_weights)

        actual_max_weights = max_weights
        for top_conf, _ in result_set.top_cluster_confs(
            t, max_weights, limit=1
        ):
            actual_max_weights = top_conf.spec.weights()

        # Estimation for the number of runs with weights between
        # max_weights // 2 and max_weights
        effective_nruns = nruns / (math.log2(actual_max_weights) - 9)

        # The number of runs per conf depends on the order by cluster score.
        # The number of confs with >= n+1 runs is 1/3 of all the confs with
        # >= n runs.
        # This means that confs with the order <= 2 * enr / 3^k should have
        # >= k runs.

        best_conf = None
        best_gap = -1

        k = 16
        for i, (conf, results) in enumerate(
            result_set.top_cluster_confs(t, max_weights)
        ):
            while i + 1 > 2 * effective_nruns / 3 ** (k - 1) and k > 1:
                k -= 1
            gap = k - results.num_runs
            if gap > best_gap:
                best_conf = conf
                best_gap = gap
            if gap > k:
                break

        return best_conf


def select_conf(result_set, max_time, t, max_weights, init_conf):
    with latency.timer("select_conf"):
        fixed_weights = max_weights is not None

        fixed_time = t is not None
        if t is None:
            t = select_time(result_set, max_time)
        if not fixed_weights:
            max_weights = select_max_weights(result_set, t)

        print_top_confs(result_set, t, max_weights)

        if t > 1 and not fixed_time:
            conf = select_conf_prev_time(result_set, t, max_weights, fixed_weights)
            if conf is not None:
                return conf

        conf = select_conf_neighbors(result_set, t, max_weights)
        if conf is not None:
            return conf

        return init_conf._replace(t=t)


def main(
    data,
    db,
    log,
    max_time,
    vary,
    spec,
    batch,
    lr,
    sample_len,
    regularization,
    init_scale,
    time,
    max_weights,
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

    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()

    print(f"Creating ResultDB from {db}")
    result_db = ResultDB(db)

    print(f"Creaing ResultSet.")
    result_set = ResultSet(result_db, init_conf, vary)

    print("Starting search")

    first = True
    while True:
        conf = select_conf(result_set, max_time, t, max_weights, init_conf)
        weights = conf.spec.weights()
        print(
            f"\nT = {conf.t}  {conf.spec} ({weights})  LR {conf.lr}  "
            + f"LEN {conf.sample_len}  "
            + f"B {conf.batch}  R {conf.regularization}  I {conf.init_scale}"
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
        record = train_and_eval(
            manager,
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

        if first:
            first = False
        else:
            with latency.timer("add_record"):
                result_set.add_record(record)
        print()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="directory with training data",
    )
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
        "--max-time",
        type=int,
        default=None,
        metavar="SECONDS",
        help="maximum time limit for training",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="max weights. Will vary if left undefined",
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

    return parser.parse_args()


if __name__ == "__main__":
    print("TexMo parameter search")
    args = parse_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        print("\nInterrupted\n")
        latency.report()
