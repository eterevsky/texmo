import argparse
from itertools import islice
import math
import random

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


def select_time(result_set):
    with latency.timer("select_time"):
        total_runs = result_set.total_runs

        # Number of expected runs for t
        t = 1
        expected_runs = total_runs * (1 - RUNS_EXP)

        best_t = 1
        most_lacking_runs = 0

        print(f"Total runs: {total_runs}")

        while expected_runs >= 1:
            complete_runs = result_set.runs_count(t)
            gap = expected_runs - complete_runs
            print(
                f"t = {t}  complete = {complete_runs}  expected = {expected_runs:.2f}  gap = {gap:.2f}"
            )
            if expected_runs - complete_runs > most_lacking_runs:
                most_lacking_runs = expected_runs - complete_runs
                best_t = t
            t *= 2
            expected_runs *= RUNS_EXP

        return best_t


def select_max_weights(result_set, t):
    with latency.timer("select_max_weights"):
        top_cr = result_set.top_conf(t)
        if top_cr is None:
            return 1024
        maxw = top_cr.conf.spec.weights()
        l = random.uniform(10, math.log2(maxw) + 1)
        return int(2**l)


def select_conf(result_set, time_limit):
    with latency.timer("select_conf"):
        t = time_limit if time_limit is not None else select_time(result_set)
        max_weights = select_max_weights(result_set, t)

        print(f"\nT = {t}  W ≤ {max_weights}")

        if t > 1:
            with latency.timer("select_conf-previous"):
                t2 = t // 2
                nconfs = result_set.confs_count(t2)
                n_top_confs = round(
                    nconfs / (4 * (math.log2(max_weights) - 9))
                )
                print(f"Checking {n_top_confs} out of {nconfs} confs for T = {t2}, W ≤ {max_weights} ")
                for i, cr in enumerate(result_set.top_confs(t2, max_weights)):
                    if i >= n_top_confs: break
                    conf_t = cr.conf._replace(t=t)
                    if not result_set.find(conf_t):
                        print(f"Picked conf #{i}")
                        return conf_t

        with latency.timer("select_conf-neighbors"):
            nruns = result_set.runs_count(t, max_weights)
            # Estimation for the number of runs with weights between
            # max_weights // 2 and max_weights
            effective_nruns = nruns / (math.log2(max_weights) - 9)

            # The number of runs per conf depends on the order by cluster score.
            # The number of confs with >= n+1 runs is 1/3 of all the confs with
            # >= n runs.
            # This means that confs with the order <= 2 * enr / 3^k should have
            # >= k runs.

            best_conf = None
            best_gap = -1

            print()

            k = 16
            for i, cr in enumerate(
                result_set.top_cluster_confs(t, max_weights)
            ):
                while i + 1 > 2 * effective_nruns / 3 ** (k - 1) and k > 1:
                    k -= 1
                conf = cr.conf
                if i < 20:
                    score = f"{cr.score:.4f}" if cr.scores else "      "
                    count = len(cr.scores)
                    print(
                        f"{conf.spec:60}  LR{conf.lr:6}  LEN{conf.sample_len:4}  "
                        + f"B{conf.batch:4}  R{conf.regularization:4}  "
                        + f"I{conf.init_scale:4}  {score} ({count})"
                    )
                gap = k - len(cr.scores)
                if gap > best_gap:
                    best_conf = conf
                    best_gap = gap
                if gap > k and i > 20:
                    break

            if best_conf is not None:
                return best_conf

        default_conf = Configuration(
            spec=ModelSpec.parse("dense.1.relu"),
            lr=0.5,
            sample_len=128,
            batch=256,
            regularization=0.1,
            init_scale=1,
            t=t,
        )
        return default_conf


def main(
    data,
    model_spec,
    learning_rate,
    sample_len,
    batch_size,
    regularization,
    time_limit,
    log,
    load,
    max_weights,
    init_scale,
    vary="",
):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()

    vary = vary.split(",")

    if load:
        print(f"Loading old results from {load}")
        result_set = ResultSet.from_csv(load, time_limit, vary)
        confs = result_set.total_confs

        print(f"Loaded {result_set.total_runs} runs, {result_set.total_confs} confs")
    else:
        result_set = ResultSet(time_limit, vary)

    first = True
    while True:
        conf = select_conf(result_set, time_limit)
        weights = conf.spec.weights()
        print(
            f"\nT = {conf.t}  {conf.spec} ({weights})  LR {conf.lr}  LEN {conf.sample_len}  "
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

        if first:
            first = False
        else:
            result_set.add_record(record)
            # res = result_set.find(conf)
            # print(res.neighbors)
            # return


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
        "-c",
        "--model-spec",
        metavar="SPEC",
        default=None,
        help="initial model spec",
    )
    parser.add_argument(
        "--max-weights",
        type=int,
        default=None,
        help="the maximum number of weights in the model",
    )
    parser.add_argument(
        "-v",
        "--vary",
        type=str,
        default="struct,size,suffix,lr,batch,activation",
        help="model parameters that can be varied with search. "
        + "A comma-separated list of struct, size, act, lr, batch, len.",
    )
    parser.add_argument(
        "-t",
        "--time-limit",
        type=int,
        default=None,
        metavar="SECONDS",
        help="time limit for training",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        metavar="RATE",
        help="learning rate",
        default=0.05,
    )
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        help="L2 regularization coefficient",
        default=0.1,
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=128,
        metavar="LEN",
        help="length of text fragments used for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        metavar="BATCH",
        default=32,
        help="training data batch size",
    )
    parser.add_argument(
        "--log",
        default=None,
        metavar="LOG",
        help="path to a CSV file for logging",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="LOG",
        help="a CSV log file with previous runs",
    )
    parser.add_argument(
        "--init_scale",
        default=1,
        type=float,
        help="scale weight initialization",
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
