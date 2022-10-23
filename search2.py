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
from train import train_and_eval


# The number of runs with t = 2^(k+1) should be RUNS_EXP time number of runs
# with t = 2^k
RUNS_EXP = 0.6
INF = float("inf")


def select_time(result_set, time_bounds):
    with latency.timer("select_time"):
        lo, hi = time_bounds
        if lo == hi:
            return lo

        runs_count = result_set.runs_count_per_t()
        runs = runs_count.get(lo, 0)
        print(f"t = {lo:3}  complete = {runs:5}")

        if runs == 0:
            return lo

        min_ratio = RUNS_EXP
        best_t = lo
        prev_runs = runs

        t = 2 * lo
        while t <= hi:
            runs = runs_count.get(t, 0)
            ratio = runs / prev_runs
            print(f"t = {t:3}  complete = {runs:5}  ratio = {ratio:.2f}")
            if runs == 0:
                return t
            if ratio < min_ratio:
                min_ratio = ratio
                best_t = t
            prev_runs = runs
            t *= 2

        return best_t


def select_max_weights(result_set, t, min_max_weights):
    with latency.timer("select_max_weights"):
        top_conf = result_set.top_conf(t)
        if top_conf is None:
            return min_max_weights
        maxw = top_conf.spec.weights()
        if min_max_weights >= maxw + 2:
            return min_max_weights
        l = random.uniform(math.log2(min_max_weights), math.log2(maxw) + 2)
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
            n_top_confs = nconfs // 12
        else:
            actual_max_weights = max_weights
            for top_conf, _ in result_set.top_cluster_confs(
                t, max_weights, limit=1
            ):
                actual_max_weights = top_conf.spec.weights()
            n_top_confs = round(
                nconfs / (12 * (math.log2(actual_max_weights) - 9))
            )

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


def select_conf(result_set, template, max_weights, init_conf, min_max_weights):
    with latency.timer("select_conf"):
        fixed_weights = max_weights is not None

        t = select_time(result_set, template.t)
        if not fixed_weights:
            max_weights = select_max_weights(result_set, t, min_max_weights)

        print_top_confs(result_set, t, max_weights)

        if t > template.t[0]:
            conf = select_conf_prev_time(
                result_set, t, max_weights, fixed_weights
            )
            if conf is not None:
                return conf

        conf = select_conf_neighbors(result_set, t, max_weights)
        if conf is not None:
            return conf

        return init_conf._replace(t=t)


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
    train_and_eval(
        manager,
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
    search = Search(result_db, template, init_conf, max_weights, min_max_weights)

    print("Warming up training")
    warmup(dataset)

    print("Starting search")
    while True:
        conf = search.select_conf()

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

        search.add_record(record)
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
