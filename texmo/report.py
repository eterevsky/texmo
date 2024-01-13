import math
from io import StringIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .results import ResultSet
from .configuration2 import Template
from .common import INF, ttoa3


def max_points(result_set: ResultSet, t: int, maxx: int):
    x = []
    y = []
    max_loss = 5

    weight_loss = []
    max_weights = INF
    while True:
        conf_results = result_set.top_conf(t, max_weights)
        if not conf_results:
            break
        conf = conf_results.conf
        weight_loss.append((conf.model.weights, conf_results.median_score))
        max_weights = conf.model.weights - 1

    prev_loss = max_loss

    for weight, loss in reversed(weight_loss):
        if loss > max_loss:
            continue
        x.append(weight - 1)
        y.append(prev_loss)

        x.append(weight)
        y.append(loss)

        prev_loss = loss

    x.append(maxx)
    y.append(prev_loss)

    return x, y


def draw_weight_loss_graph(
    result_set: ResultSet, template: Template, train_time: tuple[float, float]
):
    fig, ax = plt.subplots()
    ax.set_ylabel("cross-entropy (bits per byte)")
    ax.set_xlabel("weights")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True)
    ax.set_yticks([1, 2, 3, 4])
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    yticks = [x / 10 for x in range(1, 30)] + [x / 10 for x in range(30, 50, 2)]
    ax.set_yticks(yticks, minor=True)

    t = 1
    legends = []

    top_conf_results = result_set.top_conf(train_time[1])

    while t <= train_time[1]:
        if t >= train_time[0]:
            x, y = max_points(result_set, t, top_conf_results.conf.model.weights * 4)
            ax.plot(x, y)
            legends.append(t)
        t *= 2

    ax.legend(legends)

    plt.savefig("results/graph.png")
    plt.show()


def draw_loss_by_time(result_set: ResultSet, template: Template):
    _, ax = plt.subplots()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cross-entropy (bits per byte)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(visible=True)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())

    times = []
    t = template.t[0]
    while t <= template.t[1]:
        times.append(t)
        t *= 2

    best = []
    best_for_bytes = []

    for t in times:
        best.append(result_set.top_conf(t).median_score)
        bytes_top_conf = result_set.top_conf_for_tokenset(t, "tokens256_raw_all")
        best_for_bytes.append(bytes_top_conf.median_score if bytes_top_conf else 4)

    ax.plot(times, best, label="best")
    ax.plot(times, best_for_bytes, label="best for bytes")

    plt.show()


def get_top_confs(
    result_set: ResultSet,
    template: Template,
    min_max_weights: int,
    train_time: tuple[float, float],
    system: str,
):
    top_confs = {}  # (weights_limit, planned_time_s) -> (conf, score)
    count = {}  # (weights_limit, planned_time_s) -> count
    tlo, thi = train_time
    for conf_results in result_set.get_results_by_weights():
        t = conf_results.median_time(system)
        if not conf_results.median_score or not t:
            continue
        t = 2 ** math.ceil(math.log2(t))
        if not tlo <= t <= thi:
            continue
        conf = conf_results.conf
        weights = conf.model.weights
        weights_bucket = max(2 ** math.ceil(math.log2(weights)), min_max_weights)
        try:
            count[(weights_bucket, t)] += len(conf_results.runs)
        except KeyError:
            count[(weights_bucket, t)] = len(conf_results.runs)

        while weights_bucket < 2**33:
            key = (weights_bucket, t)
            if key in top_confs:
                top_conf, top_score = top_confs[key]
                if (
                    conf_results.median_score is not None
                    and conf_results.median_score < top_score
                ):
                    assert conf_results.median_score is not None
                    top_confs[key] = (conf, conf_results.median_score)
            else:
                assert conf_results.median_score is not None
                top_confs[key] = (conf, conf_results.median_score)
            weights_bucket *= 2

    return top_confs, count


def print_top_confs(top_confs, run_count) -> str:
    out = StringIO()
    spec_len = 0
    for conf, score in top_confs.values():
        l = len(str(conf.model))
        if l > spec_len:
            spec_len = l
    for log_weights in range(6, 25):
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
            model_str = str(conf.model)
            print(
                model_str,
                " " * (spec_len - len(model_str)),
                file=out,
                end="",
            )
            print(f"| {score:.4f} | ", file=out, end="")
            print(f"B{conf.batch} LR{conf.lr:.4f}", file=out, end="")
            print(f" LEN{conf.length}", file=out, end="")

            print(" |", file=out)

    return out.getvalue()


# def generate_report_by_weight(
#     result_set: ResultSet,
#     template: Template,
#     min_max_weights: int,
#     train_time: tuple[float, float],
#     system: str,
# ) -> str:
#     top_confs, run_count = get_top_confs(
#         result_set, template, min_max_weights, train_time, system
#     )
#     return print_top_confs(top_confs, run_count)


def generate_report_by_weight(result_set: ResultSet, template: Template, system: str) -> str:
    out = StringIO()
    best_loss = INF
    prev_best = None
    for conf_results in result_set.get_results_by_weights():
        if prev_best is not None and conf_results.conf.model.weights > prev_best.conf.model.weights:
            t = prev_best.median_time(system)
            if t:
                t = ttoa3(t)
            else:
                t = "?    "
            print(f"{prev_best.median_score:.4f}  {t}  {prev_best.conf}", file=out)
            prev_best = None

        if (conf_results.median_score is None or
            conf_results.median_score > best_loss or
            not template.match(conf_results.conf)):
            continue

        prev_best = conf_results
        best_loss = conf_results.median_score

    if prev_best is not None:
        t = prev_best.median_time(system)
        if t:
            t = ttoa3(t)
        else:
            t = "?"
        print(f"{prev_best.median_score:.4f}  {t}  {prev_best.conf}", file=out)

    return out.getvalue()


def generate_report_by_weight_(
    result_set: ResultSet,
    template: Template,
    min_max_weights: int,
    train_time: tuple[float, float],
    system: str,
) -> str:
    max_weights = 32
    while True:
        print("\nW ≤", max_weights)
        current_best = None
        printed_conf = False
        for conf_results in result_set.get_results_by_time(max_weights=max_weights):
            if not conf_results.median_score:
                continue
            if (
                current_best is not None
                and conf_results.median_score > current_best.median_score
            ):
                continue
            if (
                current_best is not None
                and current_best.median_time(system) < 1
                and conf_results.median_time(system) > 1
                and current_best.conf.model.weights > max_weights // 2
            ):
                print(
                    f"{current_best.median_score:.4f} "
                    + f"{ttoa3(current_best.median_time(system))}  "
                    + f"{conf_results.conf}"
                )
                printed_conf = True

            current_best = conf_results

            if printed_conf or (
                current_best.median_time(system) > 1
                and current_best.conf.model.weights > max_weights // 2
            ):
                print(
                    f"{current_best.median_score:.4f} "
                    + f"{ttoa3(current_best.median_time(system))}  "
                    + f"{conf_results.conf}"
                )
                printed_conf = True

        if not printed_conf:
            break
        max_weights *= 2


def generate_max_report(
    result_set: ResultSet, template: Template, train_time: tuple[float, float]
) -> str:
    out = StringIO()
    lo, hi = train_time
    assert 1 <= lo <= hi

    # runs_count = result_set.runs_count_per_t()
    t = lo
    while t <= hi:
        print(f"\nT ≤ {t}", file=out)

        for conf_results in result_set.top_confs(t, limit=5):
            num_runs = len(conf_results.runs)
            score = f"{conf_results.median_score:.4f} ({num_runs})"
            if num_runs < 10:
                score += " "
            print(f"{score} {conf_results.conf}", file=out)

        t *= 2

    return out.getvalue()


def generate_param_report(
    result_set: ResultSet, template: Template, extract_param, is_float=True
) -> str:
    out = StringIO()
    lo, hi = template.t

    t = lo
    while t <= hi:
        top_confs: dict = {}  # bucket -> (conf, score)
        count = {}  # bucket -> count

        for conf_results in result_set.all_results_for_t(t):
            conf = conf_results.conf
            bucket = extract_param(conf)

            if bucket not in top_confs:
                top_confs[bucket] = (conf, conf_results.median_score)
                count[bucket] = len(conf_results.runs)
            else:
                if top_confs[bucket][1] > conf_results.median_score:
                    top_confs[bucket] = (conf, conf_results.median_score)

                count[bucket] += len(conf_results.runs)

        print(f"\nT = {t}", file=out)
        for bucket in sorted(top_confs.keys()):
            conf, score = top_confs[bucket]
            runs = count[bucket]
            c = conf_to_string(conf)
            if is_float:
                b = f"{bucket:.3f}"
            else:
                b = f"{bucket:5}"
            print(f"{b}  {score:.4f} ({runs:5})  {c}", file=out)

        t *= 2

    return out.getvalue()
