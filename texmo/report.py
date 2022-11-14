import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

from texmo.results import ResultSet
from texmo.configuration import Configuration, Template, add_template_args


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


def draw_weight_loss_graph(result_set: ResultSet, template: Template):
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

    top_conf = result_set.top_conf_all_t(template.t[0], template.t[1])

    while t <= template.t[1]:
        if t >= template.t[0]:
            x, y = max_points(result_set, t, top_conf.model.weights * 4)
            ax.plot(x, y)
            legends.append(t)
        t *= 2

    ax.legend(legends)

    plt.savefig("results/graph.png")
    plt.show()


def get_top_confs(result_set, template, min_max_weights):
    top_confs = {}  # (weights_limit, time_round) -> (conf, score)
    count = {}  # (weights_limit, time_round) -> count
    tlo, thi = template.t
    for conf, results in result_set.all_results_by_weights():
        if not tlo <= conf.t <= thi:
            continue
        weights = conf.model.weights
        weights_bucket = max(
            2 ** math.ceil(math.log2(weights)), min_max_weights
        )
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


def print_top_confs(top_confs, run_count) -> str:
    out = StringIO()
    report = ""
    spec_len = 0
    for conf, score in top_confs.values():
        l = len(str(conf.model))
        if l > spec_len:
            spec_len = l
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

    return out.getvalue()


def generate_report_by_weight(
    result_set: ResultSet, template: Template, min_max_weights: int
) -> str:
    top_confs, run_count = get_top_confs(result_set, template, min_max_weights)
    return print_top_confs(top_confs, run_count)


def print_conf(conf):
    extras = ""
    if conf.sample_len != 128:
        extras += f"  LEN {conf.sample_len}"
    if conf.regularization != 0.125:
        extras += f"  R {conf.regularization}"
    if conf.init_scale != 1.0:
        extras += f"  I {conf.init_scale}"
    return f"{conf.model}   LR{conf.lr:.3f}  B{conf.batch}{extras}"


def generate_max_report(result_set: ResultSet, template: Template) -> str:
    out = StringIO()
    lo, hi = template.t
    assert 1 <= lo <= hi

    runs_count = result_set.runs_count_per_t()
    t = lo
    while t <= hi:
        runs = runs_count.get(t, 0)
        print(f"\nT = {t:3}  runs = {runs:5}", file=out)

        for conf, results in result_set.top_confs(t, limit=5):
            score = f"{results.score:.4f} ({results.num_runs})"
            if results.num_runs < 10:
                score += " "
            c = print_conf(conf)
            print(f"{score} {c}", file=out)

        t *= 2

    return out.getvalue()


def generate_param_report(result_set: ResultSet, template: Template, extract_param, is_float=True) -> str:
    out = StringIO()
    lo, hi = template.t

    t = lo
    while t <= hi:
        top_confs = {}  # bucket -> (conf, score)
        count = {}  # bucket -> count

        for conf, results in result_set.all_results_for_t(t):
            bucket = extract_param(conf)

            if bucket not in top_confs:
                top_confs[bucket] = (conf, results.score)
                count[bucket] =results.num_runs
            else:
                if top_confs[bucket][1] > results.score:
                    top_confs[bucket] = (conf, results.score)

                count[bucket] += results.num_runs

        print(f'\nT = {t}', file=out)
        for bucket in sorted(top_confs.keys()):
            conf, score = top_confs[bucket]
            runs = count[bucket]
            c = print_conf(conf)
            if is_float:
                b = f"{bucket:.3f}"
            else:
                b = f"{bucket:5}"
            print(f"{b}  {score:.4f} ({runs:5})  {c}", file=out)

        t *= 2

    return out.getvalue()
