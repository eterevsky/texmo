import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_ylabel("loss")
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
