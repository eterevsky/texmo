import matplotlib.pyplot as plt
import sys

from record import TrainingRecord
from results import ResultSet


def max_points(result_set, t, maxx):
    x = []
    y = []
    min_loss = 5

    for cr in sorted(result_set.all_results(), key=lambda cr: cr.conf.spec.weights()):
        if not cr.scores or cr.conf.t != t: continue
        if cr.score > min_loss: continue

        weights = cr.conf.spec.weights()

        if x:
            if x[-1] < weights - 1:
                x.append(weights - 1)
                y.append(min_loss)
            elif x[-1] == weights:
                x.pop()
                y.pop()

        x.append(weights)
        y.append(cr.score)
        min_loss = cr.score

    x.append(maxx)
    y.append(min_loss)

    return x, y


def main(fname):
    result_set = ResultSet.from_csv(fname)

    plt.xscale("log")
    plt.yscale("log")

    for t in (1, 2, 4, 8, 16, 32, 64, 128):
        x, y = max_points(result_set, t, 1E8)
        plt.plot(x, y, label=str(t))

    plt.savefig("graph.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])