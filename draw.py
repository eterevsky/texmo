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


def get_top_confs(result_set):
    top_confs = {}  # (weights_limit, time_round)
    count = {}  # (weights_limit, time_round) -> count
    for cr in result_set.all_results():
        try:
            count[(cr.weights_limit, cr.conf.t)] += len(cr.scores)
        except KeyError:
            count[(cr.weights_limit, cr.conf.t)] = len(cr.scores)

        weights = cr.weights_limit
        while weights < 2**33:
            key = (weights, cr.conf.t)
            if key in top_confs:
                top_cr = top_confs[key]
                if cr.score < top_cr.score:
                    top_confs[key] = cr
            else:
                top_confs[key] = cr
            weights *= 2

    return top_confs, count


def print_top_confs(top_confs, run_count, filename):
    spec_len = 0
    for cr in top_confs.values():
        l = len(str(cr.conf.spec))
        if l > spec_len:
            spec_len = l
    with open(filename, "w") as out:
        for log_weights in range(10, 25):
            weights = 2 ** log_weights
            first_weight_line = True
            for log_t in range(0, 8):
                t = 2 ** log_t
                if (weights, t) not in top_confs: continue
                cr = top_confs[(weights, t)]
                count = run_count.get((weights, t), 0)
                if count == 0 or log_weights > 10 and top_confs[(weights//2, t)].conf == cr.conf:
                    continue
                if first_weight_line:
                    print('| ------- | ---- | ---- |', '-' * spec_len, '| ------ | --------- |', file=out)
                    print(f'| {weights:>7} | ', file=out, end='')
                    first_weight_line = False
                else:
                    print('|         | ', file=out, end='')
                print(f"{t:>4} | {count:>4} | ", file=out, end='')
                print(cr.conf.spec, ' ' * (spec_len - len(str(cr.conf.spec))), file=out, end='')
                print(f"| {cr.score:.4f} | ", file=out, end='')
                print(f"B{cr.conf.batch} LR{cr.conf.lr} |", file=out)


def main(fname):
    result_set = ResultSet.from_csv(fname)

    top_confs, run_count = get_top_confs(result_set)
    print_top_confs(top_confs, run_count, "report.txt")

    plt.xscale("log")
    plt.yscale("log")

    for t in (1, 2, 4, 8, 16, 32, 64, 128):
        x, y = max_points(result_set, t, 1E8)
        plt.plot(x, y, label=str(t))

    plt.savefig("graph.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])