import csv
import matplotlib.pyplot as plt

from record import TrainingRecord


def max_points(records, maxt, maxx):
    x = []
    y = []
    min_loss = 5

    for r in records:
        if r.train_time_s < maxt and r.loss < min_loss:
            if x and x[-1] < r.weights - 1:
                x.append(r.weights - 1)
                y.append(min_loss)

            x.append(r.weights)
            y.append(r.loss)
            min_loss = r.loss

    x.append(maxx)
    y.append(min_loss)

    return x, y


def main():
    records = []

    with open("search4-gpu.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            records.append(TrainingRecord.from_csv_tuple(row))

    records.sort(key=lambda r: r.weights)

    plt.xscale("log")
    plt.yscale("log")

    for t in (1, 2, 4, 8, 16, 32, 64, 128):
        x, y = max_points(records, t + 0.2, 1E7)
        plt.plot(x, y)

    plt.savefig("graph.png")
    plt.show()


if __name__ == "__main__":
    main()