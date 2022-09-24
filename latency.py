from statistics import quantiles
from time import perf_counter_ns


_measures = {}


class Timer(object):
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = perf_counter_ns()

    def __exit__(self, *args):
        end = perf_counter_ns()
        named_measures = _measures.get(self.name)
        if named_measures is None:
            named_measures = []
            _measures[self.name] = named_measures

        named_measures.append(end - self.start)


def timer(name):
    return Timer(name)


def report():
    for name, measures in sorted(_measures.items()):
        percentiles = quantiles(measures, n=100)
        p50th = percentiles[48] / 1E6
        p90th = percentiles[88] / 1E6
        p99th = percentiles[98] / 1E6
        print(f"{name}  {p50th:.3f} ms  {p90th:.3f} ms  {p99th:.3f} ms")