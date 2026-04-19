import logging
from statistics import quantiles
from time import perf_counter_ns

from .common import itoa3, ttoa3

_start = perf_counter_ns()
_measures: dict[str, list[float]] = {}


class Timer(object):
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = perf_counter_ns()
        return self

    @property
    def elapsed(self) -> float:
        now = perf_counter_ns()
        return (now - self.start) * 1E-9

    def value(self) -> float:
        now = perf_counter_ns()
        return (now - self.start) * 1E-9

    def __exit__(self, *args):
        end = perf_counter_ns()
        named_measures = _measures.get(self.name)
        if named_measures is None:
            named_measures = []
            _measures[self.name] = named_measures

        named_measures.append(end - self.start)


def timer(name):
    return Timer(name)


def get_report() -> str:
    total_time = perf_counter_ns() - _start
    report = "Latency report:\n"
    for name, measures in sorted(_measures.items()):
        if len(measures) == 1:
            val = measures[0] * 1E-9
            report += f"{name}(1)  {ttoa3(val)}\n"
        else:
            percentiles = quantiles(measures, n=100)
            p50th = percentiles[48] * 1e-9
            p90th = percentiles[88] * 1e-9
            p99th = percentiles[98] * 1e-9
            avg = sum(measures) / len(measures) * 1e-9
            total = len(measures)
            percent = (sum(measures) / total_time) * 100
            report += f"{name}({itoa3(total)})  {ttoa3(p50th)}  {ttoa3(p90th)}  {ttoa3(p99th)}  AVG {ttoa3(avg)}  {percent:.1f}%\n"
    return report

def report():
    logging.info(get_report())
