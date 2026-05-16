"""Per-call latency tracking.

Public surface is unchanged: `Timer` / `timer(name)` as a context
manager that records the wall-clock span on `__exit__`, `get_report()`
to build the percentiles report, and `report()` to log it.

Internally every `__exit__` posts a `_Record` message onto a private
queue consumed by a single daemon thread that owns the `_measures`
dict; `get_report()` posts a `_Report` with a one-shot reply queue
and blocks until the thread answers. FIFO ordering on the queue
guarantees that every record posted before `get_report()` is folded
into the response.
"""

import logging
import threading
from dataclasses import dataclass
from queue import Queue
from statistics import quantiles
from time import perf_counter_ns

from .common import itoa3, ttoa3


_start = perf_counter_ns()


# --- internal queue messages ------------------------------------------------


@dataclass
class _Record:
    name: str
    elapsed_ns: int


@dataclass
class _Report:
    reply: Queue


_LatencyMessage = _Record | _Report

_queue: Queue = Queue()


def _build_report(measures: dict[str, list[int]]) -> str:
    total_time = perf_counter_ns() - _start
    out = "Latency report:\n"
    for name, ms in sorted(measures.items()):
        if len(ms) == 1:
            val = ms[0] * 1E-9
            out += f"{name}(1)  {ttoa3(val)}\n"
        else:
            percentiles = quantiles(ms, n=100)
            p50th = percentiles[48] * 1e-9
            p90th = percentiles[88] * 1e-9
            p99th = percentiles[98] * 1e-9
            avg = sum(ms) / len(ms) * 1e-9
            total = len(ms)
            percent = (sum(ms) / total_time) * 100
            out += (
                f"{name}({itoa3(total)})  "
                f"{ttoa3(p50th)}  {ttoa3(p90th)}  {ttoa3(p99th)}  "
                f"AVG {ttoa3(avg)}  {percent:.1f}%\n"
            )
    return out


def _latency_loop():
    measures: dict[str, list[int]] = {}
    while True:
        m: _LatencyMessage = _queue.get()
        match m:
            case _Record(name=name, elapsed_ns=elapsed):
                measures.setdefault(name, []).append(elapsed)
            case _Report(reply=reply):
                reply.put(_build_report(measures))
            case _:
                assert False, f"Unknown latency message: {m!r}"


# Daemon thread started at module load — it dies with the process and
# owns no resources that need explicit cleanup.
_thread = threading.Thread(
    target=_latency_loop, daemon=True, name='latency')
_thread.start()


# --- public surface ---------------------------------------------------------


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
        _queue.put(_Record(name=self.name, elapsed_ns=end - self.start))


def timer(name):
    return Timer(name)


def get_report() -> str:
    reply: Queue = Queue()
    _queue.put(_Report(reply=reply))
    return reply.get()


def report():
    logging.info(get_report())
