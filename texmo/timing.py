import logging
from statistics import mean

from .configuration import Configuration, conf_tokens_name


class Timing(object):
    def __init__(self):
        self._sample_latency = {}
        self._confs = []
        self._first_step_latency = []
        self._step_latencies = []
        self._steps = []
        self._total_latency = []

    def register_sample_latency(
        self, token_set_name: str, sample_len: int, batch: int, latency_s: float
    ):
        key = (token_set_name, sample_len, batch)

        l = self._sample_latency.get(key)
        if l is None:
            l = []
            self._sample_latency[key] = l

        l.append(latency_s)

    def generate_timing_key(self, conf: Configuration):
        key = len(self._confs)
        self._confs.append(conf)
        self._first_step_latency.append(None)
        self._step_latencies.append([])
        self._steps.append(None)
        self._total_latency.append(None)
        return key

    def register_step(self, key, first: bool, latency_s: float):
        if first:
            self._first_step_latency[key] = latency_s
        else:
            self._step_latencies[key].append(latency_s)

    def register_training_time(self, key, steps: int, latency_s: float):
        self._steps[key] = steps
        self._total_latency[key] = latency_s

    def fit(self):


    def report(self):
        for key in sorted(self._sample_latency.keys()):
            avg = mean(self._sample_latency[key]) * 1000
            n = len(self._sample_latency[key])
            print(f"{key}  {avg:.3f} ms ({n})")
