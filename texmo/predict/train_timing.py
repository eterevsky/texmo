from collections import namedtuple
import json
import os
from statistics import mean


RunTiming = namedtuple(
    "ConfTiming",
    [
        "spec",
        "ntokens",
        "sample_len",
        "batch",
        "first_step",
        "avg_step",
    ]
)


class TrainTiming(object):
    def __init__(self, jsonl_path=None):
        self._conf_timings = []
        if jsonl_path is not None:
            try:
                with open(jsonl_path, "r") as f:
                    for line in f:
                        json.loads(line)
            except FileNotFoundError:
                pass
            self._file = open(jsonl_path, "a", encoding="utf-8", newline="\n")
        else:
            self._file = None

    def register_step_latency(self, conf, step_times):
        if step_times:
            first_step = step_times[0]
        else:
            first_step = None

        if len(step_times) >= 2:
            avg_step = mean(step_times[1:])
        else:
            avg_step = None

        record = RunTiming(spec=str(conf.model), ntokens=conf.ntokens, sample_len=conf.sample_len, batch=conf.batch, first_step=first_step, avg_step=avg_step)

        self._conf_timings.append(record)
        if self._file is not None:
            print(json.dumps(record._asdict()), file=self._file)