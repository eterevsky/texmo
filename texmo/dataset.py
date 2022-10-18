import mmap
import numpy as np
import os
from queue import Queue
import random
from threading import Thread

from . import latency


def _worker(all, request_queue, data_queues):
    while True:
        length, batch_size = request_queue.get()
        if length is None or batch_size is None:
            break
        with latency.timer(f"dataset-worker"):
            batch = []
            for _ in range(batch_size):
                start = random.randrange(len(all) - length)
                sample = all[start : start + length]
                batch.append(np.frombuffer(sample, dtype=np.ubyte))
            batch = np.array(batch, dtype=np.ubyte)
            data_queues[(length, batch_size)].put(batch)


class DataSet(object):
    def __init__(self, path):
        if os.path.isdir(path):
            raise Exception(
                "Directory with training data is currently not supported"
            )
        self.file = open(path, "rb")
        self.all = mmap.mmap(
            self.file.fileno(),
            0,
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ,
        )

        total = len(self.all) / 1e9
        print(f"Dataset mmap'ed: {total:.2f} GB")

        # A queue of (length, batch_size) tuples.
        self._request_queue = Queue()

        # Queues of training batches, with (length, batch_size) as key. Each
        # queue should have at least one batch ready at any time.
        self._data_queues = {}

        self._worker_thread = Thread(
            target=_worker,
            args=[self.all, self._request_queue, self._data_queues],
        )
        self._worker_thread.start()

    def __del__(self):
        self.join()

    def join(self):
        self._request_queue.put((None, None))
        self._worker_thread.join()

    def sample(self, length, batch_size):
        key = (length, batch_size)
        if key not in self._data_queues:
            self._data_queues[key] = Queue()
            self._request_queue.put(key)

        with latency.timer(f"dataset-sample"):
            self._request_queue.put(key)
            return self._data_queues[key].get()

    @property
    def total_size(self):
        return len(self.all)

    def warmup(self):
        _ = self.sample(1024, 1024)


def build_dataset(data):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()
    return dataset
