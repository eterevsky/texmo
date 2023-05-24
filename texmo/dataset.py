import argparse
import logging
import mmap
import numpy as np
import os
from queue import Queue
import random
from threading import Thread
from typing import Optional

from . import latency


def _worker(all, request_queue, data_queues, debug):
    while True:
        length, batch_size, ntokens = request_queue.get()
        if length is None or batch_size is None:
            break
        with latency.timer("dataset-worker"):
            batch = []
            for _ in range(batch_size):
                start = random.randrange(len(all) - length)
                sample = all[start : start + length]
                if debug:
                    try:
                        s = sample.decode("utf-8")
                    except UnicodeDecodeError:
                        s = sample
                    print(f"=== Sample:\n{s}\n===")
                batch.append(np.frombuffer(sample, dtype=np.ubyte))
            batch = np.array(batch, dtype=np.ubyte)
            data_queues[(length, batch_size, ntokens)].put(batch)


class DataSet(object):
    def __init__(
        self, path: Optional[str] = None, data: Optional[bytes] = None,
        debug: bool=False
    ):
        self._debug: bool = debug
        self.all: bytes | mmap.mmap = b""
        if path is not None:
            assert data is None

            if os.path.isdir(path):
                raise Exception(
                    "Directory with training data is currently not supported"
                )
            self._file = os.open(path, os.O_RDONLY)
            try:
                self.all = mmap.mmap(
                    self._file,
                    0,
                    flags=mmap.MAP_PRIVATE,
                    prot=mmap.PROT_READ,
                    access=mmap.ACCESS_READ,
                )
            except AttributeError:
                # Windows
                self.all = mmap.mmap(
                    self._file,
                    0,
                    access=mmap.ACCESS_READ,
                )

            total = len(self.all) / 1e9
            logging.info(f"Dataset mmap'ed: {total:.2f} GB")
        else:
            assert data is not None
            self.all = data

        # A queue of (length, batch_size) tuples.
        self._request_queue: Queue = Queue()

        # Queues of training batches, with (length, batch_size) as key. Each
        # queue should have at least one batch ready at any time.
        self._data_queues: dict[tuple[int, int], Queue] = {}

        self._worker_thread = Thread(
            target=_worker,
            args=[self.all, self._request_queue, self._data_queues, debug],
        )
        self._worker_thread.start()

    def __del__(self):
        self.join()
        os.close(self._file)

    def join(self):
        self._request_queue.put((None, None, None))
        self._worker_thread.join()

    def sample(self, bytes_length, batch_size, ntokens=None):
        key = (bytes_length, batch_size, ntokens)
        if key not in self._data_queues:
            self._data_queues[key] = Queue()
            self._request_queue.put(key)

        with latency.timer("DataSet.sample"):
            self._request_queue.put(key)
            return self._data_queues[key].get()

    @property
    def total_size(self):
        return len(self.all)

    def warmup(self):
        _ = self.sample(1024, 1024)


def build_dataset(path):
    print(f"Loading dataset from {path}")
    dataset = DataSet(path=path)
    dataset.warmup()
    return dataset


def build_fake_dataset():
    """Build 'abacabadabacaba...' training set."""
    s = b"aba"
    for c in range(ord("c"), ord("o")):
        s = s + bytes([c]) + s
    return DataSet(data=s)


def dataset_sample(args: argparse.Namespace):
    logging.getLogger().setLevel(logging.INFO)
    dataset = DataSet(args.data, debug=True)
    sample = dataset.sample(args.length, args.batch, args.ntokens)
    print(f"Prepared sample:\n{sample}")


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d", "--data", type=str, help="training data file", required=True
    )
    parser.add_argument(
        "-b", "--batch", type=int, help="batch size", default=1
    )
    parser.add_argument(
        "-t", "--tokens", type=str, help="path to token set definition", required=True
    )
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument(
        "-n", "--ntokens", type=int, help="length of the sample in tokens", default=256
    )
    length_group.add_argument(
        "-l", "--length", type=int, help="length in bytes", default=None
    )
    parser.set_defaults(func=dataset_sample)
