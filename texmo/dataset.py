import argparse
from collections import namedtuple
from itertools import repeat
import logging
import mmap
import numpy as np
import os
from queue import Queue
import random
from threading import Thread
from typing import Optional

from . import latency
from .tokens import TokenSet, Tokenizer


Request = namedtuple(
    "Request", ["length", "batch", "ntokens", "token_set_size"]
)


def _worker(data, request_queue, data_queues, tokenizers, debug):
    while True:
        request = request_queue.get()
        logging.info(f"Request: {request}")
        if request is None:
            logging.info("Stopping the worker")
            break
        with latency.timer("dataset-worker"):
            batch = []
            samples = []
            for _ in range(request.batch):
                if request.length is not None:
                    length_est = request.length
                else:
                    length_est = request.ntokens * 5
                start = random.randrange(len(data) - length_est)
                paragraph = data.rfind(b"\n\n", 0, start)
                doc_end = data.find(b"\x13", paragraph, start)
                if doc_end >= 0:
                    start = doc_end + 1
                elif paragraph >= 0:
                    start = paragraph + 2
                else:
                    start = 0
                if request.token_set_size:
                    if debug:
                        end = start + 256
                        while end < len(data) and 128 <= data[end] < 192:
                            end += 1
                        sample = data[start : end]
                        try:
                            s = sample.decode("utf-8")
                        except UnicodeDecodeError:
                            s = sample
                        print(f"=== Sample:\n{s}\n...\n===")

                    tokens = tokenizers[request.token_set_size].tokenize(data, start, request.ntokens, request.length)

                    if debug:
                        debug_str = "|".join(str(t) for t in tokens)
                        print(debug_str + "\n===")

                    samples.append([token.id for token in tokens])
                else:
                    sample = data[start : start + request.length]
                    if debug:
                        try:
                            s = sample.decode("utf-8")
                        except UnicodeDecodeError:
                            s = sample
                        print(f"=== Sample:\n{s}\n===")
                    samples.append(sample)

            if request.ntokens:
                assert all(len(sample) == request.ntokens for sample in samples)
                batch = np.array(samples, dtype=np.uint16)
                data_queues[request].put(batch)
            else:
                lengths = []
                max_len = max(len(sample) for sample in samples)
                for sample in samples:
                    l = len(sample)
                    lengths.append(l)
                    sample.extend(repeat(0, max_len - l))
                batch = np.array(samples, dtype=np.uint16)
                data_queues[request].put((batch, lengths))


class DataSet(object):
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        token_sets: dict[int, TokenSet] = None,
        debug: bool = False,
        in_process: bool = False,
    ):
        logging.info("Createing DataSet")
        self._debug: bool = debug
        self._token_sets = token_sets
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
            self._file = None
            self.all = data

        # A queue of (length, batch_size) tuples.
        self._request_queue: Queue[Request] = Queue()

        # Queues of training batches, with (length, batch_size) as key. Each
        # queue should have at least one batch ready at any time.
        self._data_queues: dict[tuple[int, int], Queue] = {}

        self._tokenizers = {}
        if token_sets is not None:
            for token_set_size, token_set in token_sets.items():
                self._tokenizers[token_set_size] = Tokenizer(token_set)

        self._in_process = in_process
        if not in_process:
            self._worker_thread = Thread(
                target=_worker,
                args=[
                    self.all,
                    self._request_queue,
                    self._data_queues,
                    self._tokenizers,
                    debug,
                ],
            )
            self._worker_thread.start()

    def __del__(self):
        if not self._in_process:
            self.join()
        if self._file is not None:
            os.close(self._file)

    def join(self):
        self._request_queue.put(None)
        self._worker_thread.join()

    def _execute_request(self, request: Request):
        if request not in self._data_queues:
            self._data_queues[request] = Queue()
            if not self._in_process:
                self._request_queue.put(request)

        self._request_queue.put(request)
        if self._in_process:
            self._request_queue.put(None)
            _worker(self.all, self._request_queue, self._data_queues, self._tokenizers, self._debug)
        result = self._data_queues[request].get()
        return result

    def sample_bytes(
        self, length, batch_size, token_set_size=None
    ):
        request = Request(length, batch_size, None, token_set_size)

        with latency.timer("DataSet.sample_bytes"):
            return self._execute_request(request)

    def sample_tokens(
        self, ntokens, batch_size, token_set_size=None
    ):
        request = Request(None, batch_size, ntokens, token_set_size)

        with latency.timer("DataSet.sample_tokens"):
            return self._execute_request(request)

    def sample(self, length, batch_size):
        batch, _ = self.sample(length, batch_size, None)
        return batch

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


def sample(args: argparse.Namespace):
    logging.getLogger().setLevel(logging.INFO)

    if args.tokens is None:
        token_set = None
        token_sets = {}
        token_set_size = None
    else:
        token_set = TokenSet.from_json_file(args.tokens)
        token_sets = {token_set.ntokens: token_set}
        token_set_size = token_set.ntokens

    dataset = DataSet(
        args.data, debug=True, in_process=True, token_sets=token_sets
    )

    if args.length:
        batch, lengths = dataset.sample_bytes(args.length, args.batch, token_set_size)
        print(f"Prepared sample:\n{batch}")
        print(f"Lengths: {lengths}")
    else:
        batch = dataset.sample_tokens(args.ntokens, args.batch, token_set_size)
        print(f"Prepared sample:\n{batch}")


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d", "--data", type=str, help="training data file", required=True
    )
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=1)
    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        help="path to token set definition",
        default=None,
    )
    length_group = parser.add_mutually_exclusive_group(required=True)
    length_group.add_argument(
        "-n",
        "--ntokens",
        type=int,
        help="length of the sample in tokens",
        default=None,
    )
    length_group.add_argument(
        "-l", "--length", type=int, help="length in bytes", default=None
    )
    parser.set_defaults(func=sample)
