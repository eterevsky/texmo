import argparse
import logging
import mmap
import os
import random
import time
from collections import namedtuple
from itertools import repeat
from queue import Queue
from threading import Thread, Lock
from typing import Optional

import numpy as np

from . import latency
from .tokens import Tokenizer, TokenSet, get_tokenizer


Request = namedtuple(
    "Request", ["length", "batch", "ntokens", "token_set_name"]
)


class SamplerThread(Thread):
    def __init__(
        self,
        data: bytes | mmap.mmap,
        requests_queue: Queue,
        data_queues: dict[str, Queue],
        tokenizers: dict[str, Tokenizer],
        token_sets_lock: Lock,
        debug: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._data: bytes | mmap.mmap = data
        self._requests_queue: Queue = requests_queue
        self._token_sets_lock = token_sets_lock
        self._data_queues: dict[str, Queue] = data_queues
        self._tokenizers: dict[str, Tokenizer] = tokenizers
        self._debug: bool = debug

    def run(self):
        while True:
            request = self._requests_queue.get()

            if request is None:
                logging.info("Stopping SamplerThread")
                break

            self._execute_request(request)

    def _find_start(self, length: Optional[int], ntokens: Optional[int]) -> int:
        if length is not None:
            length_est = length
        elif ntokens is not None:
            length_est = ntokens * 5
        start = random.randrange(len(self._data) - length_est)
        paragraph = self._data.rfind(b"\n\n", 0, start)
        doc_end = self._data.find(b"\x13", paragraph, start)
        if doc_end >= 0:
            return doc_end + 1
        elif paragraph >= 0:
            return paragraph + 2
        else:
            return 0

    def _get_sample(
        self,
        length: Optional[int],
        ntokens: Optional[int],
        tokenizer: Optional[Tokenizer],
    ) -> list[int]:
        start = self._find_start(length, ntokens)
        if tokenizer is not None:
            if self._debug:
                end = start + 256
                while end < len(self._data) and 128 <= self._data[end] < 192:
                    end += 1
                sample = self._data[start:end]
                try:
                    s: str | bytes = sample.decode("utf-8")
                except UnicodeDecodeError:
                    s = sample
                print(f"=== Sample:\n{s!r}\n...\n===")

            tokens = tokenizer.tokenize(self._data, start, ntokens, length)

            if self._debug:
                debug_str = "|".join(str(t) for t in tokens)
                print(debug_str + "\n===")

            return [token.id for token in tokens]
        else:
            if length is None:
                length = ntokens
            sample = list(self._data[start : start + length])
            if self._debug:
                try:
                    s = sample.decode("utf-8")
                except UnicodeDecodeError:
                    s = sample
                print(f"=== Sample:\n{s}\n===")
            return sample

    def _execute_request(self, request: Request):
        with latency.timer("SamplerThread._execute_request"):
            with self._token_sets_lock:
                tokenizer = (
                    None
                    if request.token_set_name is None
                    else self._tokenizers[request.token_set_name]
                )
                response_queue = self._data_queues[request]

            samples = [
                self._get_sample(request.length, request.ntokens, tokenizer)
                for _ in range(request.batch)
            ]

            if request.ntokens is not None:
                lengths = None
            else:
                lengths = []
                max_len = max(len(sample) for sample in samples)
                for sample in samples:
                    l = len(sample)
                    lengths.append(l)
                    sample.extend(repeat(0, max_len - l))

            batch = np.array(samples, dtype=np.uint16)
            response_queue.put((batch, lengths))


class DataSet(object):
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        token_sets: Optional[dict[str, TokenSet]] = None,
        tokens_dir: str = None,
        debug: bool = False,
        in_process: bool = False,
    ):
        logging.info("Creating DataSet")
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

        self._tokens_dir = tokens_dir
        self._tokenizers = {}
        if token_sets is not None:
            for token_set_name, token_set in token_sets.items():
                self._tokenizers[token_set_name] = Tokenizer(token_set)

        self._token_sets_lock = Lock()

        self._in_process = in_process
        if not in_process:
            self._worker_threads = []
            for _ in range(1):
                worker_thread = SamplerThread(
                    self.all,
                    self._request_queue,
                    self._data_queues,
                    self._tokenizers,
                    self._token_sets_lock,
                    self._debug,
                )
                worker_thread.start()
                self._worker_threads.append(worker_thread)

    def __del__(self):
        if not self._in_process:
            self.join()
        if self._file is not None:
            os.close(self._file)

    def join(self):
        for th in self._worker_threads:
            self._request_queue.put(None)
        for th in self._worker_threads:
            th.join()

    def _execute_request(self, request: Request):
        if request.ntokens is not None:
            # assert request.token_set_name is not None
            assert request.length is None
        else:
            assert request.length is not None
        if request not in self._data_queues:
            with self._token_sets_lock:
                if request.token_set_name is not None and request.token_set_name not in self._tokenizers:
                    tokenizer = get_tokenizer(request.token_set_name)
                    assert tokenizer is not None
                    self._tokenizers[request.token_set_name] = tokenizer
                self._data_queues[request] = Queue()
            if not self._in_process:
                # 1 warmup request per queue
                self._request_queue.put(request)
                self._request_queue.put(request)

        self._request_queue.put(request)
        if self._in_process:
            self._request_queue.put(None)
            worker = SamplerThread(self.all,
                    self._request_queue,
                    self._data_queues,
                    self._tokenizers,
                    self._token_sets_lock,
                    self._debug)
            worker.run()
        result = self._data_queues[request].get()
        return result

    def sample_bytes(self, length, batch_size, token_set_name=None) -> tuple[np.ndarray, list]:
        request = Request(length, batch_size, None, token_set_name)

        with latency.timer("DataSet.sample_bytes"):
            return self._execute_request(request)

    def sample_tokens(self, ntokens, batch_size, token_set_name) -> np.ndarray:
        request = Request(None, batch_size, ntokens, token_set_name)

        with latency.timer("DataSet.sample_tokens"):
            batch, _lengths = self._execute_request(request)
            return batch

    def sample(self, length, batch_size):
        batch, _ = self.sample_bytes(length, batch_size, None)
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


def benchmark(dataset, length, ntokens, batch, token_set_size):
    samples = 0
    total_tokens = 0

    start = time.time()
    while time.time() - start < 10:
        if length:
            sample, lengths = dataset.sample_bytes(
                length, batch, token_set_size
            )
            total_tokens += sum(lengths)
        else:
            sample = dataset.sample_tokens(ntokens, batch, token_set_size)
            total_tokens += ntokens * batch
        samples += 1
        if samples % 100 == 0:
            print(".", end="", flush=True)

    finish = time.time()
    print()

    samples_per_sec = samples / (finish - start)
    tokens_per_sec = total_tokens / (finish - start)
    print(f"{samples_per_sec:.1f} samples/s")
    print(f"{tokens_per_sec:.1f} tokens/s")


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
        args.data,
        debug=not args.benchmark,
        in_process=not args.benchmark,
        token_sets=token_sets,
    )

    if args.benchmark:
        benchmark(
            dataset, args.length, args.ntokens, args.batch, token_set_size
        )
    else:
        if args.length:
            sample, lengths = dataset.sample_bytes(
                args.length, args.batch, token_set_size
            )
            print(f"Prepared sample:\n{sample}")
            print(f"Lengths: {lengths}")
        else:
            sample = dataset.sample_tokens(
                args.ntokens, args.batch, token_set_size
            )
            print(f"Prepared sample:\n{sample}")


def init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d", "--data", type=str, help="training data file", required=True
    )
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=1)
    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        help="path to the token set definition",
        default=None,
    )
    parser.add_argument(
        "--benchmark",
        default=False,
        action="store_true",
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
