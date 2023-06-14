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
from .timing import Timing
from .tokens import Tokenizer, TokenSet, get_tokenizer, set_tokens_dir


def file_reader(
    filename: str, request_queue: Queue, data_queue: Queue, chunk_size: int
):
    size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        while True:
            request = request_queue.get()

            if request is None:
                break

            with latency.timer("file_reader"):
                start = random.randrange(size - chunk_size)
                f.seek(start)
                chunk = f.read(chunk_size)
                data_queue.put(chunk)


def mem_reader(
    data: bytes, request_queue: Queue, data_queue: Queue, chunk_size: int
):
    while True:
        request = request_queue.get()

        if request is None:
            break

        with latency.timer("mem_reader"):
            start = random.randrange(len(data) - chunk_size)
            chunk = data[start : start + chunk_size]
            data_queue.put(chunk)


def create_tokens_sample(
    chunk: bytes, tokenizer: Tokenizer, ntokens: int
) -> list[int]:
    start = chunk.find(b"\n\n")
    if start < 0:
        return None
    end = len(chunk)
    while end > 0 and 128 <= chunk[end - 1] < 192:
        end -= 1
    if start >= end:
        return None
    tokens = tokenizer.tokenize_ids(chunk[start:end], max_tokens=ntokens)
    assert len(tokens) <= ntokens
    if len(tokens) < ntokens:
        return None
    return tokens


def create_bytes_sample(
    chunk: bytes, tokenizer: Tokenizer, length: int
) -> bytes:
    start = chunk.find(b"\n\n")
    if start < 0:
        return None
    end = start + length
    while end < len(chunk) and 128 <= chunk[end] < 192:
        end += 1
    assert end <= len(chunk)
    return tokenizer.tokenize_ids(chunk[start:end])


SampleRequest = namedtuple(
    "SampleRequest", ["token_set_name", "ntokens", "length", "batch"]
)


class SamplerThread(Thread):
    def __init__(
        self,
        reader_request_queue: Queue,
        reader_data_queue: Queue,
        requests_queue: Queue,
        data_queues: dict[str, Queue],
        token_sets_lock: Lock,
    ):
        super().__init__()
        self._reader_request_queue: Queue = reader_request_queue
        self._reader_data_queue: Queue = reader_data_queue
        self._requests_queue: Queue = requests_queue
        self._data_queues: dict[str, Queue] = data_queues
        self._token_sets_lock = token_sets_lock

    def run(self):
        while True:
            request = self._requests_queue.get()
            if request is None:
                break

            self._execute_request(request)

    def _execute_request(self, request: SampleRequest):
        with latency.timer("SamplerThread._execute_request") as timer:
            with self._token_sets_lock:
                tokenizer = get_tokenizer(request.token_set_name)
                response_queue = self._data_queues[request]

            assert (
                request.length is not None
                and request.ntokens is None
                or request.length is None
                and request.ntokens is not None
            )

            samples = []
            lengths = None if request.ntokens is not None else []

            while len(samples) < request.batch:
                self._reader_request_queue.put(True)
                with latency.timer(
                    "SamplerThread._execute_request.wait_for_data"
                ):
                    chunk = self._reader_data_queue.get()

                if request.ntokens is not None:
                    sample = create_tokens_sample(
                        chunk, tokenizer, request.ntokens
                    )
                else:
                    sample = create_bytes_sample(
                        chunk, tokenizer, request.length
                    )

                if sample is not None:
                    samples.append(sample)
                    if lengths is not None:
                        lengths.append(len(sample))

            if request.length is not None:
                max_len = max(len(sample) for sample in samples)
                for sample in samples:
                    l = len(sample)
                    sample.extend(repeat(0, max_len - l))

            batch = np.array(samples, dtype=np.uint16)
            response_queue.put((batch, lengths, timer.value()))


class DataSet(object):
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        tokens_dir: str = None,
        debug: bool = False,
        in_process: bool = False,
        timing: Timing = None,
    ):
        set_tokens_dir(tokens_dir)

        logging.info("Creating DataSet")
        self._debug: bool = debug
        self._timing = timing

        self._reader_requests_queue: Queue = Queue()
        self._reader_queue: Queue = Queue()

        self._reader_threads: list[Thread] = []

        if path is not None:
            assert data is None
            assert not os.path.isdir(path)

            self.total_size = os.path.getsize(path)
            if self.total_size < 2**30:
                logging.info("Reading training set into memory")
                with open(path, "rb") as f:
                    data = f.read()
                assert len(data) == self.total_size
                logging.info(f"Read {self.total_size} bytes")

                self._reader_threads.append(Thread(
                    target=mem_reader,
                    args=(
                        data,
                        self._reader_requests_queue,
                        self._reader_queue,
                        16384,
                    ),
                ))
            else:
                for i in range(4):
                    self._reader_threads.append(Thread(
                        target=file_reader,
                        args=(
                            path,
                            self._reader_requests_queue,
                            self._reader_queue,
                            16384,
                        ),
                    ))
        else:
            assert data is not None
            self.total_size = len(data)
            self._reader_threads.append(Thread(
                target=mem_reader,
                args=(
                    data,
                    self._reader_requests_queue,
                    self._reader_queue,
                    16384,
                ),
            ))
        for th in self._reader_threads:
            th.start()

        # Warmup
        self._reader_requests_queue.put(True)
        self._reader_requests_queue.put(True)
        self._reader_requests_queue.put(True)
        self._reader_requests_queue.put(True)

        # A queue of (length, batch_size) tuples.
        self._request_queue: Queue[SampleRequest] = Queue()

        # Queues of training batches, with (length, batch_size) as key. Each
        # queue should have at least one batch ready at any time.
        self._data_queues: dict[SampleRequest, Queue] = {}

        self._token_sets_lock = Lock()

        self._in_process = in_process
        self._worker_threads = []
        for _ in range(1):
            worker_thread = SamplerThread(
                self._reader_requests_queue,
                self._reader_queue,
                self._request_queue,
                self._data_queues,
                self._token_sets_lock,
            )
            if not in_process:
                worker_thread.start()
            self._worker_threads.append(worker_thread)

    def __del__(self):
        if not self._in_process:
            self.join()

    def join(self):
        for th in self._worker_threads:
            self._request_queue.put(None)
        for th in self._worker_threads:
            th.join()
        for th in self._reader_threads:
            self._reader_requests_queue.put(None)
        for th in self._reader_threads:
            th.join()

    def _execute_request(self, request: SampleRequest):
        first_request = request not in self._data_queues
        if first_request:
            with self._token_sets_lock:
                self._data_queues[request] = Queue()
            if not self._in_process:
                # Warmup
                self._request_queue.put(request)
                self._request_queue.put(request)

        self._request_queue.put(request)
        if self._in_process:
            self._request_queue.put(None)
            self._worker_threads[0].run()
        (batch, lengths, timer) = self._data_queues[request].get()
        if (
            self._timing is not None
            and not first_request
            and request.ntokens is not None
        ):
            self._timing.register_sample_latency(
                request.token_set_name,
                request.ntokens,
                request.batch,
                timer / 1e9,
            )
        return (batch, lengths)

    def sample_bytes(
        self, length, batch_size, token_set_name
    ) -> tuple[np.ndarray, list]:
        request = SampleRequest(token_set_name, ntokens=None, length=length, batch=batch_size)

        with latency.timer("DataSet.sample_bytes"):
            return self._execute_request(request)

    def sample_tokens(self, ntokens, batch_size, token_set_name) -> np.ndarray:
        request = SampleRequest(token_set_name, ntokens=ntokens, length=None, batch=batch_size)

        with latency.timer("DataSet.sample_tokens"):
            batch, _lengths = self._execute_request(request)
            return batch


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


def sample_init_args(parser: argparse.ArgumentParser):
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


def benchmark(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    dataset = DataSet(
        path=args.data,
        tokens_dir=args.tokens_dir,
        in_process=args.in_process,
    )

    ntokens = args.sample_len
    batch = args.batch
    token_set = args.token_set

    samples = 0
    total_tokens = 0
    start = time.time()
    while time.time() - start < 10:
        sample = dataset.sample_tokens(ntokens, batch, token_set)
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


def benchmark_init_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="a file with training data",
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default="tokens",
        help="directory with token sets",
    )
    parser.add_argument(
        "--token-set",
        default="tokens256_raw_all",
        type=str,
        help="token set name",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=256,
        help='range of acceptable batch sizes, for example "1-256" (default: unrestricted)',
    )
    parser.add_argument(
        "--sample-len",
        type=int,
        default=256,
        help="range of acceptable sample lens (default: 128)",
    )
    parser.add_argument("--in-process", default=False, action="store_true")
    parser.add_argument(
        "--in-memory",
        default=False,
        action="store_true",
        help="Read up to 32G into memory and serve from there",
    )

    parser.set_defaults(func=benchmark)
