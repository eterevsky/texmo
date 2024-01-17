import sys
import argparse
import logging
import math
import os
import random
import time
from collections import namedtuple
from itertools import repeat
from queue import Queue
from threading import Lock, Thread
from typing import Optional

import numpy as np

from . import latency
from .tokens import Tokenizer, TokenSet, get_tokenizer, set_tokens_dir
from .common import ceil_power2


def file_reader(filename: str, request_queue: Queue, data_queues: dict[int, Queue]):
    size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        while True:
            request = request_queue.get()

            if request is None:
                break

            chunk_size = request
            assert chunk_size in data_queues

            start = random.randrange(size - chunk_size)
            f.seek(start)
            try:
                chunk = f.read(chunk_size)
            except PermissionError as e:
                print("PermissionError", filename, start, chunk_size)
                print(e)
                os._exit(1)
            data_queues[chunk_size].put(chunk)


def mem_reader(data: bytes, request_queue: Queue, data_queues: dict[int, Queue]):
    """A thread that reads random chunks from a bytes object."""
    while True:
        request = request_queue.get()
        if request is None:
            break

        chunk_size = request
        assert chunk_size in data_queues

        with latency.timer("mem_reader"):
            start = random.randrange(len(data) - chunk_size)
            chunk = data[start : start + chunk_size]
            data_queues[chunk_size].put(chunk)


def create_tokens_sample(
    chunk: bytes,
    tokenizer: Tokenizer,
    ntokens: int,
    start_paragraph: bool,
) -> tuple[list[int], float]:
    if start_paragraph:
        start = chunk.find(b"\n\n")
        if start < 0:
            return None
        start += 2  # Skip "\n\n"
    else:
        start = 0
        while start < len(chunk) and chunk[start] > 127:
            start += 1
    end = len(chunk) - 1
    while end > 0 and 128 <= chunk[end] < 192:
        end -= 1
    if start >= end:
        return None, None
    tokens, entropy = tokenizer.tokenize_ids(chunk[start:end], max_tokens=ntokens)
    assert len(tokens) <= ntokens
    if len(tokens) < ntokens:
        return None, None
    return tokens, entropy


def create_bytes_sample(
    chunk: bytes, tokenizer: Tokenizer, length: int, start_paragraph: bool
) -> tuple[list[int], float]:
    if start_paragraph:
        start = chunk.find(b"\n\n")
        if start < 0:
            return None, None
        start += 2  # Skip "\n\n"
    else:
        start = 0
        while start < len(chunk) and chunk[start] > 127:
            start += 1
    end = start + length
    while end < len(chunk) and 128 <= chunk[end] < 192:
        end += 1
    if end > len(chunk):
        return None, None
    return tokenizer.tokenize_ids(chunk[start:end])


def create_tokens_batch(
    reader_request_queue: Queue,
    reader_data_queues: dict[int, Queue],
    tokenizer: Tokenizer,
    ntokens: int,
    batch: int,
) -> (np.array, list[float]):
    bytes_size = 2 ** math.ceil(
        math.log2(ntokens * tokenizer.token_set.bytes_per_token + 16)
    )
    samples = []
    entropies = []
    while len(samples) < batch:
        assert bytes_size is not None
        reader_request_queue.put(bytes_size)
        chunk = reader_data_queues[bytes_size].get()
        sample, entropy = create_tokens_sample(chunk, tokenizer, ntokens, start_paragraph=False)
        if sample is None:
            bytes_size *= 2
            continue
        samples.append(sample)
        entropies.append(entropy)
    return np.array(samples, dtype=np.uint16), entropies


def create_bytes_batch(
    reader_request_queue: Queue,
    reader_data_queues: dict[int, Queue],
    tokenizer: Tokenizer,
    length: int,
    batch: int,
) -> tuple[np.array, list[int], list[float]]:
    lengths = []
    samples = []
    entropies = []
    bytes_size = ceil_power2(length + 16)
    while len(samples) < batch:
        assert bytes_size is not None
        reader_request_queue.put(bytes_size)
        chunk = reader_data_queues[bytes_size].get()
        sample, entropy = create_bytes_sample(chunk, tokenizer, length, start_paragraph=False)
        if sample is None:
            bytes_size *= 2
            assert bytes_size < length * 256
            continue
        samples.append(sample)
        entropies.append(entropy)
        lengths.append(len(sample))

    max_len = max(lengths)

    padded_samples = []
    for sample in samples:
        l = len(sample)
        if l == max_len:
            padded_samples.append(sample)
        elif isinstance(sample, np.ndarray):
            padded_samples.append(np.pad(sample, (0, max_len - l), "constant"))
        else:
            sample.extend(repeat(0, max_len - l))
            padded_samples.append(sample)

    batch = np.array(padded_samples, dtype=np.uint16)
    return batch, lengths, entropies


SampleRequest = namedtuple(
    "SampleRequest", ["token_set_name", "ntokens", "length", "batch"]
)


class SamplerThread(Thread):
    def __init__(
        self,
        reader_request_queue: Queue,
        reader_data_queues: dict[int, Queue],
        requests_queue: Queue,
        data_queues: dict[str, Queue],
        token_sets_lock: Lock,
    ):
        super().__init__()
        self._reader_request_queue: Queue = reader_request_queue
        self._reader_data_queues: dict[int, Queue] = reader_data_queues
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

            if request.ntokens is not None:
                assert request.length is None
                batch, entropies = create_tokens_batch(
                    self._reader_request_queue,
                    self._reader_data_queues,
                    tokenizer,
                    request.ntokens,
                    request.batch,
                )
                response_queue.put((batch, None, entropies))
            else:
                assert request.length is not None
                batch, lengths, entropies = create_bytes_batch(
                    self._reader_request_queue,
                    self._reader_data_queues,
                    tokenizer,
                    request.length,
                    request.batch,
                )
                response_queue.put((batch, lengths, entropies))


class DataSet(object):
    def __init__(
        self,
        path: Optional[str] = None,
        data: Optional[bytes] = None,
        path_processed: Optional[str] = None,
        in_process: bool = False,
        n_sampler_threads: int = 1,
    ):
        in_process = "synch" if in_process else "concurrent"
        logging.info(f"Creating DataSet ({in_process})")
        self._in_process = in_process
        self._init_reader_queues()
        self._init_reader_thread(path, data)
        if not self._in_process:
            self._init_sampler_threads(n_sampler_threads)

    def __del__(self):
        self.join()

    def _init_reader_queues(self):
        self._reader_requests_queue: Queue = Queue()
        self._reader_data_queues: dict[int, Queue] = {}
        for i in range(0, 32):
            self._reader_data_queues[2**i] = Queue()

    def _init_reader_thread(self, path: Optional[str], data: Optional[bytes]):
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

                self._reader_thread = Thread(
                    target=mem_reader,
                    args=(
                        data,
                        self._reader_requests_queue,
                        self._reader_data_queues,
                    ),
                )
            else:
                self._reader_thread = Thread(
                    target=file_reader,
                    args=(
                        path,
                        self._reader_requests_queue,
                        self._reader_data_queues,
                    ),
                )

        else:
            assert data is not None
            self.total_size = len(data)
            self._reader_thread = Thread(
                target=mem_reader,
                args=(
                    data,
                    self._reader_requests_queue,
                    self._reader_data_queues,
                ),
            )
        self._reader_thread.start()

    def _init_sampler_threads(self, n_sampler_threads: int):
        # A queue of (length, batch_size) tuples.
        self._request_queue: Queue[SampleRequest] = Queue()

        # Queues of training batches, with (length, batch_size) as key. Each
        # queue should have at least one batch ready at any time.
        self._data_queues: dict[SampleRequest, Queue] = {}

        self._token_sets_lock = Lock()

        self._sampler_threads = []

        for _ in range(n_sampler_threads):
            thread = SamplerThread(
                self._reader_requests_queue,
                self._reader_data_queues,
                self._request_queue,
                self._data_queues,
                self._token_sets_lock,
            )
            thread.start()
            self._sampler_threads.append(thread)

    def join(self):
        if not self._in_process:
            for _ in self._sampler_threads:
                self._request_queue.put(None)
            for th in self._sampler_threads:
                th.join()

        self._reader_requests_queue.put(None)
        self._reader_thread.join()

    def _send_request(self, request: SampleRequest):
        first_request = request not in self._data_queues
        if first_request:
            with self._token_sets_lock:
                self._data_queues[request] = Queue()
                get_tokenizer(request.token_set_name)
            # Warmup
            self._request_queue.put(request)
            self._request_queue.put(request)

        self._request_queue.put(request)
        (batch, lengths) = self._data_queues[request].get()
        return (batch, lengths)

    def sample_bytes(
        self, length: int, batch_size: int, token_set_name: str
    ) -> tuple[np.ndarray, list, list[float]]:
        with latency.timer("DataSet.sample_bytes"):
            if self._in_process:
                tokenizer = get_tokenizer(token_set_name)
                return create_bytes_batch(
                    self._reader_requests_queue,
                    self._reader_data_queues,
                    tokenizer,
                    length,
                    batch_size,
                )
            else:
                request = SampleRequest(
                    token_set_name, ntokens=None, length=length, batch=batch_size
                )
                batch, lengths, entropies = self._send_request(request)
                return batch, lengths, entropies

    def sample_tokens(
        self, ntokens, batch_size, token_set_name
    ) -> tuple[np.ndarray, list[float]]:
        with latency.timer("DataSet.sample_tokens"):
            if self._in_process:
                tokenizer = get_tokenizer(token_set_name)
                return create_tokens_batch(
                    self._reader_requests_queue,
                    self._reader_data_queues,
                    tokenizer,
                    ntokens,
                    batch_size,
                )
            else:
                request = SampleRequest(
                    token_set_name, ntokens=ntokens, length=None, batch=batch_size
                )
                batch, _lengths, entropies = self._send_request(request)
                return batch, entropies


def build_dataset(path):
    logging.info(f"Loading dataset from {path}")
    dataset = DataSet(path=path)
    dataset.warmup()
    return dataset


def build_fake_dataset():
    """Build 'abacabadabacaba...' training set."""
    s = b"aba"
    for c in range(ord("c"), ord("n")):
        s = s + bytes([c]) + s

    s = b"\n\n".join([s] * 16)
    return DataSet(data=s)


def sample(args: argparse.Namespace):
    set_tokens_dir(args.tokens_dir)

    dataset = DataSet(
        args.data,
        in_process=not args.benchmark,
        path_processed=args.data_processed,
    )

    if args.benchmark:
        benchmark(dataset, args.length, args.ntokens, args.batch, args.tokens)
    else:
        if args.length:
            sample, lengths = dataset.sample_bytes(args.length, args.batch, args.tokens)
            print(f"Prepared sample:\n{sample}")
            print(f"Lengths: {lengths}")
        else:
            sample = dataset.sample_tokens(args.ntokens, args.batch, args.tokens)
            print(f"Prepared sample:\n{sample}")
            token_set = get_tokenizer(args.tokens).token_set

            sep = "" if token_set.type == "chars" else "|"
            for token in sample[0]:
                print(token_set.tokens[token], end=sep)
            print()
        dataset.join()


def sample_init_args(parser: argparse.ArgumentParser, config):
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="training data file",
        default=config.DATA,
    )
    parser.add_argument(
        "--data-processed",
        type=str,
        help="training data that has been already processed with capswords filter",
        default=config.DATA_CAPS_WORDS,
    )
    parser.add_argument(
        "--tokens-dir",
        type=str,
        default=config.TOKENS_DIR,
        help="directory with token sets",
    )
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=1)
    parser.add_argument(
        "-t",
        "--tokens",
        type=str,
        help="name of the token set",
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
