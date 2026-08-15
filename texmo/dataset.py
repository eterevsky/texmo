import itertools
import logging
import mmap
import os
import random
from queue import Queue
from threading import Lock, Thread

import numpy as np

from .common import itoa3
from .latency import timer
from .tokens import get_tokenizer
from .tokens.processing import apply_processing

# os.pread (read at an explicit offset, no shared file cursor) is
# Unix-only; on Windows we fall back to lseek+read (serialized by
# _read_lock). pread also releases the GIL and is safe to call
# concurrently on a shared fd, which is what lets multiple
# DataSetWrapper worker threads overlap their I/O waits.
_HAS_PREAD = hasattr(os, "pread")


def _open_mmap(path: str) -> tuple["file", mmap.mmap]:
    file = open(path, "rb")
    return (file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ))


class DataSet(object):
    def __init__(
        self,
        path: str | None = None,
        data: bytes | mmap.mmap | None = None,
        parallel_chunks: bool = True,
        read_mode: str = "mmap",
    ):
        # read_mode selects how sample_tokens fetches a chunk:
        #   "mmap"  -- slice the memory-mapped file (default)
        #   "pread" -- read at an explicit offset via os.pread
        # Temporary knob so benchmark-dataset can A/B the two. mmap's
        # per-fault readahead reads ~128 KB for every random 400-byte
        # sample (~16x amplification); pread reads only what's asked.
        # pread is also the path that parallelizes across
        # DataSetWrapper workers (see _read / DataSetWrapper).
        self.read_mode = read_mode

        # Serializes the Windows lseek+read fallback when sampling runs
        # on multiple DataSetWrapper threads. os.pread is atomic and
        # needs no lock, so this is uncontended on Unix.
        self._read_lock = Lock()

        if data is not None:
            assert path is None
            logging.info(f"Creating DataSet from data with len = {itoa3(len(data))}")
            self.data_file = None
            self.data = data
        else:
            assert path is not None
            self.data_file, self.data = _open_mmap(path)
            logging.info(f"Creating DataSet from {path} ({itoa3(len(self.data))})")

        # File descriptor for the pread path; None when the buffer is
        # an in-memory bytes object (pread then falls back to slicing).
        self.data_fd = self.data_file.fileno() if self.data_file is not None else None

        self.data_size = len(self.data)

    def __del__(self):
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()

    def _select_chunk_start(self) -> int:
        """A random start offset into the raw buffer. Processing is
        always applied to the chunk on the fly (the DataSetWrapper
        thread pool absorbs the cost), so sampling never consults a
        preprocessed copy."""
        return random.randint(0, self.data_size)

    def _read(
        self, data: bytes | mmap.mmap, fd: int | None, start: int, size: int
    ) -> bytes:
        """Fetch `size` bytes at `start`, honoring self.read_mode."""
        if self.read_mode == "pread" and fd is not None:
            if _HAS_PREAD:
                return os.pread(fd, size, start)
            # Windows fallback: lseek+read isn't atomic, so serialize
            # it for the multi-worker case (no-op cost single-threaded).
            with self._read_lock:
                os.lseek(fd, start, os.SEEK_SET)
                return os.read(fd, size)
        return data[start : start + size]

    def sample_tokens(self, ntokens: int, batch: int, tokenset_name: str) -> np.ndarray:
        """Generate a random sample of input data.

        Returns:
            An integer array with the shape (batch, ntokens).
        """
        tokenizer = get_tokenizer(tokenset_name)
        tokenset = tokenizer.tokenset
        assert tokenset.processing in (
            "raw", "capswords", "capswords2", "gemma")

        samples = []
        while len(samples) < batch:
            start = self._select_chunk_start()
            # "gemma" is a vocabulary convention, not a byte transform:
            # nothing is applied to the chunk.
            need_processing = tokenset.processing in ("capswords", "capswords2")

            if need_processing:
                size = max(round(ntokens * tokenset.avg_bytes_per_token * 1.2), 16)
            else:
                size = max(round(ntokens * tokenset.avg_proc_bytes_per_token * 1.2), 16)

            assert size <= self.data_size

            while True:
                if start + size > self.data_size:
                    break

                chunk = self._read(self.data, self.data_fd, start, size)
                valid_start = 0
                while 128 <= chunk[valid_start] < 192:
                    valid_start += 1
                valid_end = len(chunk) - 1
                while 128 <= chunk[valid_end] < 192:
                    valid_end -= 1
                chunk = chunk[valid_start : valid_end]

                if need_processing:
                    chunk = apply_processing(tokenset.processing, chunk)

                sample = tokenizer.tokenize_processed(chunk)
                if len(sample) >= ntokens:
                    break
                size *= 2

            if start + size > self.data_size:
                continue
            samples.append(sample[:ntokens])

        return np.array(samples)

    def sample_bytes(
        self, nbytes: int, batch: int, tokenset_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random samples of fixed length in bytes.

        Since the samples might have different length due to tokenization, the
        maximal length is used as the second dimension of the array.
        The remainders of the samples are filled with 0s. The actual lengths
        are returned in the second array.

        Returns:
            (An array with the shape (batch, *) of np.int32 with tokens,
             an array with the shape (batch,) with lengths of significant
             portions of each sample)
        """
        tokenizer = get_tokenizer(tokenset_name)
        tokenset = tokenizer.tokenset
        assert tokenset.processing in (
            "raw", "capswords", "capswords2", "gemma")

        samples = []
        while len(samples) < batch:
            start = random.randint(0, self.data_size - nbytes)
            while start < self.data_size and 128 <= self.data[start] < 192:
                start += 1
            end = min(start + nbytes, self.data_size)
            while end < self.data_size and 128 <= self.data[end] < 192:
                end += 1

            if end - start < nbytes:
                continue

            # Always sampled from the raw buffer, so the tokenset's own
            # pipeline applies (identity for "raw"/"gemma").
            chunk = apply_processing(
                tokenset.processing, self.data[start : end])

            sample = tokenizer.tokenize_processed(chunk)
            samples.append(sample)

        lengths = list(map(len, samples))

        max_len = max(lengths)

        for (i, sample) in enumerate(samples):
            l = len(sample)
            if max_len > l:
                if type(sample) is list:
                    sample.extend(itertools.repeat(0, max_len - l))
                else:
                    samples[i] = np.pad(sample, (0, max_len - l))

        if type(samples) is list:
            samples = np.array(samples, dtype=np.int32)
        lengths = np.array(lengths)
        return samples, lengths

class DataSetWrapper(object):
    """Runs sampling on background worker threads feeding off a shared
    jobs queue, decoupling it from the training loop.

    `num_workers` > 1 produces several batches concurrently. This only
    speeds things up when the underlying DataSet uses read_mode="pread"
    -- mmap page faults hold the GIL and so can't overlap across
    threads. Batches are interchangeable, so workers may finish out of
    request order without consequence.
    """

    def __init__(self, dataset: DataSet, num_workers: int = 1):
        self.dataset = dataset
        self.num_workers = num_workers
        self.jobs_queue = Queue()
        self.results_queues: dict[tuple[int, int, str], Queue] = {}
        self.results_queues_lock = Lock()
        self.threads = [
            Thread(target=self._thread) for _ in range(num_workers)
        ]
        for t in self.threads:
            t.start()

    def join(self):
        # One sentinel per worker so each loop exits.
        for _ in range(self.num_workers):
            self.jobs_queue.put(None)

    def sample_tokens(
        self, ntokens: int, batch: int, tokenset_name: str
    ) -> np.ndarray:
        with timer("DataSet.sample_tokens"):
            key = (ntokens, batch, tokenset_name)

            with self.results_queues_lock:
                if key not in self.results_queues:
                    logging.info(f'Initializing the data queue for {key}')
                    self.results_queues[key] = Queue()
                    # Prime enough in-flight jobs to keep every worker
                    # busy plus one buffered batch of prefetch.
                    for _ in range(self.num_workers + 1):
                        self.jobs_queue.put(key)
                results_queue = self.results_queues[key]

            self.jobs_queue.put(key)
            return results_queue.get()

    def sample_bytes(
            self, nbytes: int, batch: int, tokenset_name: str) -> tuple[np.ndarray, np.ndarray]:
        with timer("DataSet.sample_bytes"):
            return self.dataset.sample_bytes(nbytes, batch, tokenset_name)

    def _thread(self):
        while True:
            key = self.jobs_queue.get()
            if key is None:
                break
            with timer("DataSet.sample_tokens"):
                sample = self.dataset.sample_tokens(*key)
            with self.results_queues_lock:
                results_queue = self.results_queues[key]
            results_queue.put(sample)
