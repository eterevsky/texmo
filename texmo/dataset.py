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
from .tokens.processing import process

# os.pread (read at an explicit offset, no shared file cursor) is
# Unix-only. On Windows we fall back to lseek+read, which is fine for
# the single-threaded sampling path.
_HAS_PREAD = hasattr(os, "pread")


def _open_mmap(path: str) -> tuple["file", mmap.mmap]:
    file = open(path, "rb")
    return (file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ))


class DataSet(object):
    def __init__(
        self,
        path: str | None = None,
        path_processed: str | None = None,
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
        self.read_mode = read_mode

        if data is not None:
            assert path is None
            assert path_processed is None
            logging.info(f"Creating DataSet from data with len = {itoa3(len(data))}")
            self.data_file = None
            self.processed_file = None
            self.data = data
            self.processed_data = process(data)
            logging.info(f"Processed size: {itoa3(len(self.processed_data))}")
        else:
            assert path is not None
            self.data_file, self.data = _open_mmap(path)
            logging.info(f"Creating DataSet from {path} ({itoa3(len(self.data))})")
            if path_processed:
                self.processed_file, self.processed_data = _open_mmap(path_processed)
                logging.info(
                    f"Using pre-processed data from {path_processed} ({itoa3(len(self.processed_data))})"
                )
            else:
                self.processed_file = None
                self.processed_data = None

        # File descriptors for the pread path; None when the buffer is
        # an in-memory bytes object (pread then falls back to slicing).
        self.data_fd = self.data_file.fileno() if self.data_file is not None else None
        self.processed_fd = (
            self.processed_file.fileno() if self.processed_file is not None else None
        )

        self.data_size = len(self.data)
        self.processed_data_size = len(self.processed_data) if self.processed_data else None

    def __del__(self):
        if getattr(self, "data_file", None) is not None:
            self.data_file.close()
        if getattr(self, "processed_file", None) is not None:
            self.processed_file.close()

    def _select_chunk_start(
        self, processing: bool
    ) -> tuple[bytes | mmap.mmap, int | None, int, int, bool]:
        """
        Returns:
            (data buffer, its file descriptor or None, data size,
             start offset, whether the data needs processing)
        """
        if processing and self.processed_data:
            data = self.processed_data
            fd = self.processed_fd
            size = self.processed_data_size
            processing = False
        else:
            data = self.data
            fd = self.data_fd
            size = self.data_size

        start = random.randint(0, size)
        return data, fd, size, start, processing

    def _read(
        self, data: bytes | mmap.mmap, fd: int | None, start: int, size: int
    ) -> bytes:
        """Fetch `size` bytes at `start`, honoring self.read_mode."""
        if self.read_mode == "pread" and fd is not None:
            if _HAS_PREAD:
                return os.pread(fd, size, start)
            # Windows fallback: no atomic pread. Safe here because
            # sampling is single-threaded.
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
        assert tokenset.processing in ("raw", "capswords")
        processing = tokenset.processing == "capswords"

        samples = []
        while len(samples) < batch:
            data, fd, data_size, start, need_processing = self._select_chunk_start(processing)

            if need_processing:
                size = max(round(ntokens * tokenset.avg_bytes_per_token * 1.2), 16)
            else:
                size = max(round(ntokens * tokenset.avg_proc_bytes_per_token * 1.2), 16)

            assert size <= data_size

            while True:
                if start + size > data_size:
                    break

                chunk = self._read(data, fd, start, size)
                valid_start = 0
                while 128 <= chunk[valid_start] < 192:
                    valid_start += 1
                valid_end = len(chunk) - 1
                while 128 <= chunk[valid_end] < 192:
                    valid_end -= 1
                chunk = chunk[valid_start : valid_end]

                if need_processing:
                    chunk = process(chunk)

                sample = tokenizer.tokenize_processed(chunk)
                if len(sample) >= ntokens:
                    break
                size *= 2

            if start + size > data_size:
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
        assert tokenset.processing in ("raw", "capswords")
        processing = tokenset.processing == "capswords"

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

            chunk = self.data[start : end]
            if processing:
                chunk = process(chunk)

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
    def __init__(self, dataset: DataSet):
        self.dataset = dataset
        self.jobs_queue = Queue()
        self.results_queues: dict[tuple[int, int, str], Queue] = {}
        self.results_queues_lock = Lock()
        self.thread = Thread(target=self._thread)
        self.thread.start()

    def join(self):
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
                    self.jobs_queue.put(key)
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
