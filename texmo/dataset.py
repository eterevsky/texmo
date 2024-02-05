import logging
import mmap
import numpy as np
import random
import itertools
from queue import Queue
from threading import Thread

from .common import itoa3
from .tokens import get_tokenizer
from .tokens.processing import process
from .latency import timer


def _open_mmap(path: str) -> tuple["file", mmap.mmap]:
    file = open(path, "rb")
    return (file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ))


def _find_random_paragraph_start(data: bytes | mmap.mmap, size: int = None) -> int:
    """Find the first character of the paragraph by iterating backwards."""
    size = size or len(data)
    start = random.randrange(size)
    end_paragraph = data.rfind(b"\n\n", 0, start)
    if end_paragraph == -1:
        return 0
    else:
        return end_paragraph + 2


def _sample_start_thread(jobs: Queue, results: Queue, data: bytes | mmap.mmap):
    size = len(data)
    while jobs.get():
        results.put(_find_random_paragraph_start(data, size))


class DataSet(object):
    def __init__(
        self,
        path: str | None = None,
        path_processed: str | None = None,
        data: bytes | mmap.mmap | None = None,
        parallel_chunks: bool = False,
    ):
        if data is not None:
            assert path is None
            assert path_processed is None
            logging.info(f"Creating DataSet from data with len = {itoa3(len(data))}")
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
        
        self._parallel_chunks = parallel_chunks
        if self._parallel_chunks:
            self._data_jobs = Queue()
            self._data_results = Queue()
            self._data_thread = Thread(
                target=_sample_start_thread,
                args=(self._data_jobs, self._data_results, self.data),
            )
            self._data_thread.start()
            for _ in range(4):
                self._data_jobs.put(True)

            if path_processed:
                self._processed_jobs = Queue()
                self._processed_results = Queue()
                self._processed_thread = Thread(
                    target=_sample_start_thread,
                    args=(self._processed_jobs, self._processed_results, self.processed_data),
                )
                self._processed_thread.start()
                for _ in range(4):
                    self._processed_jobs.put(True)

    def __del__(self):
        if hasattr(self, "_data_jobs"):
            self._data_jobs.put(False)
        if hasattr(self, "_processed_jobs"):
            self._processed_jobs.put(False)
        if hasattr(self, "data_file"):
            self.data_file.close()
        if hasattr(self, "processed_file"):
            self.processed_file.close()

    def join(self):
        if hasattr(self, "_data_jobs"):
            self._data_jobs.put(False)
        if hasattr(self, "_processed_jobs"):
            self._processed_jobs.put(False)

    def _select_chunk_start(
        self, processing: bool
    ) -> tuple[bytes | mmap.mmap, int, bool]:
        """
        Returns:
            (data, start offset, whether the data needs processing)
        """
        with timer("DataSet._select_chunk_start"):
            if processing and self.processed_data:
                if self._parallel_chunks:
                    self._processed_jobs.put(True)
                    start = self._processed_results.get()
                else:
                    start = _find_random_paragraph_start(self.processed_data)
                return self.processed_data, start, False

            if self._parallel_chunks:
                self._data_jobs.put(True)
                start = self._data_results.get()
            else:
                start = _find_random_paragraph_start(self.data)
            return self.data, start, processing

    def _sample_tokens_impl(
        self, ntokens: int, batch: int, tokenset_name: str
    ) -> np.ndarray:
        tokenizer = get_tokenizer(tokenset_name)
        tokenset = tokenizer.tokenset
        assert tokenset.processing in ("raw", "capswords")
        processing = tokenset.processing == "capswords"

        samples = []
        while len(samples) < batch:
            data, start, need_processing = self._select_chunk_start(processing)

            if need_processing:
                size = max(round(ntokens * tokenset.avg_bytes_per_token * 1.2), 1)
            else:
                size = max(round(ntokens * tokenset.avg_proc_bytes_per_token * 1.2), 1)

            while True:
                if start + size > len(data):
                    break
                chunk = data[start : start + size]
                if need_processing:
                    chunk = process(chunk)

                sample = tokenizer.tokenize_processed(chunk)
                if len(sample) >= ntokens:
                    break
                size *= 2

            if start + size > len(data):
                continue

            samples.append(sample[:ntokens])

        return np.array(samples, dtype=np.int32)

    def sample_tokens(self, ntokens: int, batch: int, tokenset_name: str) -> np.ndarray:
        """Generate a random sample of input data.

        Returns:
            An array with the shape (batch, ntokens) of the type np.int32.
        """
        with timer("DataSet.sample_tokens"):
            return self._sample_tokens_impl(ntokens, batch, tokenset_name)

    def _sample_bytes_impl(
        self, nbytes: int, batch: int, tokenset_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        tokenizer = get_tokenizer(tokenset_name)
        tokenset = tokenizer.tokenset
        assert tokenset.processing in ("raw", "capswords")
        processing = tokenset.processing == "capswords"

        samples = []
        while len(samples) < batch:
            start = _find_random_paragraph_start(self.data)
            if start + nbytes > len(self.data):
                continue
            end = start + nbytes
            while 128 <= self.data[end] < 192:
                end += 1
            chunk = self.data[start : end]
            if processing:
                chunk = process(chunk)

            sample = tokenizer.tokenize_processed(chunk)
            samples.append(sample)

        lengths = list(map(len, samples))
        max_len = max(lengths)
        for sample in samples:
            l = len(sample)
            sample.extend(itertools.repeat(0, max_len - l))

        samples = np.array(samples, dtype=np.int32)
        length = np.array(lengths)
        return samples, lengths

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
        with timer("DataSet.sample_bytes"):
            return self._sample_bytes_impl(nbytes, batch, tokenset_name)
