import logging
import mmap
import numpy as np
import random

from .common import itoa3
from .tokens import get_tokenizer
from .tokens.processing import process


def _open_mmap(path: str) -> mmap.mmap:
    file = open(path, "rb")
    return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)


def _find_random_paragraph_start(data: bytes | mmap.mmap) -> int:
    """Find the first character of the paragraph by iterating backwards."""
    start = random.randrange(len(data))

    while start > 1 and (
        data[start] == 10 or data[start - 1] != 10 or data[start - 2] != 10
    ):
        start -= 1

    if start == 1:
        if data[0] == 10 and data[1] != 10:
            return 1
        else:
            return 0

    return start


class DataSet(object):
    def __init__(
        self,
        path: str | None = None,
        path_processed: str | None = None,
        data: bytes | mmap.mmap | None = None,
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
            self.data = _open_mmap(path)
            logging.info(f"Creating DataSet from {path} ({itoa3(len(self.data))})")
            if path_processed:
                self.processed_data = _open_mmap(path_processed)
                logging.info(
                    f"Using pre-processed data from {path_processed} ({itoa3(len(self.processed_data))})"
                )
            else:
                self.processed_data = None

    def _select_chunk_start(
        self, processing: bool
    ) -> tuple[bytes | mmap.mmap, int, bool]:
        """
        Returns:
            (data, start offset, whether the data needs processing)
        """
        if processing and self.processed_data:
            start = _find_random_paragraph_start(self.processed_data)
            return self.processed_data, start, False

        start = _find_random_paragraph_start(self.data)
        print(repr(self.data[start: start + 10]))
        return self.data, start, processing

    def sample_tokens(self, ntokens: int, batch: int, tokenset_name: str) -> np.ndarray:
        """Generate a random sample of input data.

        Returns:
            An array with the shape (batch, ntokens) of the type np.int32.
        """
        samples = []
        tokenizer = get_tokenizer(tokenset_name)
        tokenset = tokenizer.tokenset
        assert tokenset.processing in ("raw", "capswords")
        processing = tokenset.processing == "capswords"

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

                print(f"initial chunk: {chunk}")
                sample = tokenizer.tokenize_processed(chunk)
                if len(sample) >= ntokens:
                    break
                size *= 2

            if start + size > len(data):
                continue

            samples.append(sample[:ntokens])

        return np.array(samples, dtype=np.int32)
