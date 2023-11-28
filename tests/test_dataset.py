import logging
import os
from unittest import TestCase

import numpy as np

from texmo.tokens import set_tokens_dir
from texmo.dataset import DataSet

# logging.disable(level=logging.ERROR)
set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


class DataSetTest(TestCase):
    def setUp(self):
        self._data = b"Roses are red,\nViolets are blue.\n\n"
        return super().setUp()

    def test_tokens_in_process_tokens_1(self):
        dataset = DataSet(data=self._data, in_process=True)
        batch = dataset.sample_tokens(1, 4, "tokens64_capswords_bits4")
        self.assertEqual(batch.shape, (4, 1))

    def test_tokens_in_process_bytes_1(self):
        dataset = DataSet(data=self._data, in_process=True)
        batch, lengths = dataset.sample_bytes(1, 4, "tokens64_capswords_bits4")
        self.assertEqual(batch.shape[0], 4)

    def test_tokens_thread_tokens_1(self):
        dataset = DataSet(data=self._data, in_process=False)
        batch = dataset.sample_tokens(1, 4, "tokens64_capswords_bits4")
        self.assertEqual(batch.shape, (4, 1))
        # dataset.join()

    def test_tokens_thread_bytes_1(self):
        dataset = DataSet(data=self._data, in_process=False)
        batch, lengths = dataset.sample_bytes(1, 4, "tokens64_capswords_bits4")
        self.assertEqual(batch.shape[0], 4)
        # dataset.join()




