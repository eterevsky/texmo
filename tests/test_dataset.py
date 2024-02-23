import logging
import os
from unittest import TestCase

import numpy as np

from texmo.tokens import set_tokens_dir
from texmo.dataset import DataSet, DataSetWrapper

# logging.disable(level=logging.ERROR)
set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


class DataSetTest(TestCase):
    def setUp(self):
        self._data = b"Roses are red,\nViolets are blue.\n\n"
        return super().setUp()

    def test_tokens_tokens_1(self):
        dataset = DataSet(data=self._data)
        batch = dataset.sample_tokens(1, 4, "tokens64_capswords_byteshuff")
        self.assertEqual(batch.shape, (4, 1))

    def test_tokens_bytes_1(self):
        dataset = DataSet(data=self._data)
        batch, lengths = dataset.sample_bytes(1, 4, "tokens64_capswords_byteshuff")
        self.assertEqual(batch.shape[0], 4)

    def test_tokens_thread_tokens_1(self):
        dataset = DataSet(data=self._data)
        wrapper = DataSetWrapper(dataset)
        batch = wrapper.sample_tokens(1, 4, "tokens64_capswords_byteshuff")
        self.assertEqual(batch.shape, (4, 1))
        wrapper.join()

    def test_tokens_thread_bytes_1(self):
        dataset = DataSet(data=self._data)
        wrapper = DataSetWrapper(dataset)
        batch, lengths = wrapper.sample_bytes(1, 4, "tokens64_capswords_byteshuff")
        self.assertEqual(batch.shape[0], 4)
        wrapper.join()
