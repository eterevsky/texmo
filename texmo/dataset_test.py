import os

import numpy as np
import pytest

from texmo.dataset import DataSet, DataSetWrapper
from texmo.tokens import set_tokens_dir

set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))

_DATA = b"Roses are red,\nViolets are blue.\n\n"


@pytest.fixture
def dataset():
    return DataSet(data=_DATA)


def test_sample_tokens(dataset):
    batch = dataset.sample_tokens(1, 4, "tokens.64.shift")
    assert batch.shape == (4, 1)
    assert np.issubdtype(batch.dtype, np.integer)


def test_sample_bytes(dataset):
    batch, lengths = dataset.sample_bytes(1, 4, "tokens.64.shift")
    assert batch.shape[0] == 4


def test_sample_tokens_bytes(dataset):
    batch = dataset.sample_tokens(8, 2, "bytes")
    assert batch.shape == (2, 8)
    assert np.issubdtype(batch.dtype, np.integer)
    assert (batch >= 0).all() and (batch < 256).all()


def test_sample_tokens_bits(dataset):
    batch = dataset.sample_tokens(16, 2, "bits.1")
    assert batch.shape == (2, 16)
    assert np.issubdtype(batch.dtype, np.integer)
    assert (batch >= 0).all() and (batch < 2).all()


def test_sample_tokens_bits4(dataset):
    batch = dataset.sample_tokens(8, 2, "bits.4")
    assert batch.shape == (2, 8)
    assert np.issubdtype(batch.dtype, np.integer)
    assert (batch >= 0).all() and (batch < 16).all()


def test_sample_tokens_hexbpe(dataset):
    # capswords2 has no preprocessed on-disk buffer, so this goes down
    # the on-the-fly path: raw chunk -> process2 -> BPE merge loop.
    batch = dataset.sample_tokens(4, 3, "tokens.32.hexbpe")
    assert batch.shape == (3, 4)
    assert np.issubdtype(batch.dtype, np.integer)
    assert (batch >= 0).all() and (batch < 32).all()


def test_sample_bytes_hexbpe(dataset):
    batch, lengths = dataset.sample_bytes(16, 2, "tokens.32.hexbpe")
    assert batch.shape[0] == 2
    assert (lengths > 0).all()
    assert (batch >= 0).all() and (batch < 32).all()


def test_wrapper_tokens(dataset):
    wrapper = DataSetWrapper(dataset)
    batch = wrapper.sample_tokens(8, 4, "bytes")
    assert batch.shape == (4, 8)
    wrapper.join()


def test_wrapper_bytes(dataset):
    wrapper = DataSetWrapper(dataset)
    batch, lengths = wrapper.sample_bytes(1, 4, "bytes")
    assert batch.shape[0] == 4
    wrapper.join()
