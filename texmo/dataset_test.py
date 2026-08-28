import os
import random
from threading import Thread

import numpy as np
import pytest

import texmo.dataset
from texmo.dataset import DataSet, DataSetWrapper, MixDataSet
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


def test_pread_mode_never_maps(tmp_path):
    """pread mode keeps no mapping at all -- just the fd -- and both
    sampling paths work off explicit reads."""
    path = tmp_path / "data.txt"
    path.write_bytes(_DATA * 50)
    ds = DataSet(path=str(path), read_mode="pread")
    assert ds.data is None
    assert ds.data_size == len(_DATA) * 50

    batch = ds.sample_tokens(8, 2, "bytes")
    assert batch.shape == (2, 8)
    batch, lengths = ds.sample_bytes(16, 2, "tokens.32.hexbpe")
    assert batch.shape[0] == 2
    assert (lengths > 0).all()


def test_mmap_mode_still_maps(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(_DATA * 50)
    ds = DataSet(path=str(path))
    assert ds.data is not None
    batch = ds.sample_tokens(8, 2, "bytes")
    assert batch.shape == (2, 8)


def _make_ascii_file(tmp_path, size: int = 64 * 1024) -> tuple[str, bytes]:
    """A file of printable ASCII (no UTF-8 continuation bytes, so a
    "bytes" sample is exactly the slice that was read) with no repeats
    long enough for a torn read to look like a valid slice."""
    data = bytes(random.Random(17).choices(range(33, 127), k=size))
    path = tmp_path / "ascii.txt"
    path.write_bytes(data)
    return str(path), data


def _sample_concurrently(wrapper: DataSetWrapper, nthreads: int, rounds: int):
    """Drive `wrapper.sample_tokens` from several threads; returns the
    batches and any exception raised on a worker's caller thread."""
    batches = []
    errors = []

    def _run():
        try:
            for _ in range(rounds):
                batches.append(wrapper.sample_tokens(32, 4, "bytes"))
        except Exception as e:
            errors.append(e)

    threads = [Thread(target=_run) for _ in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return batches, errors


def _assert_valid_slices(batches, data: bytes):
    assert batches
    for batch in batches:
        assert batch.shape == (4, 32)
        for row in batch:
            assert bytes(row.tolist()) in data


def test_pread_concurrent_sampling(tmp_path):
    path, data = _make_ascii_file(tmp_path)
    wrapper = DataSetWrapper(
        DataSet(path=path, read_mode="pread"), num_workers=4)
    try:
        batches, errors = _sample_concurrently(wrapper, nthreads=4, rounds=8)
    finally:
        wrapper.join()
    assert errors == []
    _assert_valid_slices(batches, data)


def test_pread_concurrent_sampling_without_os_pread(tmp_path, monkeypatch):
    """The no-os.pread path (always taken on Windows) reads through a
    per-thread fd; sharing one file cursor would tear these samples."""
    monkeypatch.setattr(texmo.dataset, "_HAS_PREAD", False)
    path, data = _make_ascii_file(tmp_path)
    wrapper = DataSetWrapper(
        DataSet(path=path, read_mode="pread"), num_workers=4)
    try:
        batches, errors = _sample_concurrently(wrapper, nthreads=4, rounds=8)
    finally:
        wrapper.join()
    assert errors == []
    _assert_valid_slices(batches, data)


def test_pread_reads_are_thread_isolated(tmp_path, monkeypatch):
    """Every thread must get the bytes at the offset *it* asked for.
    A shared file cursor would hand a thread another thread's offset --
    still a valid slice of the file, so only exact offsets catch it."""
    monkeypatch.setattr(texmo.dataset, "_HAS_PREAD", False)
    path, data = _make_ascii_file(tmp_path, 1 << 20)
    ds = DataSet(path=path, read_mode="pread")
    mismatches = []

    def _run(i: int):
        for r in range(500):
            start = (i * 7919 + r * 61) % (len(data) - 512)
            got = ds._read(ds.data, ds.data_fd, start, 400)
            if got != data[start : start + 400]:
                mismatches.append((i, r))

    threads = [Thread(target=_run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mismatches == []


def test_pread_switched_on_after_construction(tmp_path, monkeypatch):
    """cli/sample.py's A/B benchmark flips read_mode on a DataSet built
    in mmap mode, so per-thread fds must open lazily from the path."""
    monkeypatch.setattr(texmo.dataset, "_HAS_PREAD", False)
    path, data = _make_ascii_file(tmp_path, 8 * 1024)
    ds = DataSet(path=path)
    ds.read_mode = "pread"
    _assert_valid_slices([ds.sample_tokens(32, 4, "bytes")], data)


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


def _ab_sources(n: int = 4096) -> tuple[DataSet, DataSet]:
    """Two corpora with no byte in common, so every sampled row can be
    attributed to the source it came from."""
    return DataSet(data=b"a" * n), DataSet(data=b"b" * n)


def _row_sources(batch: np.ndarray) -> list[int]:
    """0 for a row of 'a's, 1 for a row of 'b's ('bytes' tokenset, so
    tokens are the bytes themselves)."""
    sources = []
    for row in batch:
        values = set(row.tolist())
        assert values <= {ord("a"), ord("b")}, values
        assert len(values) == 1, "a window spans two corpora"
        sources.append(0 if values == {ord("a")} else 1)
    return sources


def test_mix_weights_proportions():
    mix = MixDataSet(list(_ab_sources()), [0.75, 0.25])
    random.seed(4)
    sources = _row_sources(mix.sample_tokens(8, 2000, "bytes"))
    share_a = sources.count(0) / len(sources)
    # 2000 draws: the sampling error is ~1%; 5% is comfortable slack.
    assert abs(share_a - 0.75) < 0.05


def test_mix_default_weights_are_equal():
    mix = MixDataSet(list(_ab_sources()))
    assert mix.weights == [0.5, 0.5]
    random.seed(5)
    sources = _row_sources(mix.sample_tokens(8, 1000, "bytes"))
    assert abs(sources.count(0) / len(sources) - 0.5) < 0.05


def test_mix_shuffles_rows():
    """Rows must be interleaved: ManagerJax slices one big sample into
    consecutive per-step batches, so grouped rows would mean whole
    steps trained on a single source."""
    mix = MixDataSet(list(_ab_sources()))
    random.seed(6)
    sources = _row_sources(mix.sample_tokens(8, 200, "bytes"))
    runs = 1 + sum(a != b for a, b in zip(sources, sources[1:]))
    # Grouped output has 2 runs; a shuffle of ~100+100 has ~100.
    assert runs > 20


def test_mix_sample_bytes_mixes():
    mix = MixDataSet(list(_ab_sources()))
    random.seed(7)
    batch, lengths = mix.sample_bytes(16, 200, "bytes")
    assert batch.shape[0] == 200
    assert lengths.shape == (200,)
    sources = _row_sources(batch)
    assert abs(sources.count(0) / len(sources) - 0.5) < 0.15


def test_mix_sample_bytes_ragged_sources():
    """Sources tokenize to different lengths; the mix pads to the
    longest across sources and keeps each row's own length."""
    sources = [DataSet(data=_DATA * 20), DataSet(data=b"aaaa bbbb " * 100)]
    mix = MixDataSet(sources)
    random.seed(8)
    batch, lengths = mix.sample_bytes(16, 32, "tokens.32.hexbpe")
    assert batch.shape[0] == 32
    assert lengths.shape == (32,)
    assert batch.shape[1] == max(lengths)


def test_mix_single_source_matches_plain_dataset():
    """A one-file 'mix' is a pass-through: no multinomial, no shuffle,
    so it draws the same random numbers as the bare DataSet."""
    random.seed(11)
    plain = DataSet(data=_DATA * 20).sample_tokens(8, 16, "bytes")
    random.seed(11)
    mixed = MixDataSet([DataSet(data=_DATA * 20)]).sample_tokens(8, 16, "bytes")
    assert (plain == mixed).all()

    random.seed(12)
    pb, pl = DataSet(data=_DATA * 20).sample_bytes(16, 8, "bytes")
    random.seed(12)
    mb, ml = MixDataSet([DataSet(data=_DATA * 20)]).sample_bytes(16, 8, "bytes")
    assert (pb == mb).all() and (pl == ml).all()


def test_mix_rejects_bad_weights():
    sources = list(_ab_sources())
    with pytest.raises(ValueError):
        MixDataSet(sources, [1.0])
    with pytest.raises(ValueError):
        MixDataSet(sources, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        MixDataSet(sources, [1.0, 0.0])
    with pytest.raises(ValueError):
        MixDataSet(sources, [1.0, -1.0])


def test_mix_str_lists_sources_and_weights(tmp_path):
    path_a = tmp_path / "a.txt"
    path_b = tmp_path / "b.txt"
    path_a.write_bytes(b"a" * 4096)
    path_b.write_bytes(b"b" * 4096)
    mix = MixDataSet(
        [DataSet(path=str(path_a)), DataSet(path=str(path_b))], [3.0, 1.0])
    assert str(mix) == f"{path_a}:0.75,{path_b}:0.25"


def test_wrapper_over_mix():
    """DataSetWrapper needs nothing but the sampling surface, so the
    mix prefetches on worker threads like a plain DataSet."""
    wrapper = DataSetWrapper(MixDataSet(list(_ab_sources())), num_workers=2)
    try:
        batch = wrapper.sample_tokens(8, 64, "bytes")
        assert batch.shape == (64, 8)
        assert set(_row_sources(batch)) == {0, 1}
        bytes_batch, lengths = wrapper.sample_bytes(16, 16, "bytes")
        assert bytes_batch.shape[0] == 16
    finally:
        wrapper.join()
    assert str(wrapper) == str(wrapper.dataset)


def test_wrapper_join_is_idempotent(dataset):
    wrapper = DataSetWrapper(dataset)
    wrapper.join()
    assert wrapper.joined
    wrapper.join()
    assert wrapper.jobs_queue.qsize() <= wrapper.num_workers
