import pytest

from texmo.cli.train import build_data, parse_data_spec, parse_lr
from texmo.dataset import DataSet, MixDataSet


def test_parse_lr_float():
    assert parse_lr("0.01") == 0.01
    assert parse_lr("0.5") == 0.5


def test_parse_lr_int():
    assert parse_lr("1") == 1.0


def test_parse_lr_pow2():
    """'2^X' parses as 2**X (negative or non-negative integer)."""
    assert parse_lr("2^0") == 1.0
    assert parse_lr("2^-7") == 1 / 128
    assert parse_lr("2^3") == 8.0


def test_parse_lr_fraction():
    """'1/128' parses as 1.0 / 128.0 -- matches the format LRs appear in
    in conf reports."""
    assert parse_lr("1/128") == pytest.approx(1 / 128)
    assert parse_lr("3/8") == pytest.approx(0.375)


def test_parse_data_spec_single_file():
    spec = parse_data_spec("data/a.txt", None, None, 100)
    assert spec.files == ["data/a.txt"]
    assert spec.weights == [1.0]
    assert spec.switch_step is None


def test_parse_data_spec_mix_default_weights():
    spec = parse_data_spec("a.txt,b.txt", None, None, 100)
    assert spec.files == ["a.txt", "b.txt"]
    assert spec.weights == [1.0, 1.0]


def test_parse_data_spec_weights():
    spec = parse_data_spec("a.txt, b.txt", "0.7,0.3", None, 100)
    assert spec.files == ["a.txt", "b.txt"]
    assert spec.weights == [0.7, 0.3]


def test_parse_data_spec_switch_step_rounds_from_fraction():
    spec = parse_data_spec("a.txt,b.txt", None, 0.25, 512)
    assert spec.switch_step == 128


def test_parse_data_spec_switch_with_weights():
    """Weights cover the pre-switch mix; the last file is the
    post-switch corpus."""
    spec = parse_data_spec("a.txt,b.txt,c.txt", "3,1,1", 0.5, 200)
    assert spec.weights == [3.0, 1.0, 1.0]
    assert spec.switch_step == 100


def test_parse_data_spec_rejects_weight_count_mismatch():
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt,b.txt", "0.5", None, 100)
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt", "0.5,0.5", None, 100)


def test_parse_data_spec_rejects_nonpositive_weights():
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt,b.txt", "1,0", None, 100)
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt,b.txt", "1,-1", None, 100)


def test_parse_data_spec_rejects_nonnumeric_weights():
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt,b.txt", "1,x", None, 100)


def test_parse_data_spec_rejects_switch_out_of_range():
    for switch in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(SystemExit):
            parse_data_spec("a.txt,b.txt", None, switch, 100)


def test_parse_data_spec_switch_needs_two_files():
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt", None, 0.5, 100)


def test_parse_data_spec_switch_needs_steps():
    with pytest.raises(SystemExit):
        parse_data_spec("a.txt,b.txt", None, 0.5, None)


def test_parse_data_spec_rejects_empty_data():
    with pytest.raises(SystemExit):
        parse_data_spec(" , ", None, None, 100)


def _write_corpora(tmp_path):
    for name, byte in (("a.txt", b"a"), ("b.txt", b"b"), ("c.txt", b"c")):
        (tmp_path / name).write_bytes(byte * 4096)
    return [str(tmp_path / n) for n in ("a.txt", "b.txt", "c.txt")]


def test_build_data_single_file_is_plain_dataset(tmp_path):
    files = _write_corpora(tmp_path)
    spec = parse_data_spec(files[0], None, None, 100)
    wrapper, post, all_wrappers = build_data(spec, 1, 100)
    try:
        assert isinstance(wrapper.dataset, DataSet)
        assert post is None
        assert all_wrappers == [wrapper]
    finally:
        for w in all_wrappers:
            w.join()


def test_build_data_mix(tmp_path):
    files = _write_corpora(tmp_path)
    spec = parse_data_spec(f"{files[0]},{files[1]}", "3,1", None, 100)
    wrapper, post, all_wrappers = build_data(spec, 1, 100)
    try:
        assert isinstance(wrapper.dataset, MixDataSet)
        assert wrapper.dataset.weights == [0.75, 0.25]
        assert post is None
    finally:
        for w in all_wrappers:
            w.join()


def test_build_data_switch_splits_off_last_file(tmp_path):
    files = _write_corpora(tmp_path)
    spec = parse_data_spec(",".join(files), "3,1,1", 0.5, 100)
    wrapper, post, all_wrappers = build_data(spec, 1, 100)
    try:
        # Pre-switch: the first two files, in their given proportion.
        assert isinstance(wrapper.dataset, MixDataSet)
        assert wrapper.dataset.weights == [0.75, 0.25]
        assert [str(s) for s in wrapper.dataset.sources] == files[:2]
        # Post-switch: the last file alone.
        assert isinstance(post.dataset, DataSet)
        assert str(post.dataset) == files[2]
        assert all_wrappers == [wrapper, post]
    finally:
        for w in all_wrappers:
            w.join()
