import pytest

from texmo.cli.train import parse_lr


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
