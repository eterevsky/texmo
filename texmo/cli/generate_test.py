import argparse

import pytest

from texmo.cli.generate import (
    add_prefix_args,
    parse_temperatures,
    print_continuations,
    read_prefix,
)


def _parser(required: bool = True, default: str | None = None):
    parser = argparse.ArgumentParser()
    add_prefix_args(parser, required=required, default=default)
    return parser


def test_read_prefix_from_string():
    args = _parser().parse_args(['--prefix', 'Roses are red'])
    assert read_prefix(args) == 'Roses are red'


def test_read_prefix_from_file(tmp_path):
    path = tmp_path / 'prefix.txt'
    text = 'Roses are red\nViolets are blue\n'
    path.write_text(text, encoding='utf-8', newline='')
    args = _parser().parse_args(['--prefix-file', str(path)])
    assert read_prefix(args) == text


def test_read_prefix_file_keeps_unicode(tmp_path):
    path = tmp_path / 'prefix.txt'
    path.write_text('café — naïve\n', encoding='utf-8')
    args = _parser().parse_args(['--prefix-file', str(path)])
    assert read_prefix(args) == 'café — naïve\n'


def test_prefix_file_wins_over_default(tmp_path):
    """train's --prefix has a default; --prefix-file must override it."""
    path = tmp_path / 'prefix.txt'
    path.write_text('from file', encoding='utf-8')
    parser = _parser(required=False, default='the default')
    args = parser.parse_args(['--prefix-file', str(path)])
    assert read_prefix(args) == 'from file'


def test_prefix_optional_when_not_required():
    args = _parser(required=False).parse_args([])
    assert read_prefix(args) is None


def test_prefix_and_prefix_file_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        _parser().parse_args(['--prefix', 'a', '--prefix-file', 'b'])


def test_one_prefix_required():
    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_parse_temperatures():
    assert parse_temperatures('0.3,1.0') == [0.3, 1.0]
    assert parse_temperatures('1.0') == [1.0]
    assert parse_temperatures('0.1, ,0.2') == [0.1, 0.2]


def test_print_continuations_sections():
    lines = []
    print_continuations(
        lambda t: f'text at {t}'.encode(), [0.3, 1.0], lines.append)
    assert lines == [
        '--- T = 0.3 ---', 'text at 0.3',
        '--- T = 1.0 ---', 'text at 1.0',
        '',
    ]


def test_print_continuations_undecodable_bytes():
    """A continuation can end mid character; the raw repr is printed."""
    lines = []
    print_continuations(lambda t: b'ab\xff', [1.0], lines.append)
    assert lines[1] == str(b'ab\xff')
