"""Tests for the hand-edited named-regex preset store.

Every test points `load` at a tmp file: the real store is per-host
personal state at the repo root, and the loader must never be the
reason a page 500s, so the interesting cases are all malformed input.
"""

import json
import logging

from texmo import named_regexes


def _write(tmp_path, data):
    """Write `data` (a dict, or raw text) to a tmp store; return path."""
    path = tmp_path / 'named_regexes.json'
    if isinstance(data, str):
        path.write_text(data, encoding='utf-8')
    else:
        path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


def test_missing_file_is_no_presets(tmp_path):
    # Not an error, and emphatically not a file to create.
    path = str(tmp_path / 'named_regexes.json')
    assert named_regexes.load(path) == []
    assert not (tmp_path / 'named_regexes.json').exists()


def test_loads_and_sorts_case_insensitively(tmp_path):
    path = _write(tmp_path, {
        'zeta': {'regex': '.*zeta.*'},
        'Alpha': {'regex': '.*alpha.*', 'default_spec': 'bytes|gru.4'},
        'beta': {'regex': '.*beta.*'},
    })
    presets = named_regexes.load(path)
    assert [p['name'] for p in presets] == ['Alpha', 'beta', 'zeta']
    assert presets[0]['default_spec'] == 'bytes|gru.4'
    # Optional field normalizes to empty rather than going missing.
    assert presets[1]['default_spec'] == ''


def test_malformed_json_is_empty(tmp_path, caplog):
    path = _write(tmp_path, '{"broken": ')
    with caplog.at_level(logging.WARNING):
        assert named_regexes.load(path) == []
    assert any('named_regexes.json' in r.message for r in caplog.records)


def test_top_level_list_is_empty(tmp_path, caplog):
    path = _write(tmp_path, ['not', 'an', 'object'])
    with caplog.at_level(logging.WARNING):
        assert named_regexes.load(path) == []


def test_bad_regex_is_skipped_with_a_warning(tmp_path, caplog):
    path = _write(tmp_path, {
        'good': {'regex': '.*gru.*'},
        'unbalanced': {'regex': '.*(gru.*'},
    })
    with caplog.at_level(logging.WARNING):
        presets = named_regexes.load(path)
    assert [p['name'] for p in presets] == ['good']
    assert any('unbalanced' in r.message for r in caplog.records)


def test_reserved_name_is_skipped(tmp_path, caplog):
    """'Unrestricted' is the virtual first choice, so a stored entry
    under that name could never be picked."""
    path = _write(tmp_path, {
        named_regexes.RESERVED_NAME: {'regex': '.*'},
        'unrestricted': {'regex': '.*'},  # same name, other case
        'real': {'regex': '.*gru.*'},
    })
    with caplog.at_level(logging.WARNING):
        presets = named_regexes.load(path)
    assert [p['name'] for p in presets] == ['real']
    assert sum('reserved' in r.message for r in caplog.records) == 2


def test_non_object_and_regexless_entries_are_skipped(tmp_path, caplog):
    path = _write(tmp_path, {
        'a_string': '.*gru.*',
        'no_regex': {'default_spec': 'bytes|gru.4'},
        'blank_regex': {'regex': '   '},
        'keeper': {'regex': '.*lstm.*'},
    })
    with caplog.at_level(logging.WARNING):
        presets = named_regexes.load(path)
    assert [p['name'] for p in presets] == ['keeper']


def test_blank_name_is_skipped_and_names_are_stripped(tmp_path, caplog):
    path = _write(tmp_path, {
        '   ': {'regex': '.*'},
        '  padded  ': {'regex': '.*gru.*'},
    })
    with caplog.at_level(logging.WARNING):
        presets = named_regexes.load(path)
    assert [p['name'] for p in presets] == ['padded']


def test_non_string_default_spec_is_dropped_not_fatal(tmp_path, caplog):
    path = _write(tmp_path, {'a': {'regex': '.*', 'default_spec': 17}})
    with caplog.at_level(logging.WARNING):
        presets = named_regexes.load(path)
    assert len(presets) == 1 and presets[0]['default_spec'] == ''


def test_load_reflects_edits_without_restart(tmp_path):
    """The file is hand-edited while the server runs, so each call
    reads it afresh."""
    path = _write(tmp_path, {'first': {'regex': '.*a.*'}})
    assert [p['name'] for p in named_regexes.load(path)] == ['first']
    _write(tmp_path, {'second': {'regex': '.*b.*'}})
    assert [p['name'] for p in named_regexes.load(path)] == ['second']
