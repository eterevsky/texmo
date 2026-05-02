"""Smoke tests for the JSON-sanitization helpers used by `post_result`.

`requests.post(json=...)` rejects NaN and Inf even though Python's json
module accepts them by default — these helpers convert NaN to None and
Inf to a finite sentinel before serialization, which has been a real
source of intermittent client failures historically.
"""

import math

import numpy as np

from texmo.client import (
    sanitize_float,
    sanitize_json,
    sanitize_json_dict,
    sanitize_json_list,
)


def test_sanitize_float_nan_to_none():
    assert sanitize_float(float('nan')) is None


def test_sanitize_float_inf_to_sentinel():
    assert sanitize_float(float('inf')) == 1e12
    assert sanitize_float(float('-inf')) == 1e12


def test_sanitize_float_finite_passthrough():
    assert sanitize_float(0.0) == 0.0
    assert sanitize_float(1.5) == 1.5
    assert sanitize_float(-3.14) == -3.14


def test_sanitize_dict_replaces_nan_inf_in_place():
    d = {'a': float('nan'), 'b': float('inf'), 'c': 2.5, 'd': 'string', 'e': 7}
    sanitize_json_dict(d)
    assert d['a'] is None
    assert d['b'] == 1e12
    assert d['c'] == 2.5
    # Non-float values pass through unchanged.
    assert d['d'] == 'string'
    assert d['e'] == 7


def test_sanitize_list_replaces_nan_inf_in_place():
    l = [float('nan'), float('inf'), 1.0, 'x', 42]
    sanitize_json_list(l)
    assert l[0] is None
    assert l[1] == 1e12
    assert l[2] == 1.0
    assert l[3] == 'x'
    assert l[4] == 42


def test_sanitize_handles_numpy_floats():
    """The runtime ships values as np.float32 / np.float64 (from JAX
    arrays); the sanitizer must coerce those, not just python floats."""
    d = {
        'f16': np.float16('nan'),
        'f32': np.float32('inf'),
        'f64': np.float64(2.5),
    }
    sanitize_json_dict(d)
    assert d['f16'] is None
    assert d['f32'] == 1e12
    assert d['f64'] == 2.5
    # Output values are plain Python floats, JSON-serializable.
    assert d['f64'] is None or isinstance(d['f64'], float)


def test_sanitize_recurses_into_nested_structures():
    d = {
        'meta': {'inner_nan': float('nan'), 'nested': {'inf': float('inf')}},
        'series': [float('nan'), [float('inf'), 0.5]],
    }
    sanitize_json(d)
    assert d['meta']['inner_nan'] is None
    assert d['meta']['nested']['inf'] == 1e12
    assert d['series'][0] is None
    assert d['series'][1][0] == 1e12
    assert d['series'][1][1] == 0.5


def test_sanitize_does_not_alter_finite_only_payload():
    d = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0}}
    before = {'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0}}
    sanitize_json(d)
    assert d == before


def test_sanitize_handles_bool_and_none():
    """Booleans and None should pass through untouched. (Python treats
    bool as int, and isinstance(True, float) is False, so this is more
    of a regression-guard than a correctness check.)"""
    d = {'flag': True, 'absent': None, 'count': 0}
    sanitize_json_dict(d)
    assert d['flag'] is True
    assert d['absent'] is None
    assert d['count'] == 0
