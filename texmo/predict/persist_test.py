"""Tests for the file-backed predictor store (rotation + recovery)."""
import os
import pickle

from texmo.predict.persist import (
    _KEEP_BACKUPS,
    load_predictor,
    predictor_path,
    save_predictor,
)


def test_round_trip(tmp_path):
    db = str(tmp_path / 'db.sqlite')
    assert load_predictor(db, 'loss') is None
    save_predictor(db, 'loss', {'v': 1})
    assert load_predictor(db, 'loss') == {'v': 1}


def _backups(tmp_path, name):
    return sorted(
        f for f in os.listdir(tmp_path)
        if f.startswith(f'{name}_model.') and f != f'{name}_model.pickle')


def test_rotation_keeps_last_backups(tmp_path):
    db = str(tmp_path / 'db.sqlite')
    for v in range(6):
        save_predictor(db, 'loss', {'v': v})
        # Force distinct mtimes (same-second saves would share a
        # backup stamp): mtime = v seconds after the epoch, so the
        # backup stamps sort in version order.
        os.utime(predictor_path(db, 'loss'), (v, v))

    assert load_predictor(db, 'loss') == {'v': 5}
    backups = _backups(tmp_path, 'loss')
    assert len(backups) == _KEEP_BACKUPS
    # The newest backup holds the immediately previous version.
    with open(tmp_path / backups[-1], 'rb') as f:
        assert pickle.load(f) == {'v': 4}


def test_names_do_not_cross_rotate(tmp_path):
    db = str(tmp_path / 'db.sqlite')
    save_predictor(db, 'timing', {'t': 0})
    for v in range(5):
        save_predictor(db, 'loss', {'v': v})
        os.utime(predictor_path(db, 'loss'), (v, v))
    assert load_predictor(db, 'timing') == {'t': 0}
    assert _backups(tmp_path, 'timing') == []
