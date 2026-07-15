"""File-backed persistence for the fitted predictors (timing/loss).

They used to live in the DB's `model` table; as refittable artifacts
they only bloated the DB and its backups. Now they are pickles next
to the DB file:

    <db dir>/<name>_model.pickle                    (current)
    <db dir>/<name>_model.YYYYmmdd-HHMMSS.pickle    (backups)

Saves are atomic (temp file + os.replace); the previous file is
rotated into a timestamped backup first (stamped with its own
mtime), and only the newest `_KEEP_BACKUPS` backups are kept -- a
bad refit can't destroy the last good model. Loads read the current
file only and return None when it is missing or unreadable; callers
treat that as "refit" (backups are for manual recovery).
"""
import logging
import os
import pickle
import tempfile
from datetime import datetime

_KEEP_BACKUPS = 3


def predictor_path(db_path: str, name: str) -> str:
    base = os.path.dirname(os.path.abspath(db_path))
    return os.path.join(base, f'{name}_model.pickle')


def _rotate(path: str, name: str) -> None:
    if not os.path.exists(path):
        return
    stamp = datetime.fromtimestamp(
        os.path.getmtime(path)).strftime('%Y%m%d-%H%M%S')
    backup = path[:-len('.pickle')] + f'.{stamp}.pickle'
    os.replace(path, backup)  # same-second refit: overwrite, fine
    base = os.path.dirname(path)
    prefix = f'{name}_model.'
    backups = sorted(
        f for f in os.listdir(base)
        if f.startswith(prefix) and f.endswith('.pickle')
        and f != f'{name}_model.pickle')
    for old in backups[:-_KEEP_BACKUPS]:
        os.unlink(os.path.join(base, old))


def save_predictor(db_path: str, name: str, obj) -> None:
    path = predictor_path(db_path, name)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f'.{name}_model.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f)
        _rotate(path, name)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logging.info(
        f"Saved predictor '{name}' to {path} "
        f"({os.path.getsize(path)} bytes)")


def load_predictor(db_path: str, name: str):
    path = predictor_path(db_path, name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
