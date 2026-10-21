"""Shared helpers for `DbReader` and `DbWriter`.

Only things needed by both classes live here: the regex bridge,
ndarray packing, template->SQL condition builder, the one SQL
string used on both sides (`FIND_CONF`), and the connection
bootstrap. Read-only types (`ConfScore`, `ConfWithRuns`) and the
read/write-specific SQL strings live in their respective modules.
"""

import logging
import os
import re
import sqlite3
from collections.abc import Iterable
from typing import Optional

import numpy as np

from ..common import INF
from ..configuration import DecayType
from ..precision import Precision


# --- byte/array packing helpers ---------------------------------------------


def _pack_ndarray(step_loss):
    step_loss = np.array(step_loss, dtype=np.float32)
    assert len(step_loss.shape) == 1
    return step_loss.tobytes()


def _unpack_ndarray(blob):
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _build_loss_trend(step_loss, model_version, params):
    from ..predict import build_loss_trend

    return build_loss_trend(step_loss, model_version, params)


# --- template -> SQL conditions ---------------------------------------------


def _make_template_conditions(template) -> tuple[list[str], dict]:
    conditions = []
    params = {}
    if template.spec is not None:
        conditions.append('spec = :spec')
        params['spec'] = template.spec
    if template.regex is not None:
        conditions.append('spec REGEXP :regex')
        params['regex'] = template.regex.pattern
    if template.lr.min > 0:
        conditions.append('lr >= :lr_min')
        params['lr_min'] = template.lr.min
    if template.lr.max < INF:
        conditions.append('lr <= :lr_max')
        params['lr_max'] = template.lr.max
    if template.length.min > 1:
        conditions.append('length >= :length_min')
        params['length_min'] = template.length.min
    if template.length.max < INF:
        conditions.append('length <= :length_max')
        params['length_max'] = template.length.max
    if template.batch.min > 1:
        conditions.append('batch >= :batch_min')
        params['batch_min'] = template.batch.min
    if template.batch.max < INF:
        conditions.append('batch <= :batch_max')
        params['batch_max'] = template.batch.max
    if template.steps.min >= 2:
        conditions.append('steps >= :steps_min')
        params['steps_min'] = template.steps.min
    if template.steps.max < INF:
        conditions.append('steps <= :steps_max')
        params['steps_max'] = template.steps.max
    if template.max_weights.max < INF:
        conditions.append('weights <= :weights_max')
        params['weights_max'] = template.max_weights.max
    if template.num_layers.min > 0:
        conditions.append('num_layers >= :num_layers_min')
        params['num_layers_min'] = template.num_layers.min
    if template.num_layers.max < INF:
        conditions.append('num_layers <= :num_layers_max')
        params['num_layers_max'] = template.num_layers.max
    if len(template.precision) != len(Precision):
        precisions_list = ', '.join(f'"{p}"' for p in template.precision)
        conditions.append(f'precision IN ({precisions_list})')
    clause = _decay_type_clause(template.decay_types)
    if clause is not None:
        conditions.append(clause)
    return conditions, params


def _decay_type_clause(decay_types: Iterable[DecayType]) -> Optional[str]:
    """Minimal SQL clause selecting confs matching `decay_types`, or
    None when the clause is tautological (all three types). Simplified
    per case using the validity invariant `cosine=1 ⇒ decay=1`.
    """
    s = set(decay_types)
    if s == set(DecayType):
        return None  # tautological — skip the clause
    if not s:
        return '0'  # match nothing
    # Singletons.
    if s == {DecayType.NONE}:
        return '(cosine = 0 AND decay = 1)'
    if s == {DecayType.EXP}:
        return '(cosine = 0 AND decay < 1)'
    if s == {DecayType.COSINE}:
        return '(cosine = 1)'
    # Pairs.
    if s == {DecayType.NONE, DecayType.EXP}:
        # cosine=1 implies decay=1, so cosine=0 ⇔ NONE ∪ EXP.
        return '(cosine = 0)'
    if s == {DecayType.NONE, DecayType.COSINE}:
        # NONE has decay=1, COSINE implies decay=1.
        return '(decay = 1)'
    if s == {DecayType.EXP, DecayType.COSINE}:
        # Excludes only NONE = (cosine=0 AND decay=1).
        return '(cosine = 1 OR decay < 1)'
    raise ValueError(f"unhandled decay_types subset: {s!r}")


def _regexp(pattern, input_string):
    if input_string is None or pattern is None:
        return False
    return re.fullmatch(pattern, input_string) is not None


# --- SQL shared by both sides -----------------------------------------------


FIND_CONF = """
SELECT id
FROM conf
WHERE spec = :spec
    AND lr = :lr
    AND length = :length
    AND batch = :batch
    AND steps = :steps
    AND precision = :precision
    AND decay = :decay
    AND cosine = :cosine
"""


# --- connection bootstrap ---------------------------------------------------


_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')


def _maybe_add_num_layers_column(db) -> None:
    """Lazy ALTER TABLE for the num_layers column added after the
    initial schema. NULL for old rows; populate via the
    db-backfill-num-layers CLI command."""
    cur = db.execute("PRAGMA table_info(conf)")
    columns = {row['name'] for row in cur.fetchall()}
    if 'num_layers' not in columns:
        db.execute("ALTER TABLE conf ADD COLUMN num_layers INTEGER")
        db.commit()


def _maybe_add_pick_me_column(db) -> None:
    """Lazy ALTER TABLE for the pick_me column. Default 0 backfills
    existing rows so the search treats them as non-priority."""
    cur = db.execute("PRAGMA table_info(conf)")
    columns = {row['name'] for row in cur.fetchall()}
    if 'pick_me' not in columns:
        db.execute(
            "ALTER TABLE conf "
            "ADD COLUMN pick_me INTEGER NOT NULL DEFAULT 0")
        db.commit()


def _maybe_create_perf_indexes(db) -> None:
    """Lazy CREATE INDEX for the two cte indexes added to speed up
    `confs_under_time`, the writer's TOP_CONF_AT query, and
    `top_confs_global`. Idempotent: ANALYZE only runs the first time
    an index is actually created so subsequent connects are cheap.
    """
    existing = {row['name'] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    created = False
    if 'conf_time_estimate_system_time' not in existing:
        db.execute(
            'CREATE INDEX conf_time_estimate_system_time '
            'ON conf_time_estimate(system, time_s)')
        created = True
    if 'conf_time_estimate_conf_time' not in existing:
        db.execute(
            'CREATE INDEX conf_time_estimate_conf_time '
            'ON conf_time_estimate(conf_id, time_s)')
        created = True
    if 'conf_weights_score' not in existing:
        db.execute(
            'CREATE INDEX conf_weights_score '
            'ON conf(weights, median_score)')
        created = True
    if created:
        db.commit()
        # Stats refresh so the planner picks the new indexes.
        db.execute('ANALYZE')
        db.commit()


# --- conf changelog (feeds `ConfIndex`) -------------------------------------


# Every write to `conf` or `conf_time_estimate` logs the touched conf
# id (plus the system, for estimate rows) here, from triggers inside
# the DB itself. `ConfIndex` (conf_index.py) keeps an in-memory copy of
# those two tables and catches up by re-reading exactly the logged
# rows; because the log is written by SQLite rather than by our write
# paths, no writer -- the server, an admin CLI, a migration script, a
# hand-typed UPDATE -- can change an indexed row without the index
# hearing about it. Rows are only ever pruned from the bottom
# (`DbWriter.prune_conf_change`), and the prune floor is recorded in
# `conf_change_floor` so a consumer that fell behind it knows it
# missed something and rebuilds instead.
#
# The BEFORE triggers on `conf` exist for conflict-resolution deletes:
# `INSERT OR REPLACE` / `UPDATE OR REPLACE` silently delete the row
# they collide with (on the id or on the UNIQUE key), and DELETE
# triggers don't fire for those unless `recursive_triggers` is on --
# a per-connection setting a raw script won't have. Logging every
# would-be collider up front covers that on any connection. (For
# `conf_time_estimate` the only UNIQUE key is (conf_id, system), so a
# REPLACE there evicts the same pair the AFTER INSERT trigger logs.)
CONF_CHANGE_TRIGGERS = (
    'conf_change_conf_before_insert',
    'conf_change_conf_insert',
    'conf_change_conf_before_update',
    'conf_change_conf_update',
    'conf_change_conf_delete',
    'conf_change_cte_insert',
    'conf_change_cte_update',
    'conf_change_cte_delete',
)

CONF_CHANGE_DDL = """
CREATE TABLE IF NOT EXISTS conf_change (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conf_id INTEGER NOT NULL,
    -- NULL: the conf row itself changed (re-read it and all its
    -- estimates). Non-NULL: only the (conf_id, system) estimate did.
    system TEXT
);

-- One row. `floor`: every seq <= floor has been pruned. `epoch`:
-- random at creation, so a dropped-and-recreated changelog (whose
-- AUTOINCREMENT restarts) can't be mistaken for the old one.
CREATE TABLE IF NOT EXISTS conf_change_floor (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    floor INTEGER NOT NULL,
    epoch TEXT NOT NULL
);
INSERT OR IGNORE INTO conf_change_floor (id, floor, epoch)
    VALUES (0, 0, lower(hex(randomblob(8))));

CREATE TRIGGER IF NOT EXISTS conf_change_conf_before_insert
BEFORE INSERT ON conf BEGIN
    INSERT INTO conf_change (conf_id)
        SELECT id FROM conf WHERE id = NEW.id;
    INSERT INTO conf_change (conf_id)
        SELECT id FROM conf
        WHERE spec = NEW.spec AND precision = NEW.precision
          AND lr = NEW.lr AND decay = NEW.decay AND cosine = NEW.cosine
          AND length = NEW.length AND batch = NEW.batch
          AND steps = NEW.steps;
END;

CREATE TRIGGER IF NOT EXISTS conf_change_conf_insert
AFTER INSERT ON conf BEGIN
    INSERT INTO conf_change (conf_id) VALUES (NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS conf_change_conf_before_update
BEFORE UPDATE ON conf BEGIN
    INSERT INTO conf_change (conf_id)
        SELECT id FROM conf WHERE id = NEW.id AND id != OLD.id;
    INSERT INTO conf_change (conf_id)
        SELECT id FROM conf
        WHERE spec = NEW.spec AND precision = NEW.precision
          AND lr = NEW.lr AND decay = NEW.decay AND cosine = NEW.cosine
          AND length = NEW.length AND batch = NEW.batch
          AND steps = NEW.steps AND id != OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS conf_change_conf_update
AFTER UPDATE ON conf BEGIN
    INSERT INTO conf_change (conf_id) VALUES (OLD.id);
    INSERT INTO conf_change (conf_id)
        SELECT NEW.id WHERE NEW.id IS NOT OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS conf_change_conf_delete
AFTER DELETE ON conf BEGIN
    INSERT INTO conf_change (conf_id) VALUES (OLD.id);
END;

CREATE TRIGGER IF NOT EXISTS conf_change_cte_insert
AFTER INSERT ON conf_time_estimate BEGIN
    INSERT INTO conf_change (conf_id, system)
        VALUES (NEW.conf_id, NEW.system);
END;

CREATE TRIGGER IF NOT EXISTS conf_change_cte_update
AFTER UPDATE ON conf_time_estimate BEGIN
    INSERT INTO conf_change (conf_id, system)
        VALUES (OLD.conf_id, OLD.system);
    INSERT INTO conf_change (conf_id, system)
        SELECT NEW.conf_id, NEW.system
        WHERE NEW.conf_id IS NOT OLD.conf_id
           OR NEW.system IS NOT OLD.system;
END;

CREATE TRIGGER IF NOT EXISTS conf_change_cte_delete
AFTER DELETE ON conf_time_estimate BEGIN
    INSERT INTO conf_change (conf_id, system)
        VALUES (OLD.conf_id, OLD.system);
END;
"""


def _maybe_create_conf_change(db) -> None:
    """Create the changelog tables + triggers if any piece is missing.

    Checked first so the common case (everything present) costs one
    sqlite_master read and no write transaction -- every `DbWriter`
    open goes through here, including short-lived admin CLIs that may
    run next to a live server. Adding the triggers to an existing DB
    rewrites no data, so it is instant even on the full results DB.
    The DDL is all IF NOT EXISTS, so a partial earlier state (a
    trigger dropped by hand) is simply completed.
    """
    names = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')")}
    wanted = {'conf_change', 'conf_change_floor', *CONF_CHANGE_TRIGGERS}
    if wanted <= names:
        return
    db.executescript(CONF_CHANGE_DDL)
    db.commit()


def open_connection(
    path: Optional[str], readonly: bool,
) -> sqlite3.Connection:
    """Open and configure a sqlite3 connection for results.db.

    For writers, applies the schema on a fresh DB and enables WAL +
    a generous busy_timeout so multiple writer connections in the same
    process can serialize cleanly. In-memory DBs skip WAL but keep the
    timeout. Read-only callers must point at a real (or shared-cache)
    file — sqlite refuses `mode=ro` on a plain `:memory:` DB.

    Connections inherit sqlite's default `check_same_thread=True`:
    every consumer (WriterThread, SearchThread's reader, ModelThread's
    reader, per-request handler readers, the CLIs) opens its
    connection on the thread that uses it.
    """
    if path is None:
        path = ':memory:'

    is_uri = path.startswith('file:')
    is_memory = path == ':memory:' or (is_uri and 'mode=memory' in path)

    if not is_memory:
        logging.info(f'Connecting to results DB {path}')

    if readonly:
        assert not is_memory, "Can't open in-memory database read-only"
        db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    else:
        db = sqlite3.connect(path, uri=is_uri)
    db.row_factory = sqlite3.Row
    db.create_function('REGEXP', 2, _regexp)

    # Apply schema if no tables exist yet. This is true on a fresh
    # file DB, on a fresh :memory: DB, and on the first connection
    # to a shared in-memory DB; subsequent connections to the same
    # shared DB skip this step.
    cur = db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='conf'"
    )
    if cur.fetchone() is None:
        assert not readonly
        with open(_SCHEMA_PATH) as schema:
            db.executescript(schema.read())
            db.commit()
        _maybe_create_conf_change(db)
    elif not readonly:
        # Existing DB -- run any additive column migrations.
        _maybe_add_num_layers_column(db)
        _maybe_add_pick_me_column(db)
        _maybe_create_perf_indexes(db)
        _maybe_create_conf_change(db)

    # Enable WAL for file-backed DBs so the timing thread and search
    # thread can write concurrently without blocking readers.
    if not is_memory and not readonly:
        db.execute('PRAGMA journal_mode=WAL')

    # Two writers (search thread + model thread) still serialize on the
    # write lock. Without a busy_timeout, the loser of a race hits
    # SQLITE_BUSY immediately. 30 s is generous enough to absorb large
    # batched upserts from the model thread.
    db.execute('PRAGMA busy_timeout = 30000')

    return db
