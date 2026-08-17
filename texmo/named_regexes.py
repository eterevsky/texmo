"""Named spec-regex presets for the search UI.

The sub-search regexes worth keeping are long and fiddly (a
transformer-shaped pattern runs well past 80 characters), and the live
template state is rebuilt on every `/update` -- so a regex that took
real effort to get right lives exactly as long as the browser tab
does. This module gives those regexes a name and a home.

The store is a hand-edited JSON file in the repo top directory, next
to `config.py` (per-host personal state, not repo content):

    {
      "transformers": {
        "regex": ".*\\\\|(norm-)?attn\\\\..*",
        "default_spec": "bytes|attn.4.1"
      },
      "recurrent": {"regex": ".*\\\\|.*(gru|lstm|rnn)\\\\..*"}
    }

`default_spec` is optional. The server only ever reads this file, and
re-reads it on every page render, so an edit shows up on refresh with
no restart.

Because a human types it, a malformed entry must never take the page
down: `load` skips whatever it cannot use, logs which entry and why,
and returns the survivors. An unreadable file is simply no presets.

The name **Unrestricted** is reserved: the UI always offers it first
as the virtual "no filter" choice, so a stored entry by that name
could never be selected and is skipped.
"""

import json
import logging
import os
import re

# Relative to the process's working directory -- the server runs from
# the repo root, the same convention `config.py`, `tokens/` and
# `results/` already follow.
PATH = 'named_regexes.json'

# Offered first in every preset dropdown, meaning "no spec filter".
# Virtual: never stored, never editable.
RESERVED_NAME = 'Unrestricted'


def _clean_entry(name: str, body) -> dict | None:
    """Validate one raw `name -> body` pair into a preset dict.

    Returns None (after logging why) for anything unusable. Only what
    would break rendering or mislead the reader is checked here: a
    `default_spec` that doesn't parse is left for `/update` to reject,
    where the error banner can say so in context.
    """
    if not isinstance(body, dict):
        logging.warning(
            f"named regex {name!r}: expected an object, got "
            f"{type(body).__name__}; skipping")
        return None
    regex = body.get('regex')
    if not isinstance(regex, str) or not regex.strip():
        logging.warning(f"named regex {name!r}: no 'regex' value; skipping")
        return None
    try:
        re.compile(regex)
    except re.error as e:
        # The DB's REGEXP bridge compiles with the same `re`, so a
        # pattern that fails here could never match a spec either.
        logging.warning(f"named regex {name!r}: {e}; skipping")
        return None
    default_spec = body.get('default_spec') or ''
    if not isinstance(default_spec, str):
        logging.warning(
            f"named regex {name!r}: 'default_spec' is not a string; "
            f"ignoring it")
        default_spec = ''
    return {
        'name': name,
        'regex': regex,
        'default_spec': default_spec.strip(),
    }


def load(path: str | None = None) -> list[dict]:
    """Read the preset file into `{name, regex, default_spec}` dicts,
    sorted by name (case-insensitively).

    Never raises and never writes: a missing file is no presets, and
    so is a file that doesn't parse."""
    path = path or PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        logging.warning(f"Ignoring {path}: {e}")
        return []
    if not isinstance(raw, dict):
        logging.warning(
            f"Ignoring {path}: expected an object of "
            f"name -> {{regex, default_spec}}")
        return []

    out: dict[str, dict] = {}
    for raw_name, body in raw.items():
        name = str(raw_name).strip()
        if not name:
            logging.warning("named regex with a blank name; skipping")
            continue
        if name.lower() == RESERVED_NAME.lower():
            logging.warning(
                f"named regex {name!r} collides with the reserved "
                f"{RESERVED_NAME!r} entry; skipping")
            continue
        if name in out:
            logging.warning(f"duplicate named regex {name!r}; skipping")
            continue
        entry = _clean_entry(name, body)
        if entry is not None:
            out[name] = entry
    return sorted(out.values(), key=lambda e: (e['name'].lower(), e['name']))
