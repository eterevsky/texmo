"""Reformat JSON files in place with the repo's adaptive pretty-printer.

Used for the tokensets the Rust builder emits: `serde_json`'s pretty
printer puts every list element on its own line, which makes a 256-token
set unreadable. `texmo.pjson` keeps short values inline and expands only
what is long, matching the style of the hand-written sets.

    uv run python scripts/pjson_format.py tokens/tokens.32.hexbpe.json ...
"""
import json
import os
import sys

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.pjson import save_json


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        before = os.path.getsize(path)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            save_json(data, f)
        print(f"{path}: {before:,} -> {os.path.getsize(path):,} bytes")


if __name__ == "__main__":
    main()
