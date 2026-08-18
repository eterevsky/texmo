"""Render recorded dialog JSONL as plain readable text.

The companion to `scripts/dialog_harness.py`: that one writes one JSON
record per dialog, which is the right shape for processing and the
wrong shape for reading. This prints the conversations the way a
person wants to see them -- a `====` rule between dialogs, then each
turn as the speaker's name, the utterance, and a blank line:

    ====
    child
    Hi! Can I ask you a question?

    teacher
    Of course! What would you like to know?

    ====

    uv run python scripts/dialog_text.py data/dialogs/run1.jsonl \\
        [more.jsonl ...] [--out dialogs.txt]

Several inputs concatenate in order. Output goes to stdout by default,
so it pipes and redirects; `--out` writes a UTF-8 file.

Speaker names come from each record's own `participants` list, so a
file holding dialogs from different experiments still labels every
turn correctly. A record whose participants are missing or malformed
falls back to `bot0` / `bot1`.

Reading is deliberately forgiving: a line that is not valid JSON is
reported on stderr and skipped, because half of a truncated file is
still worth reading -- exactly the case of a run killed mid-write.
"""
import argparse
import io
import json
import os
import sys

SEPARATOR = "===="


def speaker_name(record: dict, speaker) -> str:
    """The display name for a turn's speaker.

    `bot{i}` whenever the record cannot answer: no participants, a
    short list, a non-object entry, or no `name`.
    """
    participants = record.get("participants")
    if isinstance(participants, list) and isinstance(speaker, int):
        if 0 <= speaker < len(participants):
            participant = participants[speaker]
            if isinstance(participant, dict):
                name = participant.get("name")
                if isinstance(name, str) and name:
                    return name
    return f"bot{speaker}"


def render_dialog(record: dict) -> str:
    """One dialog: the rule, then `name / text / blank` per turn."""
    lines = [SEPARATOR]
    turns = record.get("turns")
    if not isinstance(turns, list):
        turns = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        lines.append(speaker_name(record, turn.get("speaker")))
        lines.append(str(turn.get("text", "")))
        lines.append("")
    return "\n".join(lines) + "\n"


def render(records) -> str:
    """Every dialog, with a closing rule after the last one."""
    return "".join(render_dialog(r) for r in records) + SEPARATOR + "\n"


def read_records(path: str, warn=None):
    """Yield the records in one JSONL file, skipping unreadable lines.

    Blank lines are ignored silently; anything else that will not
    parse into a JSON object is reported through `warn` (stderr by
    default) and skipped.
    """
    if warn is None:
        warn = _warn
    with open(path, encoding="utf-8", errors="replace") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                warn(f"{path}:{number}: skipping unparseable line ({e.msg})")
                continue
            if not isinstance(record, dict):
                warn(f"{path}:{number}: skipping line, not a JSON object")
                continue
            yield record


def _warn(message: str):
    print(f"dialog_text: {message}", file=sys.stderr)


def write_text(text: str, out_path: str | None):
    """`--out` as UTF-8, else stdout.

    Model output is full of curly quotes and dashes, and the Windows
    console encoding is not UTF-8, so stdout is wrapped rather than
    trusted -- with `errors="replace"`, since losing a quote character
    beats aborting the whole dump.
    """
    if out_path:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        # Already a text stream (pytest capture); write straight to it.
        sys.stdout.write(text)
        return
    stream = io.TextIOWrapper(
        buffer, encoding="utf-8", errors="replace", newline="")
    try:
        stream.write(text)
        stream.flush()
    finally:
        # Detach, or closing the wrapper would close stdout itself.
        stream.detach()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render dialog JSONL as readable text.")
    parser.add_argument("inputs", nargs="+", help="dialog JSONL file(s)")
    parser.add_argument(
        "--out", default=None, help="output text file (default stdout)")
    args = parser.parse_args(argv)

    records = []
    for path in args.inputs:
        records.extend(read_records(path))
    write_text(render(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
