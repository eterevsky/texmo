"""Convert the SODA dialog dataset (parquet) to plain-text corpora.

Reads `data/soda/{train,valid,test}.parquet` and writes
`data/soda/soda_{train,valid,test}.txt`. One dialog looks like

    Veda: Hi, Father. I've been having some doubts lately.

    Priest: What kind of doubts?

    Veda: About my faith.

-- `speakers[i]` verbatim, a colon, the utterance; utterances
separated by one blank line, dialogs by two.

pyarrow is not a texmo dependency and must not become one, so run
this with a throwaway environment:

    uv run --with pyarrow python scripts/make_soda_txt.py
    uv run --with pyarrow python scripts/make_soda_txt.py \\
        --names user-bot [--splits valid,test]

`--names user-bot` writes `soda_{split}_ub.txt` instead: only dialogs
with exactly two distinct speakers survive (the rest are dropped and
counted separately), and the two names are replaced by "User" and
"Bot" in first-appearance order. Since `speakers[i]` speaks
`dialogue[i]`, `speakers[0]` is by definition the dialog's opener, so
the renamed corpus always opens on "User" -- which is the side the
scripted examiner impersonates (docs/roadmap.md, milestone 2). The
script asserts it.

Rows whose `dialogue` and `speakers` lists disagree in length, or
that hold an empty or blank utterance, are dropped and counted.
"""
import argparse
import os
import re
import sys

import pyarrow.parquet as pq

DATA_DIR = "data/soda"
SPLITS = ("train", "valid", "test")
COLUMNS = ["dialogue", "speakers"]
BATCH_SIZE = 4096
PROGRESS_EVERY = 100_000

# `--names` modes: the output suffix each one writes under. "original"
# keeps `speakers[i]` as-is; "user-bot" renames the two speakers.
SUFFIX = {"original": "", "user-bot": "_ub"}
USER_BOT = ("User", "Bot")

# A whitespace run containing a newline, collapsed to a single space:
# an utterance must stay on one line for `Name: text` to parse back.
NEWLINE_RUN = re.compile(r"\s*\n\s*")


class Stats:
    """Per-split tallies, printed as the split's summary line."""

    def __init__(self):
        self.dialogs = 0
        self.rows = 0
        self.skip_mismatch = 0
        self.skip_empty_dialog = 0
        self.skip_empty_utterance = 0
        self.skip_speakers = 0
        self.collapsed = 0
        self.user_opens = 0


def clean_utterance(text: str, stats: Stats) -> str:
    """Strip and flatten one utterance; "" if nothing survives."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in text:
        stats.collapsed += 1
        text = NEWLINE_RUN.sub(" ", text)
    return text


def speaker_map(speakers, stats: Stats) -> dict | None:
    """{original name: "User"/"Bot"}, or None if the row can't map.

    First appearance order, so `speakers[0]` -- the speaker of
    `dialogue[0]`, i.e. whoever opens the dialog -- becomes "User".
    """
    if any(speaker is None for speaker in speakers):
        stats.skip_empty_utterance += 1
        return None
    # dict.fromkeys de-duplicates while preserving first-appearance
    # order; a set would lose the order the mapping depends on.
    distinct = list(dict.fromkeys(speakers))
    if len(distinct) != len(USER_BOT):
        stats.skip_speakers += 1
        return None
    return dict(zip(distinct, USER_BOT))


def render_dialog(dialogue, speakers, stats: Stats,
                  user_bot: bool = False) -> str | None:
    """`Name: text` per utterance joined by blank lines, or None.

    None means the row is unusable; the reason is already counted.
    With `user_bot`, the two speaker names are replaced by "User" and
    "Bot" and rows without exactly two distinct speakers are dropped.
    """
    if dialogue is None or speakers is None:
        stats.skip_mismatch += 1
        return None
    if len(dialogue) != len(speakers):
        stats.skip_mismatch += 1
        return None
    if not dialogue:
        stats.skip_empty_dialog += 1
        return None
    rename = None
    if user_bot:
        rename = speaker_map(speakers, stats)
        if rename is None:
            return None
    turns = []
    for speaker, utterance in zip(speakers, dialogue):
        if speaker is None or utterance is None:
            stats.skip_empty_utterance += 1
            return None
        text = clean_utterance(utterance, stats)
        if not text:
            stats.skip_empty_utterance += 1
            return None
        turns.append(f"{rename[speaker] if rename else speaker}: {text}")
    if user_bot and turns[0].startswith(f"{USER_BOT[0]}: "):
        stats.user_opens += 1
    return "\n\n".join(turns)


def convert(in_path: str, out_path: str, user_bot: bool) -> Stats:
    """Stream one parquet split into one text file."""
    stats = Stats()
    parquet = pq.ParquetFile(in_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE, columns=COLUMNS):
            dialogues = batch.column("dialogue").to_pylist()
            speaker_lists = batch.column("speakers").to_pylist()
            for dialogue, speakers in zip(dialogues, speaker_lists):
                stats.rows += 1
                text = render_dialog(dialogue, speakers, stats, user_bot)
                if text is None:
                    continue
                if stats.dialogs:
                    f.write("\n\n\n")
                f.write(text)
                stats.dialogs += 1
                if stats.rows % PROGRESS_EVERY == 0:
                    print(f"  {stats.rows} rows...", file=sys.stderr)
        if stats.dialogs:
            f.write("\n")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--names", choices=sorted(SUFFIX), default="original",
        help="speaker names to write: 'original' keeps them (default), "
             "'user-bot' renames the two speakers to User/Bot and drops "
             "dialogs that don't have exactly two")
    parser.add_argument(
        "-s", "--splits", default=",".join(SPLITS),
        help=f"comma-separated splits to convert "
             f"(default: {','.join(SPLITS)})")
    args = parser.parse_args()

    user_bot = args.names == "user-bot"
    suffix = SUFFIX[args.names]
    for split in args.splits.split(","):
        in_path = os.path.join(DATA_DIR, f"{split}.parquet")
        out_path = os.path.join(DATA_DIR, f"soda_{split}{suffix}.txt")
        print(f"{in_path} -> {out_path}")
        stats = convert(in_path, out_path, user_bot)
        size = os.path.getsize(out_path)
        print(f"  rows {stats.rows}, dialogs {stats.dialogs}, "
              f"bytes {size}")
        print(f"  skipped: length mismatch {stats.skip_mismatch}, "
              f"empty dialogue {stats.skip_empty_dialog}, "
              f"empty utterance {stats.skip_empty_utterance}, "
              f"speaker count != 2 {stats.skip_speakers}")
        print(f"  newline collapses: {stats.collapsed}")
        if user_bot:
            print(f"  dialogs opening on {USER_BOT[0]}: {stats.user_opens}")
            assert stats.user_opens == stats.dialogs, (
                "a renamed dialog does not open on "
                f"{USER_BOT[0]}: {stats.user_opens} of {stats.dialogs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
