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

Rows whose `dialogue` and `speakers` lists disagree in length, or
that hold an empty or blank utterance, are dropped and counted.
"""
import os
import re
import sys

import pyarrow.parquet as pq

DATA_DIR = "data/soda"
SPLITS = ("train", "valid", "test")
COLUMNS = ["dialogue", "speakers"]
BATCH_SIZE = 4096
PROGRESS_EVERY = 100_000

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
        self.collapsed = 0


def clean_utterance(text: str, stats: Stats) -> str:
    """Strip and flatten one utterance; "" if nothing survives."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in text:
        stats.collapsed += 1
        text = NEWLINE_RUN.sub(" ", text)
    return text


def render_dialog(dialogue, speakers, stats: Stats) -> str | None:
    """`Name: text` per utterance joined by blank lines, or None.

    None means the row is unusable; the reason is already counted.
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
    turns = []
    for speaker, utterance in zip(speakers, dialogue):
        if speaker is None or utterance is None:
            stats.skip_empty_utterance += 1
            return None
        text = clean_utterance(utterance, stats)
        if not text:
            stats.skip_empty_utterance += 1
            return None
        turns.append(f"{speaker}: {text}")
    return "\n\n".join(turns)


def convert(in_path: str, out_path: str) -> Stats:
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
                text = render_dialog(dialogue, speakers, stats)
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
    for split in SPLITS:
        in_path = os.path.join(DATA_DIR, f"{split}.parquet")
        out_path = os.path.join(DATA_DIR, f"soda_{split}.txt")
        print(f"{in_path} -> {out_path}")
        stats = convert(in_path, out_path)
        size = os.path.getsize(out_path)
        print(f"  rows {stats.rows}, dialogs {stats.dialogs}, "
              f"bytes {size}")
        print(f"  skipped: length mismatch {stats.skip_mismatch}, "
              f"empty dialogue {stats.skip_empty_dialog}, "
              f"empty utterance {stats.skip_empty_utterance}")
        print(f"  newline collapses: {stats.collapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
