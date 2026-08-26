"""Pick the scripted-examiner eval's seed dialogs out of SODA valid.

Writes `data/eval/seeds.jsonl`: the first 1000 dialogs of
`data/soda/valid.parquet`, in file order, that have exactly two
distinct speakers and at least four utterances. One compact JSON
object per line,

    {"idx": 0, "opener": "Hey, how's it going?",
     "script": ["Hey, how's it going?", "Doing well, ...", ...],
     "speakers": ["User", "Bot", ...],
     "n_utterances": 6}

`idx` counts the seeds written (0..999), not the parquet rows read.
`opener` is `script[0]` verbatim -- the utterance the eval forces out
of the examiner as the dialog's first turn (docs/roadmap.md,
milestone 2). `script` is the full `dialogue` list, verbatim.
`speakers` runs parallel to it: `speakers[i]` labels `script[i]`, and
the two SODA names are replaced by "User" and "Bot" in first-
appearance order -- the same mapping `make_soda_txt.py --names
user-bot` applies to the training corpus. Since `speakers[i]` speaks
`dialogue[i]`, the opener's side is always "User", so
`speakers[0] == "User"` holds for every seed.

Attribution is spelled out rather than left to turn parity because
SODA speakers do *not* strictly alternate: ~3% of two-speaker dialogs
have a side taking two turns in a row, so parity would mislabel them.

The quick subsets in the eval design are prefixes of this file by
convention -- `head -10` is the smoke set, `head -100` ranks models --
so there are no separate 10/100 files to keep in sync.

pyarrow is not a texmo dependency and must not become one, so run
this with a throwaway environment:

    uv run --with pyarrow python scripts/make_eval_seeds.py \\
        [--seeds 1000] [--min-utterances 4]

Rows are skipped, and counted, when `dialogue` and `speakers` disagree
in length, when the speaker count is not two, when the dialog is too
short, or when an utterance is blank or spans several lines (the
`Name: utterance` corpus format keeps one utterance on one line).
"""
import argparse
import json
import os
import sys

import pyarrow.parquet as pq

IN_PATH = "data/soda/valid.parquet"
OUT_PATH = "data/eval/seeds.jsonl"
COLUMNS = ["dialogue", "speakers"]
BATCH_SIZE = 4096
USER_BOT = ("User", "Bot")
NUM_SPEAKERS = len(USER_BOT)


class Stats:
    """Tallies for the run's summary lines."""

    def __init__(self):
        self.rows = 0
        self.seeds = 0
        self.skip_mismatch = 0
        self.skip_speakers = 0
        self.skip_short = 0
        self.skip_utterance = 0


def qualifies(dialogue, speakers, min_utterances: int,
              stats: Stats) -> bool:
    """True if the row is usable as a seed; else counts the reason."""
    if dialogue is None or speakers is None:
        stats.skip_mismatch += 1
        return False
    if len(dialogue) != len(speakers):
        stats.skip_mismatch += 1
        return False
    if any(speaker is None for speaker in speakers):
        stats.skip_mismatch += 1
        return False
    if len(set(speakers)) != NUM_SPEAKERS:
        stats.skip_speakers += 1
        return False
    if len(dialogue) < min_utterances:
        stats.skip_short += 1
        return False
    for utterance in dialogue:
        if utterance is None or not utterance.strip():
            stats.skip_utterance += 1
            return False
        # Kept verbatim below, so anything that would not survive the
        # one-line `Name: utterance` format has to go now.
        if "\n" in utterance or "\r" in utterance:
            stats.skip_utterance += 1
            return False
    return True


def user_bot_names(speakers) -> list[str]:
    """`speakers` with the two SODA names replaced by User/Bot.

    dict.fromkeys de-duplicates in first-appearance order -- a set
    would lose the order -- so the opener's side becomes "User", the
    same mapping make_soda_txt.py applies to the training corpus.
    """
    rename = dict(zip(dict.fromkeys(speakers), USER_BOT))
    return [rename[speaker] for speaker in speakers]


def collect(in_path: str, out_path: str, num_seeds: int,
            min_utterances: int) -> tuple[Stats, list[int]]:
    """Stream the split until `num_seeds` seeds are written.

    Returns the tallies and the opener lengths, in seed order.
    """
    stats = Stats()
    opener_lengths = []
    parquet = pq.ParquetFile(in_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for batch in parquet.iter_batches(
                batch_size=BATCH_SIZE, columns=COLUMNS):
            dialogues = batch.column("dialogue").to_pylist()
            speaker_lists = batch.column("speakers").to_pylist()
            for dialogue, speakers in zip(dialogues, speaker_lists):
                stats.rows += 1
                if not qualifies(dialogue, speakers, min_utterances,
                                 stats):
                    continue
                names = user_bot_names(speakers)
                assert names[0] == USER_BOT[0], (
                    f"seed {stats.seeds} does not open on {USER_BOT[0]}")
                seed = {
                    "idx": stats.seeds,
                    "opener": dialogue[0],
                    "script": list(dialogue),
                    "speakers": names,
                    "n_utterances": len(dialogue),
                }
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")
                opener_lengths.append(len(dialogue[0]))
                stats.seeds += 1
                if stats.seeds == num_seeds:
                    return stats, opener_lengths
    return stats, opener_lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n", "--seeds", type=int, default=1000,
        help="how many seed dialogs to write (default: 1000)")
    parser.add_argument(
        "-m", "--min-utterances", type=int, default=4,
        help="shortest dialog accepted as a seed (default: 4)")
    parser.add_argument(
        "-o", "--output", default=OUT_PATH,
        help=f"seed file to write (default: {OUT_PATH})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"{IN_PATH} -> {args.output}")
    stats, lengths = collect(
        IN_PATH, args.output, args.seeds, args.min_utterances)
    if stats.seeds < args.seeds:
        print(f"  WARNING: only {stats.seeds} seeds; the split ran out",
              file=sys.stderr)

    size = os.path.getsize(args.output)
    print(f"  rows read {stats.rows}, seeds {stats.seeds}, bytes {size}")
    print(f"  skipped: length mismatch {stats.skip_mismatch}, "
          f"speaker count != {NUM_SPEAKERS} {stats.skip_speakers}, "
          f"under {args.min_utterances} utterances {stats.skip_short}, "
          f"bad utterance {stats.skip_utterance}")
    if lengths:
        ordered = sorted(lengths)
        print(f"  opener chars: min {ordered[0]}, "
              f"median {ordered[len(ordered) // 2]}, "
              f"mean {sum(ordered) / len(ordered):.1f}, "
              f"p90 {ordered[int(len(ordered) * 0.9)]}, "
              f"max {ordered[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
