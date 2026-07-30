"""Generate a "bucket" tokenset: capswords2 + top-(N-1) bytes + one
uniform catch-all token. tokens/tokens{N}_bucket.json, type "fold".

The design (2026-07-30): strictly one token per processed byte -- the
hypothesis is that tiny models prefer a uniform single-byte stream
over hexbpe's mixed granularity (nibble escapes, merges). The N-1
most common bytes of the capswords2-processed corpus get their own
tokens; every other byte folds into one bucket token. The bucket is
UNIFORM: no stored probabilities, so the fold machinery charges its
members log2(group size) bits and the set costs only its N-1
selection weights (stats.extra_weights). Lossy by construction; the
loss is priced by stats.residual_bits_per_byte, expressed per RAW
byte because eval divides by raw sample length.

Decode prints the bucket's head character for bucket tokens (chosen
as the most frequent printable-ASCII bucket member). A speculative
improvement, not implemented: a bucket token mid-word in a
non-all-caps word could be decoded as the \x17 upper marker.

Inputs: freq_capswords2.txt (full processed-corpus byte counts) and
freq.txt (raw counts, for the raw byte total) -- both produced by the
Rust `count-bytes` command; regenerate the first with:

    texmo.exe process -d data/books3.txt -o data/books3_capswords2.txt \
        --processing caps-words2
    texmo.exe count-bytes -d data/books3_capswords2.txt > freq_capswords2.txt

Usage: uv run python scripts/make_bucket.py [ntokens]
"""
import math
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.pjson import save_json

_ESCAPES = {"t": 9, "n": 10, "r": 13, "'": 39, "\\": 92, "0": 0}


def _parse_freq(path: str) -> dict[int, int]:
    """Parse the Rust count-bytes output (first section only): lines
    are `'c' N` for bytes < 128 (char debug form) and `NNN N` above."""
    counts: dict[int, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                break  # the sorted second section repeats the data
            repr_part, count = line.rsplit(" ", 1)
            if repr_part.startswith("'"):
                body = repr_part[1:-1]
                if body.startswith("\\u{"):
                    byte = int(body[3:-1], 16)
                elif body.startswith("\\"):
                    byte = _ESCAPES[body[1]]
                else:
                    byte = ord(body)
            else:
                byte = int(repr_part)
            assert 0 <= byte < 256 and byte not in counts, line
            counts[byte] = int(count)
    return counts


def main():
    ntokens = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    counts = _parse_freq("freq_capswords2.txt")
    raw_total = sum(_parse_freq("freq.txt").values())
    proc_total = sum(counts.values())

    ranked = sorted(counts, key=lambda b: (-counts[b], b))
    selected = sorted(ranked[: ntokens - 1])
    bucket = [b for b in range(256) if b not in set(selected)]
    bucket_count = sum(counts.get(b, 0) for b in bucket)

    # Decode face of the bucket: most frequent printable-ASCII member
    # (FoldTokenizer looks tokens up by their UTF-8 string, so the
    # head must be valid standalone UTF-8 anyway).
    head = max((b for b in bucket if 33 <= b < 127),
               key=lambda b: counts.get(b, 0))

    # Uniform group: every member charged log2(group size), exactly
    # what FoldTokenizer computes for a group without head_freq. Per
    # RAW byte: bucket events live in the processed stream, so the
    # sum is normalized by the raw length, not the processed one.
    residual = bucket_count * math.log2(len(bucket)) / raw_total
    # For the record: what a fully stored (add-1 smoothed) bucket
    # table would buy, at +len(seen) weights.
    seen = [b for b in bucket if counts.get(b, 0)]
    m = bucket_count + len(bucket)
    table_residual = sum(
        counts[b] * -math.log2((counts[b] + 1) / m) for b in seen
    ) / raw_total

    def enc(b: int):
        return chr(b) if b < 128 else [b]

    tokens = [enc(b) for b in selected] + [chr(head)]
    sequences = [
        {"string": enc(b), "tokens": [chr(head)]}
        for b in bucket if b != head
    ]

    doc = {
        "processing": "capswords2",
        "type": "fold",
        "tokens": tokens,
        "sequences": sequences,
        "stats": {
            "ntokens": ntokens,
            # RAW bytes per token; one token per processed byte.
            "bytes_per_token": raw_total / proc_total,
            # N-1 byte selections at one weight each; the uniform
            # bucket stores nothing.
            "extra_weights": ntokens - 1,
            "residual_bits_per_byte": round(residual, 6),
            "raw_bytes": raw_total,
            "scanned_bytes": proc_total,
            "total_tokens": proc_total,
            "initial_size": raw_total,
        },
    }
    out = f"tokens/tokens{ntokens}_bucket.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        save_json(doc, f)

    show = "".join(
        chr(b) if 32 < b < 127 else (" " if b == 32 else f"<{b:02x}>")
        for b in selected)
    print(f"wrote {out}")
    print(f"  selected {len(selected)}: {show}")
    print(f"  bucket: {len(bucket)} byte values ({len(seen)} seen), "
          f"head {chr(head)!r}, {bucket_count / proc_total:.4%} of stream")
    print(f"  bytes_per_token {raw_total / proc_total:.4f}   "
          f"extra_weights {ntokens - 1}")
    print(f"  residual {residual:.4f} b/rawB uniform "
          f"(stored table would give {table_residual:.4f} "
          f"at +{len(seen)} weights)")


if __name__ == "__main__":
    main()
