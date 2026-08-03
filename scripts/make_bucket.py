"""Generate a "bucket" tokenset: top-(N-1) bytes + one uniform
catch-all token, over either the raw or the capswords2-processed
corpus. Emits a fold-type set (`FoldTokenizer` runs it as-is).

Two modes, selected by --processing:

- `raw` (tokens.{N}.fold.json, e.g. tokens.128.fold): the N-1 most
  common RAW bytes get their own tokens, everything else folds into
  one uniform catch-all. The simplest possible lossy set: 1 token per
  byte, bytes_per_token exactly 1.0. Reads freq.txt.
- `capswords2` (tokens.{N}.bucket.json): same design over the
  processed stream. This variant lost the 64-token design review to
  tokens.64.shift (2026-07-31) and no set of it is currently wired
  into search; the mode remains for future experiments. Reads
  freq_capswords2.txt (plus freq.txt for the raw byte total).

The bucket is UNIFORM: no stored probabilities, so the fold
machinery charges its members log2(group size) bits and the set
costs only its N-1 selection weights (stats.extra_weights). The
residual is expressed per RAW byte because eval divides by raw
sample length.

Frequency tables are produced by the Rust `count-bytes` command; see
freq_capswords2.txt regeneration notes in the repo docs.

Overwriting an existing set needs --force: in particular
tokens.32.fold.json is the CURATED letter-folding set
(scripts/make_fold32.py), not a product of this generator -- raw
mode at N=32 would silently replace it with a different design.

Usage: uv run python scripts/make_bucket.py [ntokens] [--processing raw]
"""
import argparse
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
                break  # a sorted second section may repeat the data
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntokens", type=int, nargs="?", default=64)
    parser.add_argument(
        "--processing", choices=("raw", "capswords2"), default="capswords2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ntokens = args.ntokens

    if args.processing == "raw":
        counts = _parse_freq("freq.txt")
        raw_total = proc_total = sum(counts.values())
        variation = "fold"
    else:
        counts = _parse_freq("freq_capswords2.txt")
        raw_total = sum(_parse_freq("freq.txt").values())
        proc_total = sum(counts.values())
        variation = "bucket"

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
    # RAW byte: bucket events live in the (possibly processed)
    # stream, so the sum is normalized by the raw length.
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
        "processing": args.processing,
        "type": "fold",
        "tokens": tokens,
        "sequences": sequences,
        "stats": {
            "ntokens": ntokens,
            # RAW bytes per token; one token per (processed) byte.
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
    out = f"tokens/tokens.{ntokens}.{variation}.json"
    if os.path.exists(out) and not args.force:
        raise SystemExit(
            f"{out} exists; pass --force to overwrite (see the "
            f"tokens.32.fold.json warning in the docstring)")
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
