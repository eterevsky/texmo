"""Fold a text corpus through a fold tokenset's byte map.

A fold (forgetting) tokenset -- `tokens.32.fold` and friends -- maps
every one of the 256 bytes to exactly one token, and each token prints
back as its group's head byte. Composing the two gives a pure
byte -> byte function: 'A' -> 'a', '7' -> 'x', '\\t' -> ' ', and so on
(see docs/io.md, "Fold tokensets").

Applying that function to the corpus itself produces a corpus with the
information the fold destroys already gone. That is what makes a
comparison against a *lossless* tokenset fair: trained and evaluated on
the folded text, a `tokens.32.hexbpe` model is modelling exactly the
same information a `tokens.32.fold` model sees, so their bits/byte are
directly comparable with no residual charge to reconcile.

    uv run python scripts/fold_corpus.py \\
        data/soda/soda_valid.txt data/soda/soda_valid_folded.txt \\
        [--tokens tokens.32.fold] [--tokens-dir tokens]

The map is 1 byte -> 1 byte and stateless, so streaming in fixed-size
chunks is exact -- no chunk boundary can fall inside anything. Output
size always equals input size.
"""
import argparse
import os
import sys

import numpy as np

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.tokens import get_tokenizer, set_tokens_dir
from texmo.tokens.fold_tokenizer import FoldTokenizer

CHUNK = 64 << 20


def fold_table(tokens_name: str) -> np.ndarray:
    """The 256-entry byte -> byte lookup a fold tokenset induces."""
    tokenizer = get_tokenizer(tokens_name)
    if tokenizer is None:
        raise SystemExit(f"no tokenset '{tokens_name}'")
    if not isinstance(tokenizer, FoldTokenizer):
        raise SystemExit(
            f"'{tokens_name}' is not a fold tokenset "
            f"(type {tokenizer.tokenset.type!r}); only fold sets induce "
            f"a byte -> byte map")
    # A non-raw pipeline would insert case/word markers before the
    # fold, so the composition would no longer be per-byte.
    if tokenizer.tokenset.processing != "raw":
        raise SystemExit(
            f"'{tokens_name}' uses processing "
            f"{tokenizer.tokenset.processing!r}; the byte map is only "
            f"well defined for 'raw'")

    # Round-trip all 256 bytes through the tokenizer: tokenize maps
    # each to its group's token, untokenize prints the group's head.
    all_bytes = bytes(range(256))
    heads = tokenizer.untokenize(tokenizer.tokenize_processed(all_bytes))
    assert len(heads) == 256, "fold round-trip changed the length"
    table = np.frombuffer(heads, dtype=np.uint8).copy()
    # Folding is idempotent -- every head byte is its own group's head
    # -- so re-folding folded text is a no-op and the folded corpus
    # tokenizes to the same tokens the raw one did.
    assert (table[table] == table).all(), "fold map is not idempotent"
    return table


def fold_file(in_path: str, out_path: str, table: np.ndarray) -> int:
    """Stream one file through the byte map; returns bytes written."""
    written = 0
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        while True:
            chunk = fin.read(CHUNK)
            if not chunk:
                break
            folded = table[np.frombuffer(chunk, dtype=np.uint8)]
            fout.write(folded.tobytes())
            written += len(chunk)
            print(f"  {written >> 20} MiB...", file=sys.stderr, flush=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="corpus to read")
    parser.add_argument("output", help="folded corpus to write")
    parser.add_argument(
        "-t", "--tokens", default="tokens.32.fold",
        help="fold tokenset whose byte map to apply "
             "(default: tokens.32.fold)")
    parser.add_argument(
        "--tokens-dir", default="tokens",
        help="directory with token sets (default: 'tokens')")
    args = parser.parse_args()

    set_tokens_dir(args.tokens_dir)
    table = fold_table(args.tokens)
    kept = int((table == np.arange(256, dtype=np.uint8)).sum())
    print(f"{args.tokens}: {kept} of 256 bytes map to themselves")

    print(f"{args.input} -> {args.output}")
    written = fold_file(args.input, args.output, table)
    size_in = os.path.getsize(args.input)
    size_out = os.path.getsize(args.output)
    print(f"  bytes in {size_in:,}, out {size_out:,}")
    assert size_out == size_in == written, "fold changed the byte count"
    return 0


if __name__ == "__main__":
    sys.exit(main())
