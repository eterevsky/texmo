"""Generate tokens32_shift.json: a 32-token set built from ONE shift
marker plus two lossy buckets, over raw bytes (see docs/io.md,
"shift-32"). Type "shift_bucket", handled by
`texmo.tokens.shift_bucket_tokenizer.ShiftBucketTokenizer`.

The allocation (2026-07-31):

- token 0 is the numbered SHIFT marker (an ext token, like shift-64's
  markers): [shift, host] produces the host's semantic partner;
- 29 visible single-byte tokens: the 24 lowercase letters except q
  and z, plus ' ', ',', '.', '\\'' and '_';
- 29 shift pairs, one per visible token: 'a' -> 'A' ... for all 24
  letters, ' ' -> '\\n', ',' -> ';', '.' -> '?', '\\'' -> '"',
  '_' -> '-'. Semantic cousins, so the pair table transfers each
  host's statistics to a genuinely similar partner;
- two lossy faces, ordinary single-char tokens that double as their
  group's decode face (fold-style heads): "0" for all ten ASCII
  digits and "*" for every remaining byte value. Both groups are
  UNIFORM -- nothing is stored, so they are charged log2(10) and
  log2(188) bits per occurrence.

The four classes partition all 256 byte values exactly
(29 + 29 + 10 + 188 = 256), which the generator asserts.

Accounting: every byte costs one token except the 29 produced ones,
which cost two, so `total_tokens = raw_bytes + shifted occurrences`.
`extra_weights = 58` = 29 token selections + 29 stored pairs, the
same one-per-choice charge raw_fold and hexbpe pay; the two uniform
buckets store nothing. `residual_bits_per_byte` prices only the two
lossy groups, per RAW byte (processing is "raw", so raw and scanned
bytes coincide).

Reads the full-corpus raw byte counts from freq.txt (the Rust
`count-bytes` output: `'c' N` lines in char-debug form below 128,
`NNN N` above; the file repeats itself re-sorted after a blank line,
so only the first block is parsed).

Usage:
    uv run python scripts/make_shift32.py [freq.txt] \
        [tokens/tokens32_shift.json]

Without the output path it just prints the analysis.
"""
import math
import os
import sys

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.pjson import save_json

FREQ = sys.argv[1] if len(sys.argv) > 1 else "freq.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

SHIFT = 0  # numbered ext token: shifts the following token

_ESCAPES = {"t": 9, "n": 10, "r": 13, "'": 39, "\\": 92, "0": 0}

# The two lossy faces: ordinary single-char tokens that also decode
# for their whole group.
DIGIT_FACE = "0"
CATCH_FACE = "*"


def _parse_freq(path: str) -> list[int]:
    """Parse the Rust count-bytes output (first section only)."""
    counts = [0] * 256
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                break  # the sorted second section repeats the data
            repr_part, count = line.rsplit(" ", 1)
            if repr_part.startswith("'"):
                body = repr_part[1:-1]
                if body.startswith("\\u{"):
                    byte = int(body[3:-1], 16)
                elif body.startswith("\\"):
                    byte = _ESCAPES[body[1]]
                else:
                    assert len(body) == 1, repr(body)
                    byte = ord(body)
            else:
                byte = int(repr_part)
            assert 0 <= byte < 256, line
            counts[byte] += int(count)
    return counts


def _enc(b: int):
    """JSON spelling of a byte: a character below 128, a [int] list
    above (the same convention as the other generators)."""
    return chr(b) if b < 128 else [b]


def _show(b: int) -> str:
    return chr(b) if 32 < b < 127 else f"<{b:02x}>"


def main():
    counts = _parse_freq(FREQ)
    total = sum(counts)

    letters = [b for b in range(ord("a"), ord("z") + 1)
               if b not in (ord("q"), ord("z"))]
    visible = letters + [ord(c) for c in " ,.'_"]
    # host byte -> the byte [shift, host] produces.
    pairs = {b: b - 32 for b in letters}  # 'a' -> 'A'
    pairs.update({ord(" "): ord("\n"), ord(","): ord(";"),
                  ord("."): ord("?"), ord("'"): ord('"'),
                  ord("_"): ord("-")})
    shifted = sorted(pairs.values())
    digits = list(range(ord("0"), ord("9") + 1))
    covered = set(visible) | set(shifted) | set(digits)
    catch_all = [b for b in range(256) if b not in covered]

    assert len(visible) == 29 and len(pairs) == 29
    assert len(set(visible)) + len(set(shifted)) + len(digits) + len(
        catch_all) == 256, "the four classes must partition all bytes"
    assert not (set(visible) & set(shifted)), "a host cannot be produced"
    assert ord(DIGIT_FACE) in digits and ord(CATCH_FACE) in catch_all

    m_vis = sum(counts[b] for b in visible) / total
    m_shf = sum(counts[b] for b in shifted) / total
    m_dig = sum(counts[b] for b in digits) / total
    m_cat = sum(counts[b] for b in catch_all) / total
    # Uniform groups: every member charged log2(group size), per RAW
    # byte because eval divides the loss by the raw sample length.
    residual = m_dig * math.log2(len(digits)) + m_cat * math.log2(
        len(catch_all))
    # Shifted bytes cost two tokens; everything else costs one.
    total_tokens = total + sum(counts[b] for b in shifted)

    print(f"total bytes {total:,}")
    print(f"lossless coverage: direct {m_vis:.3%} + shifted {m_shf:.3%}"
          f" = {m_vis + m_shf:.3%}")
    print(f"digit bucket: {m_dig:.3%} @ log2({len(digits)}) -> "
          f"{m_dig * math.log2(len(digits)):.4f} b/B")
    print(f"catch-all: {len(catch_all)} values, {m_cat:.3%} @ "
          f"log2({len(catch_all)}) -> "
          f"{m_cat * math.log2(len(catch_all)):.4f} b/B")
    print(f"residual = {residual:.4f} b/B")
    print(f"tokens/byte = {total_tokens / total:.4f}  "
          f"(bytes/token {total / total_tokens:.4f})")
    print(f"extra weights = {len(visible)} selections + {len(pairs)} pairs"
          f" = {len(visible) + len(pairs)}")

    top = sorted((b for b in catch_all if counts[b]),
                 key=lambda b: -counts[b])[:12]
    print("catch-all top: " + " ".join(
        f"{_show(b)}({counts[b] / total:.3%})" for b in top))

    if not OUT:
        return

    tokens = [SHIFT] + [
        _enc(b) for b in sorted(
            set(visible) | {ord(DIGIT_FACE), ord(CATCH_FACE)})]
    assert len(tokens) == 32

    # One entry per byte that is not a token of its own, in byte
    # order: the produced bytes at 2 tokens, the two buckets' non-face
    # members at 1.
    sequences = []
    for b in range(256):
        if b in pairs.values():
            host = next(h for h, p in pairs.items() if p == b)
            sequences.append(
                {"string": _enc(b), "tokens": [SHIFT, chr(host)]})
        elif b in digits and b != ord(DIGIT_FACE):
            sequences.append({"string": _enc(b), "tokens": [DIGIT_FACE]})
        elif b in catch_all and b != ord(CATCH_FACE):
            sequences.append({"string": _enc(b), "tokens": [CATCH_FACE]})

    doc = {
        "processing": "raw",
        "type": "shift_bucket",
        "tokens": tokens,
        "sequences": sequences,
        "stats": {
            "ntokens": 32,
            "bytes_per_token": round(total / total_tokens, 6),
            # 29 token-slot selections + 29 stored shift pairs, one
            # weight per stored choice (the raw_fold/hexbpe
            # convention). The two uniform buckets store nothing.
            "extra_weights": len(visible) + len(pairs),
            "residual_bits_per_byte": round(residual, 6),
            "raw_bytes": total,
            # processing is "raw": scanned bytes are the raw bytes.
            "scanned_bytes": total,
            "total_tokens": total_tokens,
            "initial_size": total,
        },
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        save_json(doc, f)
    print(f"\nwrote {OUT}: {len(sequences)} sequences")


if __name__ == "__main__":
    main()
