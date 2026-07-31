"""Generate tokens64_shift.json: a fully lossless tokenset with
a shift marker and a hex escape (see docs/io.md). Despite growing
out of the fold family, it is an ordinary tokens+sequences set
handled by the generic DP Tokenizer -- every byte has exactly one
encoding, so the DP has no choices.

Allocation (64 tokens): 26 lowercase letters + 10 digits + the shift
marker (numbered token 0: A-Z encode as marker + lowercase letter,
'\\t' as marker + ' ', and six shifted-symbol pairs -- % + [ ] ` \\
as marker + # - ( ) ' /) + the hex escape (numbered token 1: any
byte without a shorter encoding spells its hex code with the
existing digit/letter tokens, e.g. '\\xa0' = [1, "a", "0"]) + 26
slots for the most frequent remaining bytes. Every byte round-trips
exactly, so there is no residual entropy and no stored weights --
the tail's information moves in-band, where a model can learn it.
The specials are numbered ext tokens, not marker characters: control
chars belong to str->str preprocessing (capswords), numbered tokens
to constructs living inside the tokenizer itself.

Usage:
    uv run python scripts/make_shift64.py freq.txt \
        tokens/tokens64_shift.json

Without the output path it prints the allocation and the full
escaped-byte table."""
import os
import re
import sys

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.pjson import save_json

FREQ = sys.argv[1] if len(sys.argv) > 1 else "freq.txt"
OUT = sys.argv[2] if len(sys.argv) > 2 else None

CAPS = 0  # numbered ext token: shifts the following token
ESC = 1   # numbered ext token: hex escape, followed by two digits

# Shifted-symbol pairs: byte -> the base token it encodes behind the
# shift marker (semantic cousins, all base tokens hold top slots).
SHIFT = {ord("%"): "#", ord("+"): "-", ord("["): "(",
         ord("]"): ")", ord("`"): "'", ord("\\"): "/"}

# ---- parse freq.txt (first block only; the file has a re-sorted
# duplicate listing after a blank line) ----
counts = [0] * 256
with open(FREQ, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            break
        m = re.match(r"^'(.*)' (\d+)$", line)
        if m:
            lit, n = m.group(1), int(m.group(2))
            if lit == "\\t":
                b = 9
            elif lit == "\\n":
                b = 10
            elif lit == "\\'":
                b = 39
            elif lit == "\\\\":
                b = 92
            elif lit.startswith("\\u{"):
                b = int(lit[3:-1], 16)
            else:
                assert len(lit) == 1, repr(lit)
                b = ord(lit)
            counts[b] += n
        else:
            m = re.match(r"^(\d+) (\d+)$", line)
            assert m, repr(line)
            counts[int(m.group(1))] += int(m.group(2))

total = sum(counts)

base = (set(range(ord("a"), ord("z") + 1))
        | set(range(ord("0"), ord("9") + 1)))
marker_repr = (set(range(ord("A"), ord("Z") + 1)) | {ord("\t")}
               | set(SHIFT))
candidates = sorted(
    (b for b in range(256) if b not in base and b not in marker_repr),
    key=lambda b: -counts[b])

N_SLOTS = 64 - 26 - 10 - 2
top = candidates[:N_SLOTS]
escaped = sorted(candidates[N_SLOTS:], key=lambda b: -counts[b])

marker_uses = sum(counts[b] for b in marker_repr)
esc_mass = sum(counts[b] for b in escaped) / total
# marker bytes cost 2 tokens, escaped bytes cost 3.
total_tokens = (total + marker_uses
                + 2 * sum(counts[b] for b in escaped))

print(f"total bytes {total:,}; tokens/byte "
      f"{total_tokens / total:.4f}")
print(f"top-{N_SLOTS} byte tokens: "
      + " ".join(repr(chr(b))[1:-1] if 32 <= b < 127 else f"<{b}>"
                 for b in top))
print(f"hex-escaped: {len(escaped)} bytes, {100 * esc_mass:.4f}% of "
      f"corpus at 3 tokens/byte")
print("lossless: residual 0, extra weights 0\n")

print("escaped bytes by frequency:")
for b in escaped:
    if not counts[b]:
        break
    name = repr(chr(b))[1:-1] if 32 <= b < 127 else f"<{b}>"
    print(f"  {name:>6} {counts[b]:>12,}  {100 * counts[b] / total:.5f}%")
never = [b for b in escaped if not counts[b]]
print(f"  ... plus {len(never)} bytes with count 0")

if OUT:
    def enc(b: int):
        if b < 128:
            return chr(b)
        return [b]

    tokens = [CAPS, ESC] + [enc(b) for b in sorted(base | set(top))]
    assert len(tokens) == 64

    # Every byte without its own token, in increasing byte order --
    # including bytes the corpus never contained.
    escaped_set = set(escaped)
    sequences = []
    for b in range(256):
        if b == ord("\t"):
            sequences.append({"string": "\t", "tokens": [CAPS, " "]})
        elif ord("A") <= b <= ord("Z"):
            sequences.append(
                {"string": chr(b), "tokens": [CAPS, chr(b).lower()]})
        elif b in SHIFT:
            sequences.append({"string": chr(b), "tokens": [CAPS, SHIFT[b]]})
        elif b in escaped_set:
            hexed = f"{b:02x}"
            sequences.append(
                {"string": enc(b), "tokens": [ESC, hexed[0], hexed[1]]})

    doc = {
        "processing": "raw",
        "type": "shift",
        "tokens": tokens,
        "sequences": sequences,
        "stats": {
            "ntokens": 64,
            "bytes_per_token": round(total / total_tokens, 6),
            # 62 byte-token selections + 33 stored shift pairs (26
            # capitals, tab, six symbols), 1 weight per choice --
            # the tokens.32.shift convention (recharged 2026-07-31
            # from the flat 64). The two markers and the hex-escape
            # spellings are structural and free.
            "extra_weights": 95,
            "residual_bits_per_byte": 0.0,
            "scanned_bytes": total,
            "total_tokens": total_tokens,
        },
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        save_json(doc, f)
    print(f"\nwrote {OUT}: {len(sequences)} sequences")
