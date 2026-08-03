"""Generate tokens.32.fold.json: a fold (forgetting) 32-token set
with head-character frequency accounting (see docs/io.md, "Fold
tokensets").

Reads a byte-count file (rust-style `'c' count` / `<int> count`
lines; produced by counting books3.txt), computes per-group head
frequencies and the corpus-average residual bits/byte, and writes
the tokenset JSON.

Usage:
    uv run python scripts/make_fold32.py freq.txt \
        tokens/tokens.32.fold.json

Without the output path it just prints the analysis, including the
comparison of the two underscore-placement variants."""
import math
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
            # rust-style escapes
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

TOKENS = ["\n", " ", "'", ",", ".", "?"] + [chr(c) for c in range(97, 123)]
assert len(TOKENS) == 32


def make_fold(underscore_to_space: bool):
    def fold(b: int) -> str:
        c = chr(b)
        if "a" <= c <= "z":
            return c
        if "A" <= c <= "Z":
            return c.lower()
        if "0" <= c <= "9":
            return "x"
        if c == "_":
            return " " if underscore_to_space else "'"
        if c in "'\"`()[]{}":
            return "'"
        if c in ".!:":
            return "."
        if c in ",;-/":
            return ","
        if c == "\n":
            return "\n"
        if b == 0x20 or (b < 0x20 and b != 0x0A):
            return " "
        # everything else: leftover ASCII symbols, DEL, all bytes >= 128
        return "?"
    return fold


def analyze(fold):
    groups = {t: [] for t in TOKENS}
    for b in range(256):
        groups[fold(b)].append(b)
    assert sum(len(g) for g in groups.values()) == 256

    head_freq = {}
    residual = 0.0        # head-scheme charge, corpus average bits/byte
    residual_exact = 0.0  # exact within-group conditional (256 weights)
    h_token = 0.0         # unigram entropy of the token stream
    for t in TOKENS:
        members = groups[t]
        g_count = sum(counts[b] for b in members)
        head = ord(t)
        assert head in members
        p_head = counts[head] / g_count if g_count else 1.0
        if len(members) == 1:
            p_head = 1.0
        head_freq[t] = p_head
        if g_count:
            pg = g_count / total
            h_token -= pg * math.log2(pg)
        n_rest = len(members) - 1
        for b in members:
            if not counts[b]:
                continue
            w = counts[b] / total
            residual_exact -= w * math.log2(counts[b] / g_count)
            q = p_head if b == head else (1.0 - p_head) / n_rest
            residual -= w * math.log2(q)
    return groups, head_freq, residual, residual_exact, h_token


print(f"total bytes: {total:,}")
h_byte = -sum(c / total * math.log2(c / total) for c in counts if c)
print(f"unigram byte entropy H(B) = {h_byte:.4f} b/B\n")

for us, label in ((False, "_ -> ' (delimiter group)"),
                  (True, "_ -> ' ' (space group)")):
    groups, head_freq, residual, residual_exact, h_token = analyze(
        make_fold(us))
    print(f"variant: {label}")
    print(f"  token entropy H(G)      = {h_token:.4f} b/B")
    print(f"  exact residual (256 w)  = {residual_exact:.4f} b/B")
    print(f"  head-scheme residual    = {residual:.4f} b/B "
          f"(overhead {residual - residual_exact:.4f})")
    for t in ("'", " "):
        members = groups[t]
        g_count = sum(counts[b] for b in members)
        chg = -math.log2((1 - head_freq[t]) / (len(members) - 1))
        print(f"  {t!r}: n={len(members)} p_head={head_freq[t]:.4f} "
              f"non-head charge={chg:.2f} bits")
    print()

# ---- write the chosen variant: underscore stays in the ' group ----
fold = make_fold(False)
groups, head_freq, residual, residual_exact, h_token = analyze(fold)

print(f"{'tok':>4} {'n':>3} {'group%':>7} {'p_head':>7}  members(with counts)")
for t in TOKENS:
    members = groups[t]
    g_count = sum(counts[b] for b in members)
    live = [chr(b) if 32 < b < 127 else f"<{b}>"
            for b in members if counts[b] and b != ord(t)]
    show = "".join(live[:12]) + ("..." if len(live) > 12 else "")
    print(f"{t!r:>4} {len(members):>3} {100*g_count/total:>6.2f}% "
          f"{head_freq[t]:>7.4f}  {show}")

if OUT:
    def enc(b: int):
        if b < 128:
            return chr(b)
        return [b]

    sequences = []
    for b in range(256):
        t = fold(b)
        if b == ord(t):
            continue  # the head byte is the token itself
        sequences.append({"string": enc(b), "tokens": [t]})

    doc = {
        "processing": "raw",
        "type": "fold",
        "tokens": TOKENS,
        "sequences": sequences,
        "head_freq": {t: round(head_freq[t], 6) for t in TOKENS},
        "stats": {
            "ntokens": 32,
            "bytes_per_token": 1.0,
            # 32 head-token selections + 32 head frequencies. The full
            # fold map assigns all 256 byte values, but one weight per
            # group head mirrors hexbpe's one-per-selected-byte charge
            # and is agreed good enough (2026-07-29).
            "extra_weights": 64,
            "residual_bits_per_byte": round(residual, 6),
            "scanned_bytes": total,
            "total_tokens": total,
        },
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        save_json(doc, f)
    print(f"\nwrote {OUT}: {len(sequences)} sequences, 32 head freqs")
