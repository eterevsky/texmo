"""Tests for ShiftBucketTokenizer against the real tokens32_shift set.

The tokens dir is registered by texmo/conftest.py, so `get_tokenizer`
serves the shipped JSON -- which is the point: these pin the
generated set's contract (32 ids, 1-2 tokens per byte, which bytes
survive and which are forgotten) alongside the class's behaviour.
"""
import math
import os

from .registry import get_tokenizer
from .shift_bucket_tokenizer import ShiftBucketTokenizer

_FREQ = os.path.join(
    os.path.dirname(__file__), "..", "..", "freq.txt")

_ESCAPES = {"t": 9, "n": 10, "r": 13, "'": 39, "\\": 92, "0": 0}

# The bytes the set spells losslessly: its own tokens plus everything
# reachable as [shift, host].
_LOSSLESS_TEXT = (
    b"Hello, World? It cost_a lot;\n"
    b"'Yes,' he said. \"Nothing but letters.\"\n")


def _tokenizer() -> ShiftBucketTokenizer:
    return get_tokenizer("tokens.32.shift")


def _byte_counts() -> list[int]:
    """Raw corpus byte counts from freq.txt (first block only), the
    same input the generator used."""
    counts = [0] * 256
    with open(_FREQ, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                break
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
            counts[byte] += int(count)
    return counts


def test_registry_routes_to_the_shift_bucket_tokenizer():
    tk = _tokenizer()
    assert isinstance(tk, ShiftBucketTokenizer)
    assert tk.tokenset.type == "shift_bucket"
    assert tk.tokenset.processing == "raw"
    assert tk.tokenset.ntokens == 32
    assert tk.tokenset.stats["extra_weights"] == 58
    # The marker is the numbered ext token at index 0.
    assert tk.shift_id == 0
    assert tk.tokenset.tokens[0].string is None


def test_lossless_text_round_trips_exactly():
    tk = _tokenizer()
    assert tk.untokenize(tk.tokenize(_LOSSLESS_TEXT)) == _LOSSLESS_TEXT
    # Capitals in particular: they only exist behind the shift.
    assert tk.untokenize(tk.tokenize(b"Hello, World?")) == b"Hello, World?"


def test_every_byte_costs_one_or_two_ids_under_32():
    tk = _tokenizer()
    for b in range(256):
        ids = tk.tokenize(bytes([b]))
        assert 1 <= len(ids) <= 2, b
        assert all(0 <= i < 32 for i in ids), (b, ids)
    # Round-tripping the whole byte range never raises either.
    assert isinstance(tk.untokenize(tk.tokenize(bytes(range(256)))), bytes)


def test_shifted_bytes_cost_exactly_two_tokens():
    tk = _tokenizer()
    assert len(tk.tokenize(b"a")) == 1
    assert tk.tokenize(b"A") == [tk.shift_id] + tk.tokenize(b"a")
    # The non-letter pairs are shifts too: ' ' -> '\n', ',' -> ';',
    # '.' -> '?', "'" -> '"', '_' -> '-'.
    for host, produced in ((b" ", b"\n"), (b",", b";"), (b".", b"?"),
                           (b"'", b'"'), (b"_", b"-")):
        assert len(tk.tokenize(host)) == 1, host
        assert tk.tokenize(produced) == [tk.shift_id] + tk.tokenize(host)


def test_digits_and_catch_all_are_lossy():
    tk = _tokenizer()
    # Every digit prints as the group's face, so the SHAPE of a
    # number survives but not its value.
    assert tk.untokenize(tk.tokenize(b"1984")) == b"0000"
    assert len(tk.tokenize(b"1984")) == 4
    # The catch-all prints '*', at one token each.
    assert tk.untokenize(tk.tokenize(b"$%&[]")) == b"*****"
    # q and z were dropped from the letter set: they are catch-all
    # members, not tokens of their own.
    assert tk.untokenize(tk.tokenize(b"quiz")) == b"*ui*"
    assert len(tk.tokenize(b"quiz")) == 4
    # Bytes >= 128 too (the [int] list form in the JSON).
    assert tk.untokenize(tk.tokenize(bytes([0xC3, 0x9C]))) == b"**"


def test_garbage_shift_streams_never_raise():
    tk = _tokenizer()
    shift = tk.shift_id
    a = tk.tokenize(b"a")[0]
    zero = tk.tokenize(b"0")[0]
    star = tk.tokenize(b"*")[0]
    # A trailing shift is dropped.
    assert tk.untokenize([a, shift]) == b"a"
    assert tk.untokenize([shift]) == b""
    # Shift before shift: the first one is dropped, the second binds.
    assert tk.untokenize([shift, shift, a]) == b"A"
    assert tk.untokenize([shift, shift, shift]) == b""
    # Shift before a bucket face (no pair): the shift is dropped and
    # the face prints itself.
    assert tk.untokenize([shift, zero]) == b"0"
    assert tk.untokenize([shift, star]) == b"*"
    # Ids outside the vocabulary are skipped rather than raising.
    assert tk.untokenize([a, 99, a]) == b"aa"
    assert tk.untokenize([]) == b""


def test_tokens_per_byte_matches_the_stats():
    tk = _tokenizer()
    text = (
        b"It is a truth universally acknowledged, that a single man in "
        b"possession of a good fortune, must be in want of a wife.\n"
        b"However little known the feelings or views of such a man may "
        b"be on his first entering a neighbourhood, this truth is so "
        b"well fixed in the minds of the surrounding families, that he "
        b"is considered as the rightful property of some one or other "
        b"of their daughters.\n")
    measured = len(tk.tokenize(text)) / len(text)
    expected = 1.0 / tk.tokenset.stats["bytes_per_token"]
    assert abs(measured - expected) < 0.15 * expected, (measured, expected)


def test_stats_are_internally_consistent():
    stats = _tokenizer().tokenset.stats
    assert stats["ntokens"] == 32
    assert stats["scanned_bytes"] == stats["raw_bytes"]  # processing raw
    assert stats["initial_size"] == stats["raw_bytes"]
    assert math.isclose(
        stats["bytes_per_token"],
        stats["raw_bytes"] / stats["total_tokens"], abs_tol=1e-6)


def test_residual_recomputed_from_the_sequences():
    """The stored charge must match what the set's own groups imply
    on the corpus counts: uniform log2(group size) per member."""
    tk = _tokenizer()
    counts = _byte_counts()
    total = sum(counts)

    # Rebuild the groups from the shipped JSON rather than from the
    # generator's constants.
    digit_face = tk.tokenize(b"0")[0]
    catch_face = tk.tokenize(b"*")[0]
    groups: dict[int, list[int]] = {digit_face: [], catch_face: []}
    shifted = 0
    for b in range(256):
        ids = tk.tokenize(bytes([b]))
        if len(ids) == 2:
            shifted += counts[b]
        elif ids[0] in groups:
            groups[ids[0]].append(b)
    assert len(groups[digit_face]) == 10
    assert len(groups[catch_face]) == 188

    residual = sum(
        sum(counts[b] for b in members) * math.log2(len(members))
        for members in groups.values()) / total
    assert abs(residual - tk.tokenset.residual_bits_per_byte) < 1e-4
    # ...and the token count the stats claim, from the same counts.
    assert math.isclose(
        tk.tokenset.stats["total_tokens"], total + shifted, rel_tol=1e-9)
