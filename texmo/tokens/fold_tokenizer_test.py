import math
import os

import numpy as np

from .fold_tokenizer import FoldTokenizer
from .processing import process2
from .tokenset import TokenSet

_TOKENS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "tokens")


def _make_tokenset() -> TokenSet:
    """Tiny fold set: 'a' <- {a, A}, 'b' <- {b, B}, ' ' <- {' '},
    '?' <- everything else."""
    tokens = ["a", "b", " ", "?"]

    def fold(b: int) -> str:
        c = chr(b)
        if c in "aA":
            return "a"
        if c in "bB":
            return "b"
        if c == " ":
            return " "
        return "?"

    sequences = [
        {"string": chr(b) if b < 128 else [b], "tokens": [fold(b)]}
        for b in range(256)
        if b not in (ord("a"), ord("b"), ord(" "), ord("?"))
    ]
    return TokenSet.from_json({
        "processing": "raw",
        "type": "fold",
        "tokens": tokens,
        "sequences": sequences,
        "head_freq": {"a": 0.9, "b": 0.8, " ": 1.0, "?": 0.1},
        "stats": {
            "ntokens": 4,
            "bytes_per_token": 1.0,
            "extra_weights": 4,
            "residual_bits_per_byte": 1.25,
            "scanned_bytes": 1000,
            "total_tokens": 1000,
        },
    })


def test_tokenize_folds_bytes_to_groups():
    tk = FoldTokenizer(_make_tokenset())
    ids = tk.tokenize(b"aAbB x\xc3\x9c")
    a, b, sp, q = 0, 1, 2, 3
    assert list(ids) == [a, a, b, b, sp, q, q, q]


def test_untokenize_prints_head_chars():
    tk = FoldTokenizer(_make_tokenset())
    assert tk.untokenize(tk.tokenize(b"aAbB x!")) == b"aabb ??"
    # Idempotent on already-folded text.
    assert tk.untokenize(tk.tokenize(b"ab ?")) == b"ab ?"


def test_residual_entropy_head_scheme():
    tk = FoldTokenizer(_make_tokenset())
    # Head byte of a group: -log2(p_head).
    assert math.isclose(tk.residual_entropy[ord("a")], -math.log2(0.9))
    # The other member splits (1 - p_head) over the group remainder.
    assert math.isclose(tk.residual_entropy[ord("A")], -math.log2(0.1))
    # Singleton group with p = 1: zero charge.
    assert tk.residual_entropy[ord(" ")] == 0.0
    # '?' group: 251 members, so 250 non-heads share 0.9.
    assert math.isclose(
        tk.residual_entropy[ord("x")], -math.log2(0.9 / 250))
    assert math.isclose(
        tk.chunk_residual_bits(b"aA"), -math.log2(0.9) - math.log2(0.1))


def test_byte_loss_adds_residual():
    ts = _make_tokenset()
    assert math.isclose(ts.byte_loss(2.0), 2.0 + 1.25)
    assert ts.residual_bits_per_byte == 1.25


def test_lossless_byte_loss_has_no_residual():
    ts = TokenSet("byteshuff", "raw", {"bytes_per_token": 0.5})
    assert ts.residual_bits_per_byte == 0.0
    assert math.isclose(ts.byte_loss(2.0), 4.0)


def test_real_fold32_tokenset():
    path = os.path.join(_TOKENS_DIR, "tokens32_raw_fold.json")
    ts = TokenSet.from_json_file(path)
    assert ts.name == "tokens32_raw_fold"
    assert ts.ntokens == 32
    tk = FoldTokenizer(ts)
    # Every byte covered, one token per byte.
    assert tk.groups.shape == (256,)
    assert (tk.groups >= 0).all()
    out = tk.untokenize(tk.tokenize(b'Hello, "World"! It cost $42.'))
    assert out == b"hello, 'world'. it cost ?xx."
    # Corpus-average charge is finite and matches the stored constant
    # to well under a millibit on a typical english sentence scale.
    assert 0.3 < ts.residual_bits_per_byte < 0.4
    assert np.isfinite(tk.residual_entropy[ord("e")])


def test_uniform_group_without_head_freq():
    # A group missing from head_freq is charged uniformly, costing
    # no stored weights.
    ts = _make_tokenset()
    del ts.head_freq["?"]
    tk = FoldTokenizer(ts)
    assert math.isclose(tk.residual_entropy[ord("?")], math.log2(251))
    assert math.isclose(tk.residual_entropy[ord("x")], math.log2(251))


def test_bucket64_real_set():
    """tokens64_bucket: capswords2 + top-63 bytes + uniform catch-all
    (2026-07-30). Strictly one token per processed byte; the bucket
    stores nothing, so extra_weights is the 63 selections."""
    ts = TokenSet.from_json_file(
        os.path.join(_TOKENS_DIR, "tokens64_bucket.json"))
    assert ts.processing == "capswords2"
    assert not ts.head_freq
    assert ts.stats["extra_weights"] == 63
    tk = FoldTokenizer(ts)

    text = b"The Quick brown fox, 42 lazy dogs [rare]!"
    ids = tk.tokenize(text)
    assert len(ids) == len(process2(text))  # 1:1 with processed bytes
    assert int(max(ids)) < 64
    assert isinstance(tk.untokenize(list(ids)), bytes)

    # The bucket group: 193 byte values, all charged uniformly.
    bucket_id = tk.groups[ord("[")]
    members = np.flatnonzero(tk.groups == bucket_id)
    assert len(members) == 193
    assert math.isclose(
        tk.residual_entropy[ord("[")], math.log2(193))
    # All 26 letters, all digits, and the four markers have their own
    # singleton tokens.
    for b in (list(range(ord("a"), ord("z") + 1))
              + list(range(ord("0"), ord("9") + 1))
              + [0x14, 0x15, 0x16, 0x17]):
        assert tk.groups[b] != bucket_id, chr(b)
        assert math.isclose(tk.residual_entropy[b], 0.0), chr(b)
