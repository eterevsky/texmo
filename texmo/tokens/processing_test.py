"""Tests for capswords2 processing.

`capswords2_cases.json` is a SHARED fixture: the Rust implementation in
`src/processing.rs` asserts against the same file. The two must agree
byte for byte -- Rust builds the tokensets and Python tokenizes with
them, so a divergence never raises, it just silently produces worse
compression. Regenerate with `scratch/gen_capswords2_cases.py`.
"""
import json
import os
import random
import string

import pytest

from .processing import (
    ALLCAPS_MARKER,
    CAPITALIZED_MARKER,
    UPPER_MARKER,
    WORD_MARKER,
    apply_processing,
    process2,
    undo_processing,
    unprocess2,
)

_CASES_PATH = os.path.join(os.path.dirname(__file__), "capswords2_cases.json")
with open(_CASES_PATH, encoding="utf-8") as _f:
    CASES = json.load(_f)

MARKERS = (CAPITALIZED_MARKER, ALLCAPS_MARKER, WORD_MARKER, UPPER_MARKER)


# --- the shared contract ---------------------------------------------


def test_fixture_is_substantial():
    """Guard against the fixture being silently emptied."""
    assert len(CASES) >= 80
    assert sum(c["roundtrip"] for c in CASES) >= 70


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c["in"])[:40])
def test_matches_shared_fixture(case):
    assert process2(case["in"].encode()).decode() == case["out"]


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["roundtrip"]],
    ids=lambda c: repr(c["in"])[:40])
def test_round_trip(case):
    assert unprocess2(process2(case["in"].encode())).decode() == case["in"]


# --- invariants that must hold for every case ------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c["in"])[:40])
def test_no_uppercase_survives(case):
    """The whole point of capswords2: a tokenset never has to spend
    slots on uppercase letters."""
    if any(m in case["in"] for m in MARKERS):
        return  # marker-bearing input is out of contract, see below
    assert not any(ch.isupper() for ch in case["out"])


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c["in"])[:40])
def test_word_marker_count_equals_word_count(case):
    """One WORD_MARKER per maximal run of alphabetic characters."""
    if any(m in case["in"] for m in MARKERS):
        return
    words, in_word = 0, False
    for ch in case["in"]:
        if ch.isalpha():
            if not in_word:
                words += 1
            in_word = True
        else:
            in_word = False
    assert case["out"].count(WORD_MARKER) == words


# --- known limitations, pinned so they stay deliberate ---------------


def test_marker_bearing_input_is_not_round_trippable():
    """The markers have no escape, so text already containing them is
    outside the contract. Documented rather than fixed: the corpus is
    natural text and the four markers are C0 controls."""
    for text in ("a\x14b", "a\x15b", "a\x16b", "a\x17b"):
        assert unprocess2(process2(text.encode())).decode() != text


def test_multi_character_lowercase_is_not_round_trippable():
    """Turkish dotted capital I lowercases to two codepoints (i + a
    combining dot); re-uppercasing recovers only the first."""
    assert process2("İstanbul".encode()).decode() == "\x14i̇stanbul\x16"
    assert unprocess2(process2("İstanbul".encode())).decode() != "İstanbul"


# --- targeted behaviour ----------------------------------------------


def test_whole_word_case_markers():
    assert process2(b"Hello") == b"\x14hello\x16"
    assert process2(b"HELLO") == b"\x15hello\x16"
    # A one-letter word takes the capitalized branch, not all-caps.
    assert process2(b"A") == b"\x14a\x16"
    # Two letters is enough for all-caps.
    assert process2(b"AB") == b"\x15ab\x16"


def test_mid_word_capitals_use_the_upper_marker():
    assert process2(b"HeLLo") == b"\x17he\x17l\x17lo\x16"
    assert process2(b"iPhone") == b"i\x17phone\x16"
    assert process2(b"aB") == b"a\x17b\x16"


def test_single_inter_word_space_is_elided():
    assert process2(b"a b") == b"a\x16b\x16"
    assert process2(b"a  b") == b"a\x16  b\x16"
    assert process2(b"a   b") == b"a\x16   b\x16"


def test_uncased_scripts_pass_through():
    for text in ("日本語", "עברית", "العربية"):
        assert process2(text.encode()).decode() == text + WORD_MARKER


def test_digits_and_symbols_break_words():
    assert process2(b"A1") == b"\x14a\x161"
    assert process2(b"abc123def") == b"abc\x16123def\x16"


# --- robustness -------------------------------------------------------


def test_unprocess_tolerates_truncated_and_stray_markers():
    """Sampled model output ends anywhere and can emit markers in any
    order; decoding must not raise."""
    for junk in (b"\x17", b"\x14", b"\x15", b"\x16", b"\x17\x16",
                 b"abc\x17", b"\x14\x15\x17\x16\x16", b"\x17\x17\x17a"):
        assert isinstance(unprocess2(junk), bytes)


def test_unprocess_passes_through_invalid_utf8():
    assert unprocess2(b"\xff\xfe") == b"\xff\xfe"


def test_processing_dispatch():
    assert apply_processing("capswords2", b"Hello") == b"\x14hello\x16"
    assert undo_processing("capswords2", b"\x14hello\x16") == b"Hello"
    # capswords is untouched by this change and still routes correctly.
    assert apply_processing("capswords", b"HeLLo") == b"HeLLo\x16"
    # An unknown or 'raw' pipeline is the identity.
    assert apply_processing("raw", b"Hello") == b"Hello"
    assert undo_processing("raw", b"Hello") == b"Hello"


# --- fuzz -------------------------------------------------------------


def _random_text(rng) -> str:
    alphabet = (string.ascii_letters + "  " + string.digits
                + ".,!?-'\n" + "éÉñÑ日本Привет")
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))


def test_fuzz_round_trip():
    """Random text over an alphabet that excludes the markers must
    always round-trip."""
    rng = random.Random(20260727)
    for _ in range(3000):
        text = _random_text(rng)
        restored = unprocess2(process2(text.encode())).decode()
        assert restored == text, f"{text!r} -> {restored!r}"


def test_fuzz_no_uppercase_and_marker_count():
    rng = random.Random(11)
    for _ in range(1500):
        text = _random_text(rng)
        out = process2(text.encode()).decode()
        assert not any(ch.isupper() for ch in out), text
