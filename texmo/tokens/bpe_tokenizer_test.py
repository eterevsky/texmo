"""BpeTokenizer parity tests against reference fixtures.

The fixtures in testdata/gemma_parity.json were generated once from
the HF `tokenizers` reference (scratch/gen_gemma_fixtures.py, run with
an ephemeral dependency); these tests verify our from-scratch BPE
implementation offline.
"""
import json
import os

import pytest

from texmo.tokens.bpe_tokenizer import BpeTokenizer
from texmo.tokens.tokenset import TokenSet

_HERE = os.path.dirname(__file__)
_TOKENSET = os.path.join(
    _HERE, "..", "..", "tokens", "tokens256000_gemma.json")
_FIXTURES = os.path.join(_HERE, "testdata", "gemma_parity.json")


@pytest.fixture(scope="module")
def tok() -> BpeTokenizer:
    return BpeTokenizer(TokenSet.from_json_file(_TOKENSET))


def _fixtures():
    with open(_FIXTURES, encoding="utf-8") as f:
        return json.load(f)["fixtures"]


@pytest.mark.parametrize(
    "case", _fixtures(),
    ids=[ascii(c["text"][:30]) for c in _fixtures()])
def test_parity_with_reference(tok, case):
    assert tok.encode(case["text"], add_bos=False) == case["ids"]


@pytest.mark.parametrize(
    "case", _fixtures(),
    ids=[ascii(c["text"][:30]) for c in _fixtures()])
def test_roundtrip(tok, case):
    # U+2581 in input is conflated with space by design (reference
    # behavior), so the round-trip target has it replaced.
    expected = case["text"].replace("▁", " ")
    assert tok.decode(case["ids"]) == expected


def test_add_bos(tok):
    plain = tok.encode("Hello, world!", add_bos=False)
    with_bos = tok.encode("Hello, world!", add_bos=True)
    assert with_bos == [tok.tokenset.bos_id] + plain
    # Default comes from the tokenset flag (true for Gemma).
    assert tok.encode("Hello, world!") == with_bos


def test_specials_roundtrip_via_pua(tok):
    ts = tok.tokenset
    ids = [ts.bos_id] + tok.encode("hi", add_bos=False) + [ts.eos_id]
    s = tok.decode(ids)
    assert chr(0xF0000 + ts.bos_id) in s
    assert chr(0xF0000 + ts.eos_id) in s
    # Encoding the decoded string reproduces the ids exactly.
    assert tok.encode(s, add_bos=False) == ids


def test_atom_prefers_char_piece_over_byte_token(tok):
    # " " must resolve to the converted U+2581 piece, not <0x20>.
    [space_id] = tok.encode(" ", add_bos=False)
    assert not tok.tokenset.tokens[space_id].is_byte


def test_byte_fallback_fires(tok):
    # A char with no piece decomposes into byte tokens that still
    # round-trip through UTF-8. (The 256k vocab covers a lot -- find a
    # genuinely piece-less char instead of guessing one.)
    rare = next(
        chr(c) for c in range(0x4DC0, 0x4E00)  # Yijing hexagrams
        if chr(c) not in tok._char_piece)
    ids = tok.encode(rare, add_bos=False)
    assert len(ids) >= 2  # multi-byte fallback (possibly partly merged)
    assert all(tok.tokenset.tokens[i].is_byte for i in ids)
    assert tok.decode(ids) == rare


def test_priority_matches_only_literal_u2581_runs(tok):
    # Real double space is NOT the run token (reference added tokens
    # are normalized=False): it goes through BPE.
    two_spaces = tok.encode("  ", add_bos=False)
    literal_run = tok.encode("▁▁", add_bos=False)
    run_id = [
        i for i in tok.tokenset.priority_tokens
        if tok.tokenset.tokens[i].string == b"  "
    ][0]
    assert literal_run == [run_id]
    # Whatever BPE yields for real spaces must still decode correctly.
    assert tok.decode(two_spaces) == "  "
