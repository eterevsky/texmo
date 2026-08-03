import os

from .hexbpe_tokenizer import HexBpeTokenizer
from .processing import process2
from .tokenset import TokenSet

_TOKENS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "tokens")

# A paragraph with the features capswords2 exists for: sentence-initial
# capitals, ALLCAPS, mid-word capitals, punctuation, digits and
# non-ASCII (which no hexbpe set selects, so it exercises the nibble
# fallback).
_TEXT = (
    "It is a truth universally acknowledged, that a single man in "
    "possession of a good fortune, must be in want of a wife. However "
    "little known the feelings or views of such a man may be on his "
    "first entering a neighbourhood, this truth is so well fixed in "
    "the minds of the surrounding families, that he is considered the "
    "rightful property of some one or other of their daughters.\n"
    "McDonald's iPhone costs 42% MORE in cafés — NOT OK!\n"
).encode("utf-8")


def _load(ntokens: int) -> HexBpeTokenizer:
    path = os.path.join(_TOKENS_DIR, f"tokens.{ntokens}.hexbpe.json")
    return HexBpeTokenizer(TokenSet.from_json_file(path))


def _synthetic(merges: list) -> TokenSet:
    """Minimal hexbpe-shaped set over 'a'..'d'.

    Vocabulary is fixed; only the merge ORDER varies, which is the
    whole point: a rank-driven encoder gives different answers for
    different orders over the same pieces.
    """
    selected = ["a", "b", "c", "d"]
    tokens = list(range(16)) + selected + ["ab", "bc", "cd"]
    sequences = [
        {
            "string": chr(b) if b < 128 else [b],
            "tokens": [b >> 4, b & 15],
        }
        for b in range(256)
        if chr(b) not in selected
    ]
    return TokenSet.from_json({
        "type": "hexbpe",
        "processing": "raw",
        "algorithm": "bpe",
        "tokens": tokens,
        "sequences": sequences,
        "merges": merges,
        "stats": {
            "ntokens": len(tokens),
            "bytes_per_token": 1.0,
            "scanned_bytes": 100,
            "total_tokens": 100,
        },
    })


def test_roundtrip_through_capswords2():
    for ntokens in (32, 64, 128, 256):
        tk = _load(ntokens)
        assert tk.tokenset.processing == "capswords2"
        assert tk.untokenize(tk.tokenize(_TEXT)) == _TEXT


def test_roundtrip_of_every_byte():
    # Losslessness on the raw byte space: capswords2 needs valid UTF-8,
    # so this goes through tokenize_processed directly.
    tk = _load(32)
    data = bytes(range(256))
    assert tk.untokenize(tk.tokenize_processed(data)) == data


def test_unselected_byte_is_two_nibbles():
    tk = _load(32)
    selected = {
        t.string for t in tk.tokenset.tokens
        if t.string is not None and len(t.string) == 1
    }
    unselected = [b for b in range(256) if bytes([b]) not in selected]
    assert len(unselected) == len(tk.tokenset.sequences) == 241
    for b in unselected:
        ids = tk.tokenize_processed(bytes([b]))
        assert len(ids) == 2, (b, ids)
        # Both are nibble tokens (no string of their own) spelling the
        # byte's hex code.
        assert ids == [b >> 4, b & 15]
        assert all(tk.tokenset.tokens[i].string is None for i in ids)


def test_selected_byte_is_one_token():
    tk = _load(32)
    for b in (ord(" "), ord("e"), ord("t")):
        ids = tk.tokenize_processed(bytes([b]))
        assert len(ids) == 1
        assert tk.tokenset.tokens[ids[0]].string == bytes([b])


def test_merge_order_beats_optimal_cover():
    """Rank order, not the cheapest cover, decides the split.

    Pieces are 'ab', 'bc', 'cd'. On "abcd" the optimal cover is
    ['ab', 'cd'] (2 tokens), and that is exactly what the DP tokenizer
    would find -- but with ('b','c') ranked first, BPE eats the middle
    and is left with 3 tokens it can no longer merge.
    """
    tk = _synthetic([["b", "c"], ["a", "b"], ["c", "d"]])
    ids = tk_ids = HexBpeTokenizer(tk).tokenize_processed(b"abcd")
    assert [str(tk.tokens[i]) for i in tk_ids] == ["a", "bc", "d"]
    assert len(ids) == 3

    # Same vocabulary, ('b','c') ranked last: the outer merges fire
    # first and the optimal 2-token cover falls out.
    tk2 = _synthetic([["a", "b"], ["c", "d"], ["b", "c"]])
    ids2 = HexBpeTokenizer(tk2).tokenize_processed(b"abcd")
    assert [str(tk2.tokens[i]) for i in ids2] == ["ab", "cd"]
    assert len(ids2) == 2


def test_merges_stay_inside_the_atom_stream():
    # Nibble tokens carry no string, so no merge can reference them:
    # a hex-escaped byte is a hard barrier between its neighbours.
    tk = _load(256)
    for (left, right) in tk.tokenset.merges:
        assert isinstance(left, bytes) and isinstance(right, bytes)
        assert left + right in tk.tokenset.tokens_by_str


def test_untokenize_survives_garbage():
    tk = _load(32)
    valid = tk.tokenize(b"ok then")
    garbage = [
        [],
        [0],                       # a lone high nibble
        [0, 0, 0],                 # odd nibble run
        valid[:-1],                # truncated
        list(range(32)),           # every id in order
        [31, 0, 17, 3, 3, 3, 9],   # arbitrary mixture
    ]
    for g in garbage:
        # Never raises, and never loses the valid text around it.
        out = tk.untokenize(list(valid) + list(g) + list(valid))
        assert isinstance(out, bytes)
        assert tk.untokenize(list(g)) is not None


def test_tokens_per_byte_matches_stats():
    """Measured processed-bytes-per-token lands near the corpus figure
    the tokenset stores (scanned_bytes / total_tokens). Ordinary
    English prose runs a few percent above the mixed corpus; anything
    outside +-15% means the merge loop is not doing its job."""
    processed = process2(_TEXT * 8)
    for ntokens in (32, 64, 128, 256):
        tk = _load(ntokens)
        ids = tk.tokenize_processed(processed)
        measured = len(processed) / len(ids)
        expected = tk.tokenset.avg_proc_bytes_per_token
        assert 0.85 < measured / expected < 1.15, (
            ntokens, measured, expected)


def test_merges_parse_as_string_pairs():
    ts = TokenSet.from_json_file(
        os.path.join(_TOKENS_DIR, "tokens.32.hexbpe.json"))
    assert ts.merges == [(b"e", b"\x16")]
    assert ts.type == "hexbpe" and ts.processing == "capswords2"


def test_int_triple_merges_pass_through():
    # SentencePiece-converted sets keep their [left, right, merged] id
    # triples; only 2-element (string pair) entries are reinterpreted.
    ts = TokenSet.from_json({
        "type": "bpe",
        "processing": "raw",
        "algorithm": "bpe",
        "tokens": ["a", "b", "ab"],
        "merges": [[0, 1, 2]],
        "stats": {"bytes_per_token": 1.0},
    })
    assert ts.merges == [[0, 1, 2]]
