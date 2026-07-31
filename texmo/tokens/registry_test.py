"""Tests for tokenizer registry name resolution.

The tokens dir is registered by texmo/conftest.py, so the real sets
in tokens/ are loadable here.
"""
from .fold_tokenizer import FoldTokenizer
from .registry import _normalize_name, get_tokenizer
from .shift_bucket_tokenizer import ShiftBucketTokenizer
from .tokenizer import Tokenizer


def test_spec_form_and_file_form_resolve_to_the_same_tokenizer():
    # `tokens.32.hexbpe` is how specs (and humans at the CLI) write
    # it; `tokens32_hexbpe` is the file basename the codecs pass.
    a = get_tokenizer("tokens.32.hexbpe")
    b = get_tokenizer("tokens32_hexbpe")
    # The sampler default for hexbpe sets is the generic DP tokenizer
    # (the builder-exact BPE encoder stays a scripts-only capability).
    assert isinstance(a, Tokenizer)
    assert a is b  # one cache entry, not two parallel loads


def test_hexbpe_dp_round_trips_through_capswords2():
    tok = get_tokenizer("tokens.256.hexbpe")
    text = b"The Quick brown fox, 42 iPhones!\nNew paragraph."
    ids = tok.tokenize(text)
    assert max(ids) < 256
    assert tok.untokenize(ids) == text


def test_type_routes_to_the_matching_tokenizer_class():
    # The `type` field, not the name, picks the class: shift_bucket
    # sets are neither pure-fold tables nor DP-tokenizable.
    assert isinstance(get_tokenizer("tokens.32.shift"), ShiftBucketTokenizer)
    assert isinstance(get_tokenizer("tokens.32.fold"), FoldTokenizer)
    # tokens64_shift is type "shift" (lossless, hex escapes): the
    # generic DP tokenizer, unaffected by the shift_bucket branch.
    assert isinstance(get_tokenizer("tokens.64.shift"), Tokenizer)


def test_normalize_leaves_other_names_alone():
    for name in ("bits.4", "bytes", "tokens32_hexbpe",
                 "tokens.64.shift.extra"):
        assert _normalize_name(name) == name
    assert _normalize_name("tokens.256000.gemma") == "tokens256000_gemma"


def test_missing_set_returns_none_and_is_not_cached():
    assert get_tokenizer("tokens.32.nosuchset") is None
    # A later call probes the disk again rather than serving a stale
    # cached None (the dir or file may have been fixed in between).
    assert get_tokenizer("tokens32_nosuchset") is None


def test_builtin_bit_tokenizers_pass_through():
    assert get_tokenizer("bits.4") is get_tokenizer("bits.4")
    assert get_tokenizer("bytes") is get_tokenizer("bits.8")
