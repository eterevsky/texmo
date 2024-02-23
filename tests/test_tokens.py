import logging
from unittest import TestCase

from texmo import tokens
from texmo.tokens import TokenSet, Tokenizer

logging.disable(level=logging.ERROR)


def tokenize(string, tokens, fallback_bits=4):
    tokenset = TokenSet(f"bits{fallback_bits}", "raw", {})

    for i in range(2**fallback_bits):
        tokenset.add_ext_token(i)

    for token in tokens:
        tokenset.add_token(token)

    for c in range(256):
        s = chr(c).encode("utf-8")
        if s in tokens:
            continue
        digits = []
        for _ in range(8 // fallback_bits):
            digits.append(c & (2**fallback_bits - 1))
            c >>= fallback_bits
        tokenset.add_sequence(s, [tokenset.tokens[d] for d in reversed(digits)])

    tokenizer = Tokenizer(tokenset)
    out = tokenizer.tokenize(string)
    result = [tokenset.tokens[t].value or tokenset.tokens[t].string or 0 for t in out]
    return result


class TokensTest(TestCase):
    def test_tokenize1(self):
        self.assertEqual(tokenize(b"ab", [b"a", b"b", b"ab"]), [b"ab"])

    def test_tokenize1_3(self):
        self.assertEqual(
            tokenize(b"ab ab ab", [b"a", b"b", b"ab"]),
            [b"ab", 2, 0, b"ab", 2, 0, b"ab"],
        )

    def test_tokenize3(self):
        self.assertEqual(
            tokenize(b"abc", [b"a", b"ab", b"bc"]),
            [b"a", b"bc"],
        )

    def test_tokenize3_3(self):
        self.assertEqual(
            tokenize(b"abc abc abc", [b"a", b"ab", b"bc"]),
            [b"a", b"bc", 2, 0, b"a", b"bc", 2, 0, b"a", b"bc"],
        )

    def test_tokenize4(self):
        self.assertEqual(
            tokenize(b"abcdefgh", [b"ab", b"cd", b"ef", b"gh", b"abcdefg"]),
            [b"abcdefg", 6, 8],
        )

    def test_tokenize5(self):
        self.assertEqual(
            tokenize(b"abcdefgh", [b"ab", b"cd", b"ef", b"gh", b"abcdefg"], 1),
            [b"ab", b"cd", b"ef", b"gh"],
        )

    def test_tokenize6(self):
        self.assertEqual(
            tokenize(
                b"abcdefg", [b"a", b"ab", b"bc", b"cd", b"de", b"ef", b"fg"]
            ),
            [b"a", b"bc", b"de", b"fg"],
        )

    def test_tokenize6_3(self):
        self.assertEqual(
            tokenize(
                b"abcdefg abcdefg abcdefg",
                [b"a", b"ab", b"bc", b"cd", b"de", b"ef", b"fg"],
            ),
            [
                b"a",
                b"bc",
                b"de",
                b"fg",
                2,
                0,
                b"a",
                b"bc",
                b"de",
                b"fg",
                2,
                0,
                b"a",
                b"bc",
                b"de",
                b"fg",
            ],
        )
