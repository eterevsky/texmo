import logging
from unittest import TestCase

from texmo import tokens
from texmo.tokens import Tokenizer2

logging.disable(level=logging.ERROR)


class TokensTest(TestCase):
    def test_tokenize1(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"ab", [b"a", b"b", b"ab"], minimize_non_tokens=True
                )
            ),
            [b"ab"],
        )

    def test_tokenize1_3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"ab ab ab", [b"a", b"b", b"ab"], minimize_non_tokens=True
                )
            ),
            [b"ab", b" ", b"ab", b" ", b"ab"],
        )

    def test_tokenize2(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"ab", [b"a", b"b", b"ab"], minimize_non_tokens=False
                )
            ),
            [b"ab"],
        )

    def test_tokenize2_3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"ab ab ab", [b"a", b"b", b"ab"], minimize_non_tokens=False
                )
            ),
            [b"ab", b" ", b"ab", b" ", b"ab"],
        )

    def test_tokenize3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abc", [b"a", b"ab", b"bc"], minimize_non_tokens=True
                )
            ),
            [b"a", b"bc"],
        )

    def test_tokenize3_3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abc abc abc",
                    [b"a", b"ab", b"bc"],
                    minimize_non_tokens=True,
                )
            ),
            [b"a", b"bc", b" ", b"a", b"bc", b" ", b"a", b"bc"],
        )

    def test_tokenize4(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdef",
                    [b"ab", b"cd", b"ef", b"abcde"],
                    minimize_non_tokens=False,
                )
            ),
            [b"abcde", b"f"],
        )

    def test_tokenize4_3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdef abcdef abcdef",
                    [b"ab", b"cd", b"ef", b"abcde"],
                    minimize_non_tokens=False,
                )
            ),
            [b"abcde", b"f", b" ", b"abcde", b"f", b" ", b"abcde", b"f"],
        )

    def test_tokenize5_3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdef abcdef abcdef",
                    [b"ab", b"cd", b"ef", b"abcde"],
                    minimize_non_tokens=True,
                )
            ),
            [
                b"ab",
                b"cd",
                b"ef",
                b" ",
                b"ab",
                b"cd",
                b"ef",
                b" ",
                b"ab",
                b"cd",
                b"ef",
            ],
        )

    def test_tokenize6(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdefg",
                    [b"a", b"ab", b"bc", b"cd", b"de", b"ef", b"fg"],
                    minimize_non_tokens=True,
                )
            ),
            [b"a", b"bc", b"de", b"fg"],
        )

    def test_tokenize6(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdefg abcdefg abcdefg",
                    [b"a", b"ab", b"bc", b"cd", b"de", b"ef", b"fg"],
                    minimize_non_tokens=True,
                )
            ),
            [
                b"a",
                b"bc",
                b"de",
                b"fg",
                b" ",
                b"a",
                b"bc",
                b"de",
                b"fg",
                b" ",
                b"a",
                b"bc",
                b"de",
                b"fg",
            ],
        )

    def test_tokenizer2(self):
        tokenizer = Tokenizer2([b"a", b"b", b"ab"])
        self.assertEqual(tokenizer._tokens[b"ab"].suffix_token.string, b"b")
        self.assertEqual(tokenizer._tokens[b"a"].suffix_token, None)
        self.assertEqual(tokenizer._tokens[b"b"].suffix_token, None)

    def test_tokenizer2_abc(self):
        tokenizer = Tokenizer2([b"a", b"b", b"c", b"bc", b"abc"])
        self.assertEqual(tokenizer._tokens[b"abc"].suffix_token.string, b"bc")
        self.assertEqual(tokenizer._tokens[b"a"].suffix_token, None)
        self.assertEqual(tokenizer._tokens[b"b"].suffix_token, None)
