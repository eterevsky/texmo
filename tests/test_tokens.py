import logging
from unittest import TestCase

from texmo import tokens

logging.disable(level=logging.ERROR)


class TokensTest(TestCase):
    def test_find_frequent_substrings(self):
        s = tokens.find_frequent_substrings(b"abacabadaba", 5)
        self.assertEqual(s, {b"a": 6, b"ab": 3, b"b": 3, b"ba": 3, b"aba": 3})

    def test_tokenize1(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"ab", [b"a", b"b", b"ab"], minimize_non_tokens=True
                )
            ),
            [b"ab"],
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

    def test_tokenize3(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abc", [b"a", b"ab", b"bc"], minimize_non_tokens=True
                )
            ),
            [b"a", b"bc"],
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

    def test_tokenize5(self):
        self.assertEqual(
            list(
                tokens.tokenize(
                    b"abcdef",
                    [b"ab", b"cd", b"ef", b"abcde"],
                    minimize_non_tokens=True,
                )
            ),
            [b"ab", b"cd", b"ef"],
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
