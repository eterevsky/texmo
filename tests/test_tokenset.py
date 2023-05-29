import logging
from unittest import TestCase

from texmo import tokens
from texmo.tokens import TokenSet

logging.disable(level=logging.ERROR)


class TokenSetTest(TestCase):
    def test_suffix1(self):
        token_set = TokenSet("fallback_dist", [], None, [1]*256)
        token_set._add_special_token(0)
        token_set.add_token(b"abc")
        token_set.add_token(b"bc")
        token_set.add_token(b"ab")
        token_set.add_token(b"a")
        token_set.add_token(b"ac")
        token_set.add_token(b"b")


        self.assertEqual(token_set.tokens_by_str[b"abc"].suffix.string, b"bc")
        self.assertEqual(token_set.tokens_by_str[b"bc"].suffix, ord("c"))
        self.assertEqual(token_set.tokens_by_str[b"ab"].suffix.string, b"b")
        self.assertEqual(token_set.tokens_by_str[b"a"].suffix, None)
        self.assertEqual(token_set.tokens_by_str[b"ac"].suffix, ord("c"))
        self.assertEqual(token_set.tokens_by_str[b"b"].suffix, None)
