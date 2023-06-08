import logging
import mmap
import os

from .tokenset import TokenSet, Token
from .tokenizer import Tokenizer


def _build_literal_bytes() -> TokenSet:
    token_set = TokenSet(
        "all",
        "raw",
        None,
        {"literal_cost": 256, "literal_dist_entropy": 0},
        bytes_per_token=1,
    )
    for i in range(256):
        token_set.add_token(bytes([i]))
    return token_set


class LiteralBytesTokenizer(object):
    def __init__(self):
        self.token_set = _build_literal_bytes()

    def tokenize(
        self,
        string: bytes | mmap.mmap,
        start=0,
        max_tokens=None,
        max_bytes=None,
    ) -> list[Token]:
        if max_tokens is not None:
            l = max_tokens
        elif max_bytes is not None:
            l = max_bytes
        else:
            l = len(string) - start

        return [self.token_set.tokens[b] for b in string[start : start + l]]

    def untokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)


_TOKENIZERS = {"tokens256_raw_all": LiteralBytesTokenizer()}

_TOKENS_DIR = None


def get_tokenizer(name: str):
    tokenizer = _TOKENIZERS.get(name)
    if tokenizer is not None:
        return tokenizer

    if _TOKENS_DIR is None:
        return None

    path = os.path.join(_TOKENS_DIR, name + ".json")
    logging.info(f"Loading token set from {path}")
    token_set = TokenSet.from_json_file(path)
    tokenizer = Tokenizer(token_set)
    _TOKENIZERS[name] = tokenizer

    return tokenizer


def set_tokens_dir(path: str):
    _TOKENS_DIR = path
