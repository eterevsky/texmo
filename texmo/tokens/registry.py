import logging
import mmap
import numpy as np
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

    def tokenize_ids(
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

        return np.frombuffer(string[start : start + l], dtype=np.uint8)

        # return list(string[start : start + l])

    def untokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)


_TOKENIZERS = {"tokens256_raw_all": LiteralBytesTokenizer()}
_TOKENS_DIR = None


def get_tokenizer(name: str):
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]

    if _TOKENS_DIR is None:
        logging.warning(f"Tokens directory not set")
        return None

    path = os.path.join(_TOKENS_DIR, name + ".json")
    try:
        token_set = TokenSet.from_json_file(path)
        logging.info(f"Loaded token set from {path}")
        tokenizer = Tokenizer(token_set)
    except FileNotFoundError:
        tokenizer = None
    _TOKENIZERS[name] = tokenizer

    return tokenizer


def set_tokens_dir(path: str):
    global _TOKENS_DIR
    _TOKENS_DIR = path
