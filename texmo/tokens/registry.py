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


class BytesTokenSet:
    name = 'bytes'
    processing = 'raw'
    avg_proc_bytes_per_token = 1
    avg_bytes_per_token = 1


class LiteralBytesTokenizer:
    tokenset = BytesTokenSet()

    def tokenize(self, chunk: bytes) -> list[int]:
        return list(chunk)

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)


class LiteralBitsTokenizer(object):
    def __init__(self):
        pass
        # self.token_set = TokenSet.build("fallback_bits", 1, "raw", [], {"literal_cost": 8, "literal_dist_entropy": 0}, bytes_per_token=0.125)

    def tokenize(
        self,
        string: bytes | mmap.mmap,
        start=0,
        max_tokens=None,
        max_bytes=None,
    ) -> np.ndarray:
        ids = self.tokenize_ids(string, start, max_tokens, max_bytes)
        return [self.token_set.tokens[i] for i in ids]

    def tokenize_ids(
        self,
        string: bytes | mmap.mmap,
        start=0,
        max_tokens=None,
        max_bytes=None,
    ) -> np.ndarray:
        if max_tokens is not None:
            l = (max_tokens + 7) // 8
        elif max_bytes is not None:
            l = max_bytes
            assert start + l <= len(string)
        else:
            l = len(string) - start

        b = np.frombuffer(string[start : start + l], dtype=np.uint8)
        res = np.unpackbits(b)

        if max_tokens is not None and max_tokens < 8:
            res = res[0:max_tokens]

        if max_bytes is not None:
            assert len(res) == max_bytes * 8

        return res, 0

    def untokenize(self, ids: list[int]) -> bytes:
        b = np.packbits(ids)
        return bytes(b)


_TOKENIZERS = {
    # "tokens2_raw_bits1": LiteralBitsTokenizer(),
    "bytes": LiteralBytesTokenizer()
}
_TOKENS_DIR = None


def get_tokenizer(name: str):
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]

    if _TOKENS_DIR is None:
        logging.error(f"Tokens directory not set")
        return None

    path = os.path.join(_TOKENS_DIR, name + ".json")
    try:
        token_set = TokenSet.from_json_file(path)
        logging.info(f"Loaded tokenset from {path}")
        tokenizer = Tokenizer(token_set)
    except FileNotFoundError:
        tokenizer = None
    _TOKENIZERS[name] = tokenizer

    return tokenizer


def set_tokens_dir(path: str):
    global _TOKENS_DIR
    if path is None:
        raise ValueError("Tokens directory cannot be None")
    _TOKENS_DIR = path
