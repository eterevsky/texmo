import logging
import os

from .bits_tokenizer import (
    BitsTokenizer1,
    BitsTokenizer2,
    BitsTokenizer4,
    BytesTokenizer,
)
from .tokenizer import Tokenizer
from .tokenset import TokenSet


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



_TOKENIZERS = {
    'bits.1': BitsTokenizer1(),
    'bits.2': BitsTokenizer2(),
    'bits.4': BitsTokenizer4(),
    'bits.8': BytesTokenizer(),
}
_TOKENIZERS['bytes'] = _TOKENIZERS['bits.8']
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
