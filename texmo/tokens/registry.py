import logging
import os

from .bits_tokenizer import (
    BitsTokenizer1,
    BitsTokenizer2,
    BitsTokenizer4,
    BytesTokenizer,
)
from .bpe_tokenizer import BpeTokenizer
from .fold_tokenizer import FoldTokenizer
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


def _normalize_name(name: str) -> str:
    """Accept the spec form of a tokenset name alongside the file
    basename: `tokens.32.hexbpe` (how specs and humans write it) and
    `tokens32_hexbpe` (the file name, what the codecs pass) resolve to
    the same tokenizer."""
    parts = name.split(".")
    if len(parts) == 3 and parts[0] == "tokens" and parts[1].isdigit():
        return f"tokens{parts[1]}_{parts[2]}"
    return name


def get_tokenizer(name: str):
    name = _normalize_name(name)
    if name in _TOKENIZERS:
        return _TOKENIZERS[name]

    if _TOKENS_DIR is None:
        logging.error(f"Tokens directory not set")
        return None

    path = os.path.join(_TOKENS_DIR, name + ".json")
    try:
        token_set = TokenSet.from_json_file(path)
        logging.info(f"Loaded tokenset from {path}")
        if token_set.type == "fold":
            # Forgetting sets: byte -> token is a pure table lookup,
            # and the generic Decoder can't represent many-to-one
            # groups (length-1 sequences would shadow the head char).
            tokenizer = FoldTokenizer(token_set)
        elif token_set.type == "hexbpe":
            # Checked before the generic bpe branch: hexbpe sets also
            # say `algorithm: "bpe"` (their merges are string pairs).
            # The SAMPLER uses the generic DP tokenizer over the same
            # vocabulary: measured 16x faster than the pure-Python
            # merge loop and 0.5% more compact, and no literature
            # supports merge-order segmentation helping the model.
            # The builder-exact BPE encoder (HexBpeTokenizer) remains
            # available for scripts; stats stay BPE-derived, a ~0.5%
            # skew on bytes_per_token we accept.
            tokenizer = Tokenizer(token_set)
        elif token_set.algorithm == "bpe":
            # BPE sets (converted SentencePiece vocabs) use the merge
            # loop; the DP tokenizer's dense suffix automaton would
            # also be prohibitively large at 256k pieces.
            tokenizer = BpeTokenizer(token_set)
        else:
            tokenizer = Tokenizer(token_set)
    except FileNotFoundError:
        # Not cached: the caller may fix the tokens dir (or add the
        # file) and retry within the same process.
        logging.warning(f"No tokenset file at {path}")
        return None
    _TOKENIZERS[name] = tokenizer

    return tokenizer


def set_tokens_dir(path: str):
    global _TOKENS_DIR
    if path is None:
        raise ValueError("Tokens directory cannot be None")
    _TOKENS_DIR = path
