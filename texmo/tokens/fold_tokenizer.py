import numpy as np

from .processing import process, unprocess
from .tokenset import TokenSet


class FoldTokenizer(object):
    """Tokenizer for "fold" (forgetting) tokensets.

    Every byte maps to exactly one token (its group's head character),
    so tokenization is a single table lookup and untokenization prints
    head characters. The information the fold destroys is priced by
    the head-frequency model stored in the tokenset: within a group,
    the head byte has probability `head_freq[token]` and the remaining
    members split the rest equally. `residual_entropy[b]` is the
    resulting -log2 P(b | token) charge for byte b; its corpus average
    is `stats.residual_bits_per_byte`, which `TokenSet.byte_loss` adds
    to the per-token cross-entropy so fold models compare honestly
    with lossless tokenizations. The tokenset itself counts
    `stats.extra_weights` (one per stored frequency) toward the model.
    """

    def __init__(self, tokenset: TokenSet):
        self.tokenset = tokenset
        ntokens = len(tokenset.tokens)

        group_id = np.full(256, -1, dtype=np.int32)
        head_byte = np.zeros(ntokens, dtype=np.uint8)
        for token in tokenset.tokens:
            assert token.string is not None and len(token.string) == 1, (
                f"fold token must be a single byte: {token}")
            group_id[token.string[0]] = token.id
            head_byte[token.id] = token.string[0]
        for string, tokens in tokenset.sequences.items():
            assert len(string) == 1 and len(tokens) == 1, (
                f"fold sequences map one byte to one token: {string!r}")
            assert group_id[string[0]] == -1, (
                f"byte {string[0]} mapped twice")
            group_id[string[0]] = tokens[0].id
        assert (group_id >= 0).all(), "fold must cover all 256 bytes"

        self.groups = group_id
        self._head_byte = head_byte

        sizes = np.bincount(group_id, minlength=ntokens)
        charge = np.zeros(256)
        for token in tokenset.tokens:
            p = float(tokenset.head_freq[token.string.decode("utf-8")])
            head = token.string[0]
            charge[head] = -np.log2(p) if p > 0 else np.inf
            n_rest = sizes[token.id] - 1
            if n_rest:
                q = (1.0 - p) / n_rest
                rest = -np.log2(q) if q > 0 else np.inf
                members = np.flatnonzero(group_id == token.id)
                charge[members[members != head]] = rest
        self.residual_entropy = charge

    def tokenize(self, chunk: bytes) -> np.ndarray:
        if self.tokenset.processing == "capswords":
            chunk = process(chunk)
        return self.tokenize_processed(chunk)

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.groups[np.frombuffer(chunk, dtype=np.uint8)]

    def untokenize(self, tokens: list[int]) -> bytes:
        text = bytes(self._head_byte[np.asarray(tokens, dtype=np.int64)])
        if self.tokenset.processing == "capswords":
            text = unprocess(text)
        return text

    def chunk_residual_bits(self, chunk: bytes) -> float:
        """Exact fold charge for this chunk's actual bytes -- the
        per-slice alternative to the corpus-average constant."""
        return float(
            self.residual_entropy[np.frombuffer(chunk, dtype=np.uint8)]
            .sum())
