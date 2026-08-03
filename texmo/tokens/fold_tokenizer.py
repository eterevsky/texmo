import numpy as np

from .processing import apply_processing, undo_processing
from .tokenset import TokenSet


class FoldTokenizer(object):
    """Tokenizer for "fold" (forgetting) tokensets.

    Every byte maps to exactly one token, so tokenization is a table
    lookup; untokenization prints each group's head character. This
    class exists for LOSSY sets only — a set that instead encodes
    rare bytes with marker/escape token sequences (tokens.64.shift)
    is lossless and just an ordinary `Tokenizer` set.

    The information the fold destroys is priced by the per-byte
    charge in `residual_entropy`: with a stored head frequency (the
    tokenset's `head_freq`, one weight per token) the head byte has
    probability p and the other group members split 1 - p equally;
    a group without a stored frequency is charged uniformly at
    log2(group size), which costs no weights. The corpus average is
    `stats.residual_bits_per_byte`, which `TokenSet.byte_loss` adds
    to the per-token cross-entropy so fold models compare honestly
    with lossless tokenizations; `stats.extra_weights` counts the
    stored frequencies toward the model.
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
            n = int(sizes[token.id])
            if n <= 1:
                continue
            head = token.string[0]
            members = np.flatnonzero(group_id == token.id)
            p = tokenset.head_freq.get(token.string.decode("utf-8"))
            if p is None:
                charge[members] = np.log2(n)
                continue
            p = float(p)
            charge[head] = -np.log2(p) if p > 0 else np.inf
            q = (1.0 - p) / (n - 1)
            rest = -np.log2(q) if q > 0 else np.inf
            charge[members[members != head]] = rest
        self.residual_entropy = charge

    def tokenize(self, chunk: bytes) -> np.ndarray:
        chunk = apply_processing(self.tokenset.processing, chunk)
        return self.tokenize_processed(chunk)

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.groups[np.frombuffer(chunk, dtype=np.uint8)]

    def untokenize(self, tokens: list[int]) -> bytes:
        text = bytes(self._head_byte[np.asarray(tokens, dtype=np.int64)])
        return undo_processing(self.tokenset.processing, text)

    def chunk_residual_bits(self, chunk: bytes) -> float:
        """Exact fold charge for this chunk's actual bytes -- the
        per-slice alternative to the corpus-average constant."""
        return float(
            self.residual_entropy[np.frombuffer(chunk, dtype=np.uint8)]
            .sum())
