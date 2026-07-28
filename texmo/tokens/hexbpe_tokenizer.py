"""Byte-level BPE tokenizer for "hexbpe" tokensets.

A hexbpe set is built by the Rust generator (`type: "hexbpe"`) and is
lossless by construction:

- ids 0..15 are NIBBLE tokens (no string of their own). Every byte the
  set did not select for a token of its own is spelled as the two
  nibbles of its hex code, listed in `sequences` as
  `{"string": <byte>, "tokens": [hi, lo]}`.
- the remaining tokens are strings: the selected single bytes plus the
  pieces produced by the merge table.
- `merges` is an ordered list of STRING PAIRS; the rank is the list
  index and the merged piece is the concatenation of the pair, which
  the generator guarantees is itself a token.

Encoding is the ordinary GPT-2/SentencePiece merge loop -- each pass
finds the lowest-rank adjacent pair present and applies it everywhere,
so the result depends on merge ORDER, not on what a DP would find
cheapest. It is deliberately NOT the `BpeTokenizer` in
bpe_tokenizer.py: that one is char/str-based with a priority scan and
the gemma U+2581 conflation, none of which apply here.

Nibble ids never appear in `merges` (they have no string), so a
hex-escaped byte is a natural barrier: merges never cross into or out
of one.

Decoding reuses the generic `Decoder` from tokenizer.py -- it already
knows how to fold multi-token sequences (the nibble pairs) back into
their byte and how to drop an undecodable head token instead of
raising, which matters because a sampled model emits arbitrary ids.
"""
from .processing import apply_processing, undo_processing
from .tokenizer import Decoder
from .tokenset import TokenSet


class HexBpeTokenizer(object):
    def __init__(self, tokenset: TokenSet):
        self.tokenset: TokenSet = tokenset

        # byte -> the ids it starts out as: its own single-byte token
        # when the set selected that byte, else its two hex nibbles.
        by_str = tokenset.tokens_by_str
        self._atoms: list[tuple[int, ...]] = []
        for b in range(256):
            token = by_str.get(bytes([b]))
            if token is not None:
                self._atoms.append((token.id,))
                continue
            seq = tokenset.sequences.get(bytes([b]))
            assert seq is not None, f"byte {b} is not representable"
            self._atoms.append(tuple(t.id for t in seq))

        # (left_id, right_id) -> (rank, merged_id). Piece strings are
        # unique in a hexbpe set, so `tokens_by_str` is unambiguous.
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right) in enumerate(tokenset.merges):
            merged = by_str.get(left + right)
            assert merged is not None, (
                f"merge {left!r}+{right!r} has no token")
            self._merge_ranks.setdefault(
                (by_str[left].id, by_str[right].id), (rank, merged.id))

        self._decoder = Decoder(tokenset)

    # -- encoding ---------------------------------------------------------

    def _bpe(self, ids: list[int]) -> list[int]:
        ranks = self._merge_ranks
        while len(ids) > 1:
            best_rank = None
            best_pair = None
            best_merged = None
            for a, b in zip(ids, ids[1:]):
                rm = ranks.get((a, b))
                if rm is not None and (
                        best_rank is None or rm[0] < best_rank):
                    best_rank, best_merged = rm
                    best_pair = (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            out: list[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_merged)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def tokenize(self, chunk: bytes) -> list[int]:
        chunk = apply_processing(self.tokenset.processing, chunk)
        return self.tokenize_processed(chunk)

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        atoms = self._atoms
        ids: list[int] = []
        for b in chunk:
            ids.extend(atoms[b])
        return self._bpe(ids)

    # -- decoding ---------------------------------------------------------

    def untokenize(self, token_ids: list[int]) -> bytes:
        text = self._decoder.decode(list(token_ids))
        return undo_processing(self.tokenset.processing, text)
