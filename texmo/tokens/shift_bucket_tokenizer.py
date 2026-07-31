"""Tokenizer for "shift_bucket" tokensets (tokens32_shift).

A shift_bucket set spells every byte with exactly one of

- its own single-char token (the visible bytes and the two bucket
  faces),
- the numbered SHIFT marker followed by its host token (the produced
  bytes: 'A' = [shift, 'a'], '\\n' = [shift, ' '], ...),
- a bucket face on its own, which FORGETS which member it was (all
  digits decode as "0", every other byte as "*").

So encoding is a 256-entry table lookup with no choices to make: no
DP, no merge loop. The generic `Tokenizer` would in fact get this
WRONG -- the buckets' length-1 sequences would shadow their face
character in `Decoder.tokens_to_string`, the same reason lossy fold
sets need `FoldTokenizer` -- while the shift pairs make it too
lossless-looking for `FoldTokenizer`'s one-byte-one-token table.
This class covers the mixed case: partly lossless (the shift pairs),
partly forgetting (the buckets).

The forgotten information is priced by `stats.residual_bits_per_byte`
(the uniform log2(group size) charge over the corpus, computed by
scripts/make_shift32.py), which `TokenSet.byte_loss` adds to the
per-token cross-entropy so the set compares honestly with lossless
tokenizations; `stats.extra_weights` charges the 29 token selections
and the 29 stored shift pairs as model weights.

Decoding is deliberately total: a sampled model emits arbitrary ids,
so a shift with no usable host (another shift, a bucket face, the end
of the stream) simply drops the shift instead of raising.
"""
from .processing import apply_processing, undo_processing
from .tokenset import TokenSet


class ShiftBucketTokenizer(object):
    def __init__(self, tokenset: TokenSet):
        self.tokenset: TokenSet = tokenset

        markers = [t for t in tokenset.tokens if t.string is None]
        assert len(markers) == 1, (
            f"a shift_bucket set has exactly one marker: {markers}")
        self.shift_id: int = markers[0].id

        # Decode face: token id -> the byte it prints. A dict (not a
        # list) because untokenize must survive out-of-range ids from
        # a sampled model.
        face: dict[int, int] = {}
        # byte -> the ids it encodes as.
        table: list[tuple[int, ...] | None] = [None] * 256
        for token in tokenset.tokens:
            if token.string is None:
                continue
            assert len(token.string) == 1, (
                f"shift_bucket tokens are single bytes: {token}")
            face[token.id] = token.string[0]
            assert table[token.string[0]] is None, (
                f"byte {token.string[0]} has two tokens")
            table[token.string[0]] = (token.id,)

        # (shift id, host id) -> the byte the pair produces.
        pairs: dict[tuple[int, int], int] = {}
        for string, tokens in tokenset.sequences.items():
            assert len(string) == 1, (
                f"shift_bucket sequences spell one byte: {string!r}")
            ids = tuple(t.id for t in tokens)
            assert table[string[0]] is None, f"byte {string[0]} mapped twice"
            table[string[0]] = ids
            if len(ids) == 2:
                assert ids[0] == self.shift_id, (
                    f"a 2-token sequence must start with the shift: {ids}")
                pairs[ids] = string[0]
            else:
                assert len(ids) == 1, (
                    f"sequences are 1 or 2 tokens long: {ids}")
        assert all(t is not None for t in table), (
            "shift_bucket must cover all 256 bytes")

        self._table: list[tuple[int, ...]] = table
        self._face = face
        self._pairs = pairs

    # -- encoding ---------------------------------------------------------

    def tokenize(self, chunk: bytes) -> list[int]:
        chunk = apply_processing(self.tokenset.processing, chunk)
        return self.tokenize_processed(chunk)

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        table = self._table
        ids: list[int] = []
        for b in chunk:
            ids.extend(table[b])
        return ids

    # -- decoding ---------------------------------------------------------

    def untokenize(self, token_ids: list[int]) -> bytes:
        shift_id = self.shift_id
        pairs = self._pairs
        face = self._face
        out = bytearray()
        pending = False  # a shift is waiting for its host
        for token_id in token_ids:
            if pending:
                pending = False
                produced = pairs.get((shift_id, token_id))
                if produced is not None:
                    out.append(produced)
                    continue
                # No pair for this host: drop the shift and let the
                # id decode on its own (it may be a bucket face, or
                # another shift, or garbage).
            if token_id == shift_id:
                pending = True
                continue
            byte = face.get(token_id)
            if byte is not None:
                out.append(byte)
        # A trailing shift has no host: dropped, like any other
        # unusable one.
        return undo_processing(self.tokenset.processing, bytes(out))
