"""SentencePiece-BPE-compatible tokenizer for converted tokensets
(algorithm == "bpe", e.g. tokens256000_gemma).

Encode pipeline, matching the HuggingFace/SentencePiece reference:

  1. Priority scan on the RAW input: leftmost-longest match of the
     `priority_tokens` strings (newline/space runs, HTML tags,
     <unusedN>, ...) and the specials' private-use chars. The
     reference's added tokens are all `normalized=False`, i.e. matched
     BEFORE space normalization -- so the space-run tokens only match
     literal U+2581 runs in the input, never real spaces (real space
     runs go through BPE merges instead). The scan therefore uses the
     original U+2581-form strings, reconstructed from the converted
     ones. BPE never sees matched tokens (most have no merge path).
  2. Per remaining text fragment: apply the U+2581 -> " " conflation
     (the vocab was converted the same way), split into unicode-char
     atoms -- the piece whose string is that char, or its UTF-8 bytes
     as byte-fallback tokens -- then repeatedly merge the adjacent
     pair with the lowest merge rank (GPT-2-style loop over the
     id-triple merge table).
  3. Optionally prepend <bos> (`add_bos`; defaults to the tokenset's
     flag). Use add_bos=False when tokenizing a continuation rather
     than a sequence start.

Decode is plain concatenation (tokens carry their literal bytes,
spaces included); control tokens render as their private-use chars, so
any encode(decode(ids)) round-trips through ordinary strings.

Deliberate divergence from HF: a special's NAME as literal text (e.g.
"<bos>") is NOT matched -- controls are reachable only via their
private-use chars. HF matches the names; we prefer the injection-safe
behavior. Parity fixtures avoid special names.

Performance note: the merge loop is the simple O(n^2)-per-fragment
reference implementation -- fine for prompts and tests; corpus-scale
tokenization wants the heap version (or the Rust port) later.
"""
from .tokenset import TokenSet


class BpeTokenizer(object):
    def __init__(self, tokenset: TokenSet):
        assert tokenset.algorithm == "bpe"
        self.tokenset = tokenset

        # id -> output bytes, total over all ids: pieces/priority
        # tokens decode to their strings, specials to their PUA chars.
        self._bytes_by_id: list[bytes] = [
            t.string if t.string is not None else b""
            for t in tokenset.tokens
        ]
        self._special_by_id: dict[int, str] = {}
        for s in tokenset.specials:
            self._bytes_by_id[s["id"]] = s["char"].encode("utf-8")
            self._special_by_id[s["id"]] = s["name"]

        # Atom maps. Char pieces are the string-form tokens of exactly
        # one unicode char; several ids may share a byte string with a
        # byte-fallback token -- the char piece wins for atoms, byte
        # tokens are only reachable through fallback (mirrors the
        # reference's structural distinction).
        self._char_piece: dict[str, int] = {}
        self._byte_token: dict[int, int] = {}
        for t in tokenset.tokens:
            if t.string is None:
                continue
            if t.is_byte:
                assert len(t.string) == 1
                self._byte_token.setdefault(t.string[0], t.id)
            else:
                try:
                    s = t.string.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if len(s) == 1:
                    self._char_piece.setdefault(s, t.id)

        # (left_id, right_id) -> (rank, merged_id).
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (left, right, merged) in enumerate(tokenset.merges):
            self._merge_ranks.setdefault((left, right), (rank, merged))

        # Priority strings (+ special PUA chars), grouped by first char
        # and sorted longest-first for the leftmost-longest scan. The
        # scan runs on the RAW input (reference added tokens are
        # normalized=False), so strings are reconstructed to their
        # original U+2581 form: the conversion " " <-> U+2581 is a
        # bijection on pieces.
        self._prio_by_first: dict[str, list[tuple[str, int]]] = {}
        raw_form = (
            (lambda s: s.replace(" ", "▁"))
            if tokenset.processing == "gemma" else (lambda s: s)
        )
        entries = []
        for tid in tokenset.priority_tokens:
            entries.append(
                (raw_form(self._bytes_by_id[tid].decode("utf-8")), tid))
        for s in tokenset.specials:
            entries.append((s["char"], s["id"]))
        for string, tid in entries:
            self._prio_by_first.setdefault(string[0], []).append(
                (string, tid))
        for lst in self._prio_by_first.values():
            lst.sort(key=lambda e: len(e[0]), reverse=True)

    # -- encoding ---------------------------------------------------------

    def _scan_priority(self, text: str) -> list:
        """Split text into str fragments and matched priority ids
        (leftmost-longest, like the reference's added-token scan)."""
        segments: list = []
        buf: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for string, tid in self._prio_by_first.get(text[i], ()):
                if text.startswith(string, i):
                    matched = (string, tid)
                    break
            if matched is None:
                buf.append(text[i])
                i += 1
            else:
                if buf:
                    segments.append("".join(buf))
                    buf.clear()
                segments.append(matched[1])
                i += len(matched[0])
        if buf:
            segments.append("".join(buf))
        return segments

    def _atoms(self, fragment: str) -> list[int]:
        ids: list[int] = []
        for ch in fragment:
            piece = self._char_piece.get(ch)
            if piece is not None:
                ids.append(piece)
            else:
                for b in ch.encode("utf-8"):
                    ids.append(self._byte_token[b])
        return ids

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

    def encode(self, text: str, add_bos: bool | None = None) -> list[int]:
        out: list[int] = []
        if add_bos is None:
            add_bos = self.tokenset.add_bos
        if add_bos:
            out.append(self.tokenset.bos_id)
        gemma = self.tokenset.processing == "gemma"
        for seg in self._scan_priority(text):
            if isinstance(seg, int):
                out.append(seg)
            else:
                if gemma:
                    # The reference's U+2581/space conflation: after
                    # the raw-form priority scan, remaining U+2581s
                    # are indistinguishable from spaces.
                    seg = seg.replace("▁", " ")
                out.extend(self._bpe(self._atoms(seg)))
        return out

    # -- Tokenizer-compatible surface for the sampler ---------------------

    def tokenize(self, chunk: bytes) -> list[int]:
        return self.tokenize_processed(chunk)

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        # Corpus chunks are continuations, not sequence starts: no bos.
        # The sampler trims to UTF-8 boundaries; ignore stray invalid
        # bytes defensively.
        return self.encode(
            chunk.decode("utf-8", errors="ignore"), add_bos=False)

    # -- decoding ----------------------------------------------------------

    def untokenize(self, token_ids: list[int]) -> bytes:
        return b"".join(self._bytes_by_id[t] for t in token_ids)

    def decode(self, token_ids: list[int]) -> str:
        return self.untokenize(token_ids).decode("utf-8", errors="replace")
