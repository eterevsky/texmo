from typing import Self

import mmap
import regex

from ..common import INF
from .tokenset import Token, TokenSet


class SuffixState(object):
    def __init__(self, suffix: bytes):
        self.suffix: bytes = suffix
        # The longest token which is a suffix of the current suffix.
        self.token: Token | int = None
        self.next: list[Self] = [None] * 256


def _populate_suffix_states(token_set: TokenSet):
    states = {}

    states[b""] = SuffixState(b"")

    # Add empty states for literals
    for i in range(256):
        s = bytes([i])
        states[s] = SuffixState(s)

    # Add states for all prefixes of all tokens (except len 1, since it's
    # already covered)
    for token in token_set.tokens:
        if token.string is not None:
            for l in range(2, len(token.string) + 1):
                s = token.string[:l]
                if s not in states:
                    states[s] = SuffixState(s)

    for state in states.values():
        # Looking for the suffix token
        for start in range(len(state.suffix)):
            s = state.suffix[start:]
            token = token_set.tokens_by_str.get(s)
            if token is not None:
                state.token = token
                break
        if state.token is None and state.suffix:
            state.token = state.suffix[-1]

        # Populating next
        for byte in range(256):
            s = state.suffix + bytes([byte])

            for start in range(len(s)):
                next_state = states.get(s[start:])
                if next_state is not None:
                    state.next[byte] = next_state
                    break

            assert state.next[byte] is not None

    return states


WORD_BOUNDARY = regex.compile(r"(?<=\P{L})(?=\p{L})|(?<=\p{L})(?=\P{L})")
CAPITALIZED_MARKER = "\x14"
ALLCAPS_MARKER = "\x15"
WORD_MARKER = "\x16"


class Tokenizer(object):
    def __init__(self, token_set: TokenSet):
        self._token_set: TokenSet = token_set
        self._suffix_states: dict[bytes, SuffixState] = _populate_suffix_states(
            token_set
        )
        self._literals = token_set.literal_encodings()
        self._mark_caps = "caps" in token_set.processing
        self._mark_words = "words" in token_set.processing

    def tokenize(
        self, string: bytes|mmap.mmap, start=0, max_tokens=None, max_bytes=None
    ) -> list[Token]:
        suffix_state = self._suffix_states[b""]
        cost_state = [(0, None)]

        finished = False

        for chunk in self._iterate_bytes(string, start, max_tokens, max_bytes):
            if finished:
                break
            for byte in chunk:
                if max_bytes is not None and len(cost_state) - 1 >= max_bytes:
                    finished = True
                    break
                suffix_state = suffix_state.next[byte]
                token = suffix_state.token

                if isinstance(token, int):
                    best_cost = cost_state[-1][0] + len(self._literals[token])
                    best_token = token
                else:
                    best_cost = cost_state[-len(token.string)][0] + 1
                    best_token = token

                    token = token.suffix
                    while isinstance(token, Token):
                        cost = cost_state[-len(token.string)][0] + 1
                        if cost < best_cost:
                            best_cost = cost
                            best_token = token
                        token = token.suffix

                    if token is not None:
                        cost = cost_state[-1][0] + len(self._literals[token])
                        if cost < best_cost:
                            best_cost = cost
                            best_token = token

                cost_state.append((best_cost, best_token))
                if max_tokens is not None and best_cost >= max_tokens + 10:
                    finished = True
                    break

        tokens = []
        pos = len(cost_state) - 1
        while pos > 0:
            token = cost_state[pos][1]

            if isinstance(token, Token):
                tokens.append(token)
                pos -= len(token.string)
            else:
                tokens.extend(self._literals[token])
                pos -= 1

        tokens.reverse()
        if max_tokens is not None and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokens

    def untokenize(self, tokens: list[int]) -> bytes:
        chunks = []
        sub_byte_buf = []

        for token_id in tokens:
            token = self._token_set.tokens[token_id]
            if token.string is not None:
                if sub_byte_buf:
                    chunks.append(b"?")
                    sub_byte_buf.clear()
                chunks.append(token.string)
            elif self._token_set.type in (
                "fallback_distribution",
                "str_with_fallback_distribution",
            ):
                chunks.append(b"?")
            elif self._token_set.type in (
                "str_with_fallback_bits",
                "fallback_bits",
            ):
                sub_byte_buf.append(token.value)
                if len(sub_byte_buf) == 8 // self._token_set.fallback_bits:
                    value = 0
                    for v in sub_byte_buf:
                        value <<= self._token_set.fallback_bits
                        value += v
                    chunks.append(bytes([value]))
                    sub_byte_buf.clear()
            else:
                assert False, "Unsupported TokenSet type"

        return b"".join(chunks)

    def _process(self, s: str) -> str:
        out = []
        i = 0
        words = WORD_BOUNDARY.split(s)

        for i, word in enumerate(words):
            if not word:
                continue
            if word.isalpha():
                if (
                    self._mark_caps
                    and word[0].isupper()
                    and (len(word) == 1 or word[1:].islower())
                ):
                    out.append(CAPITALIZED_MARKER)
                    out.append(word.lower())
                elif self._mark_caps and word.isupper():
                    out.append(ALLCAPS_MARKER)
                    out.append(word.lower())
                else:
                    out.append(word)
                if self._mark_words:
                    out.append(WORD_MARKER)
            else:
                if self._mark_words and word == " " and 0 < i < len(words) - 1:
                    pass
                else:
                    out.append(word)

        return "".join(out)

    def _iterate_bytes(
        self, data: bytes|mmap.mmap, start: int, max_tokens: int, max_bytes: int
    ):
        while start < len(data):
            if max_bytes is None and max_tokens is None:
                l = len(data) - start
            elif max_bytes is not None:
                assert max_tokens is None
                l = max_bytes
            else:
                assert max_tokens is not None
                l = max_tokens // 2

            end = start + l
            while end < len(data) and 128 <= data[end] < 192:
                end += 1

            string = data[start:end]

            if self._mark_caps or self._mark_words:
                string = string.decode("utf-8")
                string = self._process(string).encode("utf-8")

            yield string

            start = end
