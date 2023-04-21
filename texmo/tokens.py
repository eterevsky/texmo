from collections import deque
from collections.abc import Iterable
import math
import mmap
from operator import itemgetter
import os
import sys
from typing import Self

import numpy as np


def _sort_and_prune(substrings: dict[bytes, int], n: int) -> dict[bytes, int]:
    pairs = list(substrings.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    return dict(pairs[:n])


def find_frequent_substrings(data: bytes, n: int) -> dict[bytes, int]:
    l = 1
    substrings = {}
    while l <= len(data):
        print(f"Gathering substrings of length {l}")
        for i in range(0, len(data) - l + 1):
            if l == 1 or data[i : i + l - 1] in substrings:
                s = data[i : i + l]
                try:
                    substrings[s] += 1
                except KeyError:
                    substrings[s] = 1

        ss = len(substrings)
        print(f"Substrings before pruning: {ss}")

        substrings = _sort_and_prune(substrings, n)

        max_l = 0
        for s in substrings.keys():
            if len(s) > max_l:
                max_l = len(s)

        if max_l < l:
            break

        l += 1

    return substrings


class PositionState(object):
    def __init__(
        self, last_jump, last_token, lo_jump, lo_token, tokens, non_tokens
    ):
        self.last_jump = last_jump
        self.last_token = last_token
        self.lo_jump = lo_jump
        self.lo_token = lo_token
        self.tokens = tokens
        self.non_tokens = non_tokens

    def __repr__(self):
        return (
            f"PositionState(last_jump={self.last_jump}, "
            + f"last_token={self.last_token}, lo_jump={self.lo_jump}, "
            + f"lo_token={self.lo_token}, tokens={self.tokens}, "
            + f"non_tokens={self.non_tokens})"
        )

    @staticmethod
    def new_pos(prev_state: Self, last_token: int, init_lo_jump: bool) -> Self:
        if init_lo_jump:
            lo_jump = 1
            lo_token = last_token
        else:
            lo_jump = prev_state.lo_jump
            lo_token = prev_state.lo_token

        if last_token >= 0:
            tokens = prev_state.tokens + 1
            non_tokens = prev_state.non_tokens
        else:
            tokens = prev_state.tokens
            non_tokens = prev_state.non_tokens + 1

        return PositionState(
            1, last_token, lo_jump, lo_token, tokens, non_tokens
        )


class Tokenizer(object):
    def __init__(self, tokens: list[bytes], minimize_non_tokens: bool):
        self._tokens = {}
        self._byte_tokens = [i - 256 for i in range(256)]
        for i, t in enumerate(tokens):
            self._tokens[t] = i
            if len(t) == 1:
                self._byte_tokens[t[0]] = i
        self._max_token_len = max(len(t) for t in self._tokens.keys())
        self._minimize_non_tokens = minimize_non_tokens

    def _tokens_better(
        self,
        new_tokens: int,
        new_non_tokens: int,
        old_tokens: int,
        old_non_tokens: int,
    ) -> bool:
        if self._minimize_non_tokens:
            return new_non_tokens < old_non_tokens or (
                new_non_tokens == old_non_tokens and
                new_tokens < old_tokens
            )
        else:
            return new_non_tokens + new_tokens < old_non_tokens + old_tokens

    def _consume_lo_token(self, state):
        lo_jump = state[-1].lo_jump
        for _ in range(lo_jump):
            state.popleft()
        state[0].lo_jump = None
        state[0].lo_token = None

        steps_with_same_lo_jump = 0
        for pos in range(1, len(state)):
            pos_state = state[pos]
            if pos_state.last_jump > pos:
                pos_state.lo_jump = None
                steps_with_same_lo_jump = 0
            elif pos_state.last_jump == pos:
                pos_state.lo_jump = pos_state.last_jump
                pos_state.lo_token = pos_state.last_token
                steps_with_same_lo_jump = 1
            else:
                prev_state = state[pos - pos_state.last_jump]
                pos_state.lo_jump = prev_state.lo_jump
                pos_state.lo_token = prev_state.lo_token
                if pos_state.lo_jump == state[pos - 1].lo_jump:
                    steps_with_same_lo_jump += 1
                else:
                    steps_with_same_lo_jump = 1

        return steps_with_same_lo_jump

    def _yield_remaining_tokens(self, state) -> Iterable[int]:
        tokens = []
        pos = len(state) - 1
        while pos > 0:
            cur_state = state[pos]
            tokens.append(cur_state.last_token)

            pos -= cur_state.last_jump
            assert pos >= 0

            if cur_state.last_token >= 0:
                assert state[pos].tokens + 1 == cur_state.tokens
                assert state[pos].non_tokens == cur_state.non_tokens
            else:
                assert state[pos].tokens == cur_state.tokens
                assert state[pos].non_tokens + 1 == cur_state.non_tokens

        return reversed(tokens)

    def tokenize(self, data: bytes) -> Iterable[int]:
        state = deque([PositionState(None, None, None, None, 0, 0)])
        lo = 0
        steps_with_same_lo_jump = 0

        for hi in range(1, len(data) + 1):
            cur_state = PositionState.new_pos(
                state[-1], self._byte_tokens[data[hi - 1]], len(state) == 1
            )
            for jump in range(2, min(self._max_token_len, len(state)) + 1):
                token = self._tokens.get(data[hi - jump : hi])
                if token is None:
                    continue
                prev_state = state[-jump]
                if self._tokens_better(
                    prev_state.tokens + 1,
                    prev_state.non_tokens,
                    cur_state.tokens,
                    cur_state.non_tokens,
                ):
                    cur_state.last_jump = jump
                    cur_state.last_token = token
                    if jump == len(state):
                        assert prev_state.lo_jump is None
                        cur_state.lo_jump = jump
                        cur_state.lo_token = token
                    else:
                        assert prev_state.lo_jump is not None
                        cur_state.lo_jump = prev_state.lo_jump
                        cur_state.lo_token = prev_state.lo_token
                    cur_state.tokens = prev_state.tokens + 1
                    cur_state.non_tokens = prev_state.non_tokens

            if cur_state.lo_jump == state[-1].lo_jump:
                steps_with_same_lo_jump += 1
            else:
                steps_with_same_lo_jump = 1

            state.append(cur_state)

            if steps_with_same_lo_jump >= self._max_token_len:
                assert state[state[-1].lo_jump].last_token == state[-1].lo_token
                yield state[-1].lo_token
                lo += state[-1].lo_jump
                steps_with_same_lo_jump = self._consume_lo_token(state)

        for token in self._yield_remaining_tokens(state):
            yield token


def tokenize(
    data: bytes, tokens: list[bytes], minimize_non_tokens=True
) -> Iterable[bytes]:
    tokenizer = Tokenizer(tokens, minimize_non_tokens)
    for token_id in tokenizer.tokenize(data):
        if token_id >= 0:
            yield tokens[token_id]
        else:
            yield bytes([token_id + 256])


def calculate_tokens_entropy(
    total_size: int, token_counts: dict[bytes, int], non_tokens: int
):
    tokens_entropy = 0
    for count in token_counts.values():
        entropy = -math.log2(count / total_size)
        tokens_entropy += entropy * count

    if non_tokens > 0:
        non_tokens_entropy = (
            -math.log2(non_tokens / total_size) + 8
        ) * non_tokens
    else:
        non_tokens_entropy = 0

    tokens_entropy /= 8
    non_tokens_entropy /= 8
    total_entropy = tokens_entropy + non_tokens_entropy

    total_token_size = sum(len(t) for t in token_counts.keys())
    grand_total = total_token_size + total_entropy

    print(
        f"Tokens entropy = {tokens_entropy:.1f} bytes, non-tokens = {non_tokens_entropy:.1f}, total = {total_entropy:.1f}"
    )
    # print(f"Tokens size = {total_token_size}, grand total = {grand_total}")


def tokenize_and_count(data: bytes, tokens: list[str], minimize_non_tokens):
    used_tokens = [0] * len(tokens)
    used_non_tokens = [0] * 256
    tokens_in_text = 0
    non_tokens_in_text = 0

    tokenizer = Tokenizer(tokens, minimize_non_tokens)

    for token_id in tokenizer.tokenize(data):
        assert -256 <= token_id < len(tokens)
        if token_id >= 0:
            tokens_in_text += 1
            used_tokens[token_id] += 1
        else:
            non_tokens_in_text += 1
            used_non_tokens[256 + token_id] += 1
    
    token_counts = {}
    for token_id, count in enumerate(used_tokens):
        if count != 0:
            token_counts[tokens[token_id]] = count
    
    non_tokens = {}
    for non_token_id, count in enumerate(used_non_tokens):
        if count != 0:
            non_tokens[bytes([non_token_id])] = count
    
    return token_counts, non_tokens, tokens_in_text, non_tokens_in_text


def sort_by_count(tokens: dict[bytes, int]) -> list[tuple[bytes, int]]:
    pairs = list(tokens.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    return pairs


def sort_by_count1(tokens: dict[bytes, int]) -> list[tuple[bytes, int]]:
    pairs = list(tokens.items())
    pairs.sort(key=lambda t: 10 * t[1] if len(t[0]) == 1 else t[1], reverse=True)
    return pairs


def sort_by_count_chars_first(tokens: dict[bytes, int]) -> list[tuple[bytes, int]]:
    pairs = list(tokens.items())
    pairs.sort(key=lambda t: (len(t[0]) == 1, t[1]), reverse=True)
    return pairs


checkpoints = []
for n in range(2, 20):
    checkpoints.append(2**n - 1)
    checkpoints.append(round(2**n * math.sqrt(2)) - 1)


def iteratively_prune(data, tokens, n_final_tokens, minimize_non_tokens=True):
    while True:
        (
            token_counts,
            used_non_tokens,
            tokens_in_text,
            non_tokens_in_text,
        ) = tokenize_and_count(
            data, tokens, minimize_non_tokens=minimize_non_tokens
        )
        ntokens = len(token_counts)
        n_non_tokens = len(used_non_tokens)
        print(
            f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
            + f"Encoding {tokens_in_text} tokens "
            + f"+ {non_tokens_in_text} non-tokens"
        )
        # calculate_tokens_entropy(len(data), token_counts, non_tokens_in_text)
        # bits_per_item = math.log2(len(token_counts) + len(used_non_tokens))
        # total_size = bits_per_item * (tokens_in_text + non_tokens_in_text) / 8
        # print(f"Entropy: {total_size} bytes")
        if ntokens <= n_final_tokens:
            break
        new_ntokens = max(x for x in checkpoints if x < ntokens)
        new_ntokens = max(n_final_tokens, new_ntokens)
        for non_token, count in used_non_tokens.items():
            token_counts[non_token] = count
        pairs = sort_by_count1(token_counts)
        tokens = [token for token, _ in pairs[:new_ntokens]]
    return token_counts


def maybe_str(s: bytes):
    try:
        return s.decode("utf-8")
    except UnicodeDecodeError:
        return s


def find_frequent_bytes(data: bytes, n: int) -> list[bytes]:
    counts = [0] * 256
    for b in data:
        counts[b[0]] += 1
    d = {}
    for b, c in enumerate(counts):
        if c != 0:
            d[bytes([b])] = c
    d = _sort_and_prune(d, n)
    return d    


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: tokens <initial number of substrings> <token number> <file>"
        )
        exit(1)
    file = os.open(sys.argv[3], os.O_RDONLY)
    data = mmap.mmap(file, 0, access=mmap.ACCESS_READ)
    substrings = find_frequent_substrings(data, int(sys.argv[1]))
    # substrings = find_frequent_bytes(data, int(sys.argv[1]))

    pairs = list(substrings.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    tokens = [s for s, _ in pairs]

    token_counts = iteratively_prune(
        data, tokens, int(sys.argv[2]), minimize_non_tokens=True
    )
    pairs = sort_by_count(token_counts)

    for token, count in pairs:
        print(repr(maybe_str(token)), " ", count)
