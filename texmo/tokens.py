from collections import deque
from collections.abc import Iterable
import math
import mmap
from operator import itemgetter
import os
import sys
from typing import Self

import numpy as np

INF = float("inf")


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
                new_non_tokens == old_non_tokens and new_tokens < old_tokens
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


class Token(object):
    def __init__(self, id: int, string: bytes, literal: bool, cost: float):
        self.id = id
        self.len = len(string)
        self.string = string
        self.literal = literal
        self.cost = cost
        # In case any suffix of a current token are also tokens, this is
        # the longest one of them. This way all tokens that are suffixes of
        # this one are stored in a linked list.
        self.suffix_token = None

    def __repr__(self) -> str:
        return f"Token({self.id}, {self.string})"


class TokenizerState(object):
    def __init__(self, suffix: bytes, token: Token = None):
        self.suffix: bytes = suffix
        self.last_token: Token = token
        # Next byte -> next TokenizerState
        self.next: list[TokenizerState] = [None] * 256


class DynState(object):
    def __init__(
        self,
        tok_state: TokenizerState,
        first_token: Token,
        last_token: Token,
        cost: float,
    ):
        self.tok_state = tok_state
        self.first_token = first_token
        self.last_token = last_token
        self.cost = cost

    def __repr__(self):
        first = self.first_token.string if self.first_token else "?"
        last = self.last_token.string if self.last_token else "?"
        return f"DynamicState(first={first}, last={last}, cost={self.cost}"


class Tokenizer2(object):
    """New tokenizer implementation with a finite automaton."""

    def __init__(self, tokens: list[bytes], non_token_penalty: float = 10):
        self._non_token_penalty = non_token_penalty

        self._tokens: dict[bytes, Token] = {}
        self._init_tokens(tokens)

        self._max_token_len = max(token.len for token in self._tokens.values())

        self._states: dict[bytes, TokenizerState] = {}
        self._init_states()

    def _init_tokens(self, tokens: list[bytes]):
        for token_str in tokens:
            id = len(self._tokens)
            token = Token(id=id, string=token_str, literal=False, cost=1)
            self._tokens[token_str] = token

        for b in range(256):
            token_str = bytes([b])
            id = b - 256
            if token_str not in self._tokens:
                token = Token(
                    id=id,
                    string=token_str,
                    literal=True,
                    cost=self._non_token_penalty,
                )
                self._tokens[token_str] = token

        for token_str, token in self._tokens.items():
            assert token.string == token_str

        for token in self._tokens.values():
            for start in range(1, len(token.string)):
                token_suffix = token.string[start:]
                if token_suffix in self._tokens:
                    token.suffix_token = self._tokens[token_suffix]
                    break

    def _init_states(self):
        self._states[b""] = TokenizerState(b"", token=None)
        for token in self._tokens.values():
            for prefix_len in range(1, len(token.string) + 1):
                prefix = token.string[:prefix_len]
                if prefix not in self._states:
                    prefix_token = None
                    for token_start in range(len(prefix)):
                        prefix_token = self._tokens.get(prefix[token_start:])
                        if prefix_token is not None:
                            break
                    # Since we should have all literals among tokens
                    assert prefix_token is not None
                    self._states[prefix] = TokenizerState(prefix, prefix_token)

        for state in self._states.values():
            for b in range(256):
                new_suffix = state.suffix + bytes([b])
                for next_start in range(len(new_suffix)):
                    next_state = self._states.get(new_suffix[next_start:])
                    if next_state is not None:
                        state.next[b] = next_state
                        break

    def _consume_first_token(self, state: deque[DynState]):
        first_token = state[-1].first_token

        for _ in range(first_token.len):
            state.popleft()

        for i, dyn_state in enumerate(state):
            last_token = dyn_state.last_token
            last_token_len = last_token.len
            if last_token_len > i:
                dyn_state.first_token = None
            elif last_token_len == i:
                dyn_state.first_token = dyn_state.last_token
            else:
                dyn_state.first_token = state[i - last_token_len].first_token

        return first_token

    def tokenize(self, data: Iterable[int]) -> Iterable[Token]:
        tok_state: TokenizerState = self._states[b""]
        state: deque[DynState] = deque(
            [DynState(tok_state, last_token=None, first_token=None, cost=0)]
        )
        steps_with_same_first_token = 0

        for b in data:
            if type(b) is bytes:
                b = b[0]
            tok_state = tok_state.next[b]

            dyn_state = DynState(
                tok_state, last_token=None, first_token=None, cost=INF
            )

            last_token = tok_state.last_token
            while last_token is not None:
                if last_token.len <= len(state):
                    prev_state = state[-last_token.len]

                    cost = prev_state.cost + last_token.cost
                    if cost < dyn_state.cost:
                        dyn_state.last_token = last_token
                        dyn_state.cost = cost
                        if prev_state.first_token is None:
                            dyn_state.first_token = last_token
                        else:
                            dyn_state.first_token = prev_state.first_token
                last_token = last_token.suffix_token

            assert dyn_state.first_token is not None

            if dyn_state.first_token == state[-1].first_token:
                steps_with_same_first_token += 1
            else:
                steps_with_same_first_token = 1

            state.append(dyn_state)

            if steps_with_same_first_token >= self._max_token_len:
                yield self._consume_first_token(state)
                first_token = state[-1].first_token
                steps_with_same_first_token = 0
                for dyn_state in reversed(state):
                    if dyn_state.first_token == first_token:
                        steps_with_same_first_token += 1
                    else:
                        break

        while len(state) > 1:
            yield self._consume_first_token(state)


def tokenize(
    data: bytes, tokens: list[bytes], minimize_non_tokens=True
) -> Iterable[bytes]:
    # tokenizer = Tokenizer(tokens, minimize_non_tokens)
    # for token_id in tokenizer.tokenize(data):
    #     if token_id >= 0:
    #         yield tokens[token_id]
    #     else:
    #         yield bytes([token_id + 256])
    non_token_cost = 10 if minimize_non_tokens else 1
    tokenizer = Tokenizer2(tokens, non_token_cost)
    for token in tokenizer.tokenize(data):
        yield token.string


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


def calculate_entropy(
    total_size: int,
    token_counts: dict[bytes, int],
    non_token_counts: dict[bytes, int],
    non_token_mult: float = 10.0,
) -> float:
    total = 0
    for count in token_counts.values():
        p = -math.log2(
            count / total_size
        )  # Yes, I know that I can just take count and subtract log(total_size)
        total += count * p

    total_non_tokens = sum(non_token_counts.values())

    if total_non_tokens > 0:
        p = -math.log2(total_non_tokens / total_size)
        total += total_non_tokens * p

        for count in non_token_counts.values():
            p = -math.log2(count / total_non_tokens)
            total += non_token_mult * count * p

    return total / total_size


def tokenize_and_count(
    data: bytes, tokens: list[str], minimize_non_tokens
) -> tuple[dict[bytes, int], dict[bytes, int], int, int]:
    used_tokens = [0] * len(tokens)
    used_non_tokens = [0] * 256
    tokens_in_text = 0
    non_tokens_in_text = 0

    non_token_cost = 10 if minimize_non_tokens else 1
    tokenizer2 = Tokenizer2(tokens, non_token_cost)

    for token in tokenizer2.tokenize(data):
        token_id = token.id
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
    pairs.sort(
        key=lambda t: 10 * t[1] if len(t[0]) == 1 else t[1], reverse=True
    )
    return pairs


def sort_by_count_chars_first(
    tokens: dict[bytes, int]
) -> list[tuple[bytes, int]]:
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
            non_token_counts,
            tokens_in_text,
            non_tokens_in_text,
        ) = tokenize_and_count(
            data, tokens, minimize_non_tokens=minimize_non_tokens
        )
        ntokens = len(token_counts)
        n_non_tokens = len(non_token_counts)
        entropy = calculate_entropy(len(data), token_counts, non_token_counts)
        print(
            f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
            + f"Encoding {tokens_in_text} tokens "
            + f"+ {non_tokens_in_text} non-tokens."
            + f"Entropy per byte: {entropy}"
        )
        # calculate_tokens_entropy(len(data), token_counts, non_tokens_in_text)
        # bits_per_item = math.log2(len(token_counts) + len(used_non_tokens))
        # total_size = bits_per_item * (tokens_in_text + non_tokens_in_text) / 8
        # print(f"Entropy: {total_size} bytes")
        if ntokens <= n_final_tokens:
            break
        new_ntokens = max(x for x in checkpoints if x < ntokens)
        new_ntokens = max(n_final_tokens, new_ntokens)
        for non_token, count in non_token_counts.items():
            token_counts[non_token] = count
        pairs = sort_by_count1(token_counts)
        tokens = [token for token, _ in pairs[:new_ntokens]]
    return token_counts


def tokenize_and_prune(data, tokens, n_final_tokens):
    (
        token_counts,
        non_token_counts,
        tokens_in_text,
        non_tokens_in_text,
    ) = tokenize_and_count(data, tokens, minimize_non_tokens=True)

    ntokens = len(token_counts)
    n_non_tokens = len(non_token_counts)

    print(
        f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
        + f"Encoding {tokens_in_text} tokens "
        + f"+ {non_tokens_in_text} non-tokens."
    )

    pairs = sort_by_count1(token_counts)
    tokens = [token for token, _ in pairs[:n_final_tokens]]

    (
        token_counts,
        non_token_counts,
        tokens_in_text,
        non_tokens_in_text,
    ) = tokenize_and_count(data, tokens, minimize_non_tokens=True)

    ntokens = len(token_counts)
    n_non_tokens = len(non_token_counts)
    entropy = calculate_entropy(len(data), token_counts, non_token_counts)
    full_entropy = entropy * len(data)
    print(
        f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
        + f"Encoding {tokens_in_text} tokens "
        + f"+ {non_tokens_in_text} non-tokens."
    )
    print(f"Entropy per byte: {entropy}, full: {full_entropy}")

    return token_counts


def tokenize_and_prune1(data, tokens, n_final_tokens):
    (token_counts, _, _, _) = tokenize_and_count(data, tokens, minimize_non_tokens=True)
    tokens = list(token_counts.keys())

    while len(tokens) > n_final_tokens:
        min_entropy = INF
        useless_token = None
        best_pairs = None
        for t in tokens:
            fewer_tokens = list(tokens)
            fewer_tokens.remove(t)

            (
                token_counts,
                non_token_counts,
                tokens_in_text,
                non_tokens_in_text,
            ) = tokenize_and_count(data, fewer_tokens, minimize_non_tokens=True)
            entropy = calculate_entropy(len(data), token_counts, non_token_counts)
            ntokens = len(token_counts)
            print(f"{ntokens} (removed {t}): {tokens_in_text} tokens + {non_tokens_in_text} non-tokens, entropy: {entropy}")

            if entropy < min_entropy:
                min_entropy = entropy
                useless_token = t
                best_pairs = sort_by_count(token_counts)
        
        print(f"Removing {useless_token}")
        tokens.remove(useless_token)

        for token, count in best_pairs[:200]:
            print(repr(maybe_str(token)), " ", count)

    (
        token_counts,
        non_token_counts,
        tokens_in_text,
        non_tokens_in_text,
    ) = tokenize_and_count(data, tokens, minimize_non_tokens=True)

    ntokens = len(token_counts)
    n_non_tokens = len(non_token_counts)
    entropy = calculate_entropy(len(data), token_counts, non_token_counts)
    full_entropy = entropy * len(data)
    print(
        f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
        + f"Encoding {tokens_in_text} tokens "
        + f"+ {non_tokens_in_text} non-tokens."
    )
    print(f"Entropy per byte: {entropy}, full: {full_entropy}")

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

    # token_counts = iteratively_prune(
    #     data, tokens, int(sys.argv[2]), minimize_non_tokens=True
    # )
    token_counts = tokenize_and_prune1(data, tokens, int(sys.argv[2]))
    pairs = sort_by_count(token_counts)

    for token, count in pairs[:200]:
        print(repr(maybe_str(token)), " ", count)
