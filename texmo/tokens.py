from collections import deque
import math
import mmap
from operator import itemgetter
import os
import sys


def _sort_and_prune(substrings: dict[bytes, int], n: int) -> dict[bytes, int]:
    pairs = list(substrings.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    return dict(pairs[:n])


def find_common_substrings(data: bytes, n: int) -> dict[bytes, int]:
    l = 1
    substrings = {}
    while l <= len(data):
        print(f"Gathering substrings of length {l}")
        for i in range(0, len(data) - l):
            s = data[i : i + l]
            if l == 1 or s[:-1] in substrings:
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


def tokenize(data: bytes, tokens: list[bytes], minimize_non_tokens=True):
    max_token_len = max(len(t) for t in tokens)
    tokens = set(tokens)

    # (optimal jump, total non-token chars, total tokens)
    state = deque([(None, 0, 0)])
    lo = 0
    for hi in range(1, len(data)):
        best_jump = 1
        best_non_tokens = state[-1][1]
        best_tokens = state[-1][2] + 1
        if data[hi - 1] in tokens:
            non_tokens += 1

        for jump in range(2, min(max_token_len, len(state))):
            if data[hi - jump : hi] not in tokens:
                continue
            non_tokens = state[-jump][1]
            ntokens = state[-jump][2] + 1

            if (
                minimize_non_tokens
                and (
                    non_tokens < best_non_tokens
                    or (non_tokens == best_non_tokens and ntokens < best_tokens)
                )
            ) or (
                not minimize_non_tokens
                and non_tokens + ntokens < best_non_tokens + best_tokens
            ):
                best_jump = jump
                best_non_tokens = non_tokens
                best_tokens = ntokens

        state.append((best_jump, best_non_tokens, best_tokens))

        if len(state) > 3 * max_token_len:
            # TODO: implement a precise check
            pos = len(state) - 1
            while pos > 0:
                jump = state[pos][0]
                pos -= jump

            yield data[lo : lo + jump]
            lo += jump
            for _ in range(jump):
                state.popleft()

    jumps = []
    pos = len(state) - 1
    while pos > 0:
        jump = state[pos][0]
        pos -= jump
        jumps.append(jump)

    for jump in jumps:
        yield data[lo : lo + jump]
        lo += jump


def tokenize_and_count(data: bytes, tokens: list[str], minimize_non_tokens):
    tokens = set(tokens)
    used_tokens = {}
    used_non_tokens = set()
    tokens_in_text = 0
    non_tokens_in_text = 0

    for token in tokenize(data, tokens, minimize_non_tokens=minimize_non_tokens):
        if token in tokens:
            tokens_in_text += 1
            try:
                used_tokens[token] += 1
            except KeyError:
                used_tokens[token] = 1
        else:
            if len(token) != 1:
                print(f"Long non-token: {token}")
                assert False
            non_tokens_in_text += 1
            used_non_tokens.add(token)

    return used_tokens, used_non_tokens, tokens_in_text, non_tokens_in_text


def sort_by_count(tokens: dict[bytes, int]) -> list[tuple[bytes, int]]:
    pairs = list(tokens.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    return pairs


def iteratively_prune(data, tokens, n_final_tokens, minimize_non_tokens=True):
    while True:
        (
            token_counts,
            used_non_tokens,
            tokens_in_text,
            non_tokens_in_text,
        ) = tokenize_and_count(data, tokens, minimize_non_tokens=minimize_non_tokens)
        ntokens = len(token_counts)
        n_non_tokens = len(used_non_tokens)
        print(
            f"Using {ntokens} tokens and {n_non_tokens} non-tokens. "
            + f"Encoding {tokens_in_text} tokens "
            + f"+ {non_tokens_in_text} non-tokens"
        )
        bits_per_item = math.log2(len(token_counts) + len(used_non_tokens))
        total_size = bits_per_item * (tokens_in_text + non_tokens_in_text) / 8
        print(f"Entropy: {total_size} bytes")
        if ntokens <= n_final_tokens:
            break
        new_ntokens = max(n_final_tokens, 3 * ntokens // 4)
        pairs = sort_by_count(token_counts)
        tokens = [token for token, _ in pairs[:new_ntokens]]
    return token_counts


def maybe_str(s: bytes):
    try:
        return s.decode("utf-8")
    except UnicodeDecodeError:
        return s


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: tokens <initial number of substrings> <token number> <file>")
        exit(1)
    file = os.open(sys.argv[3], os.O_RDONLY)
    data = mmap.mmap(file, 0, access=mmap.ACCESS_READ)
    substrings = find_common_substrings(data, int(sys.argv[1]))

    pairs = list(substrings.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    tokens = [s for s, _ in pairs]

    token_counts = iteratively_prune(data, tokens, int(sys.argv[2]), minimize_non_tokens=True)
    pairs = sort_by_count(token_counts)

    for token, count in pairs:
        print(repr(maybe_str(token)), ' ', count)
