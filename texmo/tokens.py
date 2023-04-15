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


def tokenize(data: bytes, tokens: list[bytes], minimize_non_tokens=True):
    max_token_len = max(len(t) for t in tokens)
    print("max_token_len:", max_token_len)
    tokens = set(tokens)

    # (optimal jump back, total non-token chars, total tokens, best jump forward from lo)
    state = deque([[None, 0, 0, None]])
    lo = 0
    steps_with_same_lo_jump = 0
    for hi in range(1, len(data) + 1):
        best_jump = 1
        if data[hi - 1 : hi] in tokens:
            best_non_tokens = state[-1][1]
            best_tokens = state[-1][2] + 1
        else:
            best_non_tokens = state[-1][1] + 1
            best_tokens = state[-1][2]
        best_lo_forward_jump = 1 if len(state) == 1 else state[-1][3]

        for jump in range(2, min(max_token_len, len(state)) + 1):
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
                best_lo_forward_jump = (
                    jump if jump == len(state) else state[-jump][3]
                )

        if best_lo_forward_jump == state[-1][3]:
            steps_with_same_lo_jump += 1
        else:
            steps_with_same_lo_jump = 1

        assert best_jump <= max_token_len
        state.append(
            [best_jump, best_non_tokens, best_tokens, best_lo_forward_jump]
        )

        if steps_with_same_lo_jump >= max_token_len:
            yield data[lo : lo + best_lo_forward_jump]
            lo += best_lo_forward_jump
            for _ in range(best_lo_forward_jump):
                state.popleft()
            state[0][3] = None
            for pos in range(1, len(state)):
                jump = state[pos][0]
                if jump > pos:
                    state[pos][3] = None
                elif jump == pos:
                    state[pos][3] = jump
                else:
                    state[pos][3] = state[pos - jump][3]
            steps_with_same_lo_jump = 0
            for pos_state in reversed(state):
                if pos_state[3] == state[-1][3]:
                    steps_with_same_lo_jump += 1
                else:
                    break

    jumps = []
    pos = len(state) - 1
    while pos > 0:
        jump = state[pos][0]
        pos -= jump
        jumps.append(jump)

        assert pos >= 0

        if data[lo + pos : lo + pos + jump] in tokens:
            assert state[pos][1] == state[pos + jump][1]
            assert state[pos][2] + 1 == state[pos + jump][2]
        else:
            assert state[pos][1] + 1 == state[pos + jump][1]
            assert state[pos][2] == state[pos + jump][2]

    for jump in reversed(jumps):
        yield data[lo : lo + jump]
        lo += jump


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
    print(f"Tokens size = {total_token_size}, grand total = {grand_total}")


def tokenize_and_count(data: bytes, tokens: list[str], minimize_non_tokens):
    tokens = set(tokens)
    used_tokens = {}
    used_non_tokens = set()
    tokens_in_text = 0
    non_tokens_in_text = 0

    for token in tokenize(
        data, tokens, minimize_non_tokens=minimize_non_tokens
    ):
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
        calculate_tokens_entropy(len(data), token_counts, non_tokens_in_text)
        # bits_per_item = math.log2(len(token_counts) + len(used_non_tokens))
        # total_size = bits_per_item * (tokens_in_text + non_tokens_in_text) / 8
        # print(f"Entropy: {total_size} bytes")
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
        print(
            "Usage: tokens <initial number of substrings> <token number> <file>"
        )
        exit(1)
    file = os.open(sys.argv[3], os.O_RDONLY)
    data = mmap.mmap(file, 0, access=mmap.ACCESS_READ)
    substrings = find_frequent_substrings(data, int(sys.argv[1]))

    pairs = list(substrings.items())
    pairs.sort(key=itemgetter(1), reverse=True)
    tokens = [s for s, _ in pairs]

    token_counts = iteratively_prune(
        data, tokens, int(sys.argv[2]), minimize_non_tokens=True
    )
    pairs = sort_by_count(token_counts)

    for token, count in pairs:
        print(repr(maybe_str(token)), " ", count)
