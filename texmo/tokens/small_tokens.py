from dataclasses import dataclass
import json
from typing import Self

from ..common import INF


class Token(object):
    def __init__(self, bits: int, value: int):
        self.bits: int = bits
        self.value: int = value

    def __repr__(self):
        return f"Token({self.bits}, {self.value})"

    def __add__(self, other: Self):
        assert self.bits == other.bits
        return Token(2 * self.bits, self.value * 2**self.bits + other.value)

    def __eq__(self, other: Self):
        return self.bits == other.bits and self.value == other.value


def tokenize(token_set: list[Token], bits: int, value: int):
    assert bits >= 1

    if Token(bits, value) in token_set:
        # return [Token(bits, value)]
        return 1

    if bits == 1:
        return None
        # print(bits, value)
        # print(token_set)
        # assert Token(1, 0) in token_set

    new_bits = bits // 2
    mask = 1 << new_bits

    first = tokenize(token_set, new_bits, value // mask)
    second = tokenize(token_set, new_bits, value % mask)

    if first is None or second is None:
        return None

    return first + second

def count_tokens(counts, tokens):
    total = 0
    for b in range(256):
        count = tokenize(tokens, 8, b)
        if count is None:
            return None
        total += counts[b] * count
    return total

def optimize_bits(counts, ntokens):
    # tokens = [
    #     Token(1, 0),
    #     Token(1, 1),
    # ]
    tokens = [
        Token(2, 0),
        Token(2, 1),
        Token(2, 2),
        Token(2, 3),
    ]
    # tokens = [Token(4, x) for x in range(16)]


    while True:
        total_tokens = count_tokens(counts, tokens)
        print(tokens, total_tokens)
        best_total = total_tokens
        best_add = None
        best_remove = None

        for bits in (2, 4, 8):
            for value in range(2**bits):
                token = Token(bits, value)
                if token in tokens:
                    continue
                new_tokens = tokens + [token]
                total = count_tokens(counts, new_tokens)

                if total >= best_total:
                    continue

                if len(new_tokens) <= ntokens:
                    best_total = total
                    best_add = token
                else:
                    for remove in tokens:
                        new_tokens.remove(remove)
                        total = count_tokens(counts, new_tokens)
                        if total is not None and total < best_total:
                            best_total = total
                            best_add = token
                            best_remove = remove
                        new_tokens.append(remove)

        if best_add is None:
            break

        tokens.append(best_add)
        if best_remove is not None:
            tokens.remove(best_remove)

    return tokens


def main(args):
    ntokens = args.ntokens

    if args.load_counts:
        counts = list(map(int, open(args.load_counts).read().strip().split()))
    else:
        path = args.data
        counts = [0] * 256
        total = 0
        with open(path, "rb") as data:
            while True:
                chunk = data.read(2**20)
                if not chunk:
                    break
                for b in chunk:
                    counts[b] += 1
                    total += 1

    # print(total)
    # print()

    # for i in sorted(range(256), reverse=True, key=lambda x: counts[x]):
    #     print(i, counts[i])
    # print()

    # counts16 = [0] * 16
    # for b in range(256):
    #     counts16[b // 16] += counts[b]
    #     counts16[b % 16] += counts[b]

    # for i in sorted(range(16), reverse=True, key=lambda x: counts16[x]):
    #     print(i, counts16[i])
    # print()

    # counts4 = [0] * 4
    # for b in range(16):
    #     counts4[b // 4] += counts16[b]
    #     counts4[b % 4] += counts16[b]

    # for i in sorted(range(4), reverse=True, key=lambda x: counts4[x]):
    #     print(i, counts4[i])
    # print()

    tokens = optimize_bits(counts, ntokens)
    total = count_tokens(counts, tokens)

    tokens.sort(key=lambda t: 256 * t.bits + t.value)
    out = {
        "type": "bits",
        "tokens": []
    }
    for t in tokens:
        out["tokens"].append({"bits": t.bits, "value": t.value})
    if args.initial_size:
        initial_size = args.initial_size
    else:
        initial_size = sum(counts)
    out["stats"] = {
        "ntokens": ntokens,
        "initial_size": initial_size,
        "processed_size": sum(counts),
        "total_tokens": total,
        "bytes_per_token": initial_size / total
    }

    print(json.dumps(out, indent=2))


def init_args(parser):
    parser.add_argument(
        "-d", "--data", type=str, help="training data file"
    )
    parser.add_argument(
        "-n", "--ntokens", type=int, help="number of bit tokens"
    )
    parser.add_argument(
        "-l", "--load-counts", type=str, help="file with pre-computed byte counts"
    )
    parser.add_argument("--initial-size", type=int)
    parser.set_defaults(func=main)
