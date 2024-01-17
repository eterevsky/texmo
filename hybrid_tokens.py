import argparse
from bisect import bisect_left
from collections import Counter


class HybridTokenSet(object):
    def __init__(self, ninitial: int, nsub: int, nwords: int):
        self.ninitial = ninitial
        self.nsub = nsub
        self.nwords = nwords
        self.initial = []
        self.sub = []
        self.words = []


def find_top_char(lo: int, hi: int, counts: list[(int, int)]) -> int:
    top_c = None
    top_count = 0
    for c, count in counts:
        if count > top_count:
            top_c = c
            top_count = count
    return top_c


def split_interval(
    lo: int, hi: int, counts: list[(int, int)], nparts: int
) -> list[tuple[int, int, int]]:
    if nparts == 1 or len(counts) == 1:
        c = find_top_char(lo, hi, counts)
        return [(lo, c, hi)]

    if len(counts) <= nparts:
        parts = [(lo, counts[0][0], counts[1][0])]
        for i in range(1, len(counts) - 1):
            c = counts[i][0]
            parts.append((c, c, counts[i + 1][0]))
        parts.append((counts[-1][0], counts[-1][0], hi))
        return parts

    total = sum([count for _, count in counts])
    parts = []

    bound = lo
    idx = 0
    cumulative = 0

    for ipart in range(nparts):
        lo_idx = idx
        target = cumulative + (total - cumulative) / (nparts - ipart)
        while cumulative < target and len(counts) - idx >= nparts - ipart:
            cumulative += counts[idx][1]
            idx += 1

        # Select between idx-1 and idx

        if (
            idx == lo_idx + 1
            or cumulative - target < target - cumulative + counts[idx - 1][1]
        ):
            hi_idx = idx
        else:
            hi_idx = idx - 1

        if hi_idx < len(counts):
            hi = counts[hi_idx][0]
        else:
            hi = counts[-1][0] + 1
        c = find_top_char(bound, hi, counts[lo_idx:hi_idx])
        parts.append((bound, c, hi))

        bound = hi

    return parts


def populate_interval(
    lo: int,
    hi: int,
    counts: list[(int, int)],
    prefix: list[int],
    ntokens: int,
    offset: int,
    chars: list[tuple[str, list[int]]],
    intervals: list[tuple[int, int, list[int]]],
):
    assert lo < hi
    parts = split_interval(lo, hi, counts, ntokens)

    for i in range(len(parts)):
        token = i + offset
        sub_lo, top_c, sub_hi = parts[i]
        chars[top_c] = prefix + [token]
        lo_idx = bisect_left(counts, (sub_lo, 0))
        hi_idx = bisect_left(counts, (sub_hi, 0))

        if hi_idx == lo_idx:
            print(lo, hi)
            print(counts)
            print(parts)
            print(i)
            print(sub_lo, sub_hi)
            print(lo_idx, hi_idx)
        assert hi_idx > lo_idx, (lo, hi, counts)

        if hi_idx == lo_idx + 1:
            if sub_lo + 1 < sub_hi:
                intervals.append((sub_lo, sub_hi, prefix + [token]))
        else:
            sub_counts = list((c, count) for c, count in counts[lo_idx:hi_idx] if c != top_c)
            populate_interval(
                sub_lo,
                sub_hi,
                sub_counts,
                prefix + [token],
                ntokens,
                offset,
                chars,
                intervals,
            )


def optimize_tokens(ninitial: int, nsub: int, counts_dict: Counter) -> HybridTokenSet:
    # total = sum(counts_dict.values())
    chars = {}  # char -> sequence of tokens
    intervals = []

    counts = []
    for c in sorted(counts_dict.keys()):
        counts.append((ord(c), counts_dict[c]))

    top_parts = split_interval(0, counts[-1][0] + 1, counts, ninitial)

    for initial_token in range(len(top_parts)):
        lo, top_c, hi = top_parts[initial_token]
        chars[top_c] = [initial_token]
        lo_idx = bisect_left(counts, (lo, 0))
        hi_idx = bisect_left(counts, (hi, 0))

        assert hi_idx > lo_idx

        if hi_idx == lo_idx + 1:
            intervals.append((lo, hi, [initial_token]))
        else:
            sub_counts = list((c, count) for c, count in counts[lo_idx:hi_idx] if c != top_c)
            populate_interval(
                lo,
                hi,
                sub_counts,
                [initial_token],
                nsub,
                ninitial,
                chars,
                intervals,
            )

    filtered_intervals = []

    for lo, hi, seq in intervals:
        if all(c in chars for c in range(lo, hi)):
            continue
        filtered_intervals.append((lo, hi, seq))

    return chars, filtered_intervals

    # for lo, c, hi in top_parts:
    #     print(f"{repr(chr(lo))} {repr(chr(c))} {repr(chr(hi))}")
    # print(top_parts)

    # # list of tokens -> (lo, hi) bounds
    # bounds = {}
    # # list of tokens -> character
    # chars = {}


def main(data, ntokens):
    counts = Counter()
    line_no = 0
    with open(data, "r", encoding="utf-8") as f:
        try:
            for line in f:
                line_no += 1
                for c in line:
                    counts[c] += 1
        except UnicodeDecodeError:
            print(f"UnicodeDecodeError on line {line_no}")
            print(line)
            exit(1)

    best_total = None
    best_chars, best_intervals = None, None

    for ninitial in range(1, ntokens - 1):
        chars, intervals = optimize_tokens(ninitial, ntokens - ninitial, counts)
        total = 0
        for c, count in counts.items():
            total += count * len(chars[ord(c)])

        if best_total is None or total < best_total:
            best_total = total
            best_chars = chars
            best_intervals = intervals
    
    for c in sorted(chars.keys()):
        seq = chars[c]
        print(f"{repr(chr(c))} {seq}")
    print()
    for lo, hi, seq in intervals:
        print(f"{repr(chr(lo))}-{repr(chr(hi))} {seq}")
    print()

    print("Total number of tokens:", best_total)

    # 25797068
    # 26837417

    # (1, 7): 23108344
    # (2, 6): 20764596
    # (3, 5): 18887539
    # (4, 4): 17954717
    # (5, 3): 17636614
    # (6, 2): 17475438

    # (



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-n", "--ntokens", type=int, default=16)

    args = parser.parse_args()

    main(args.data, args.ntokens)
