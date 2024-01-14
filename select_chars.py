"""This is a very dirty script selecting a set of characters for small tokensets.

Before running this, you need to count the total number of various bytes in the
training data. This can be done using `cargo run --release -- count-bytes`.

It's better to generate these tokensets from _processed_ data, i.e. from
`data_capswords.txt`.

Usage:

python select_chars.py counts.txt capswords 1234567890

The last argument is the length of the training data _before_ processing.

Running this will produce tokensets `tokens/tokens<n>_capswords_chars.json` for
n from 2 to 128.
"""


from math import log2
import sys
from texmo import pjson

CHARS = {
    b"\x00": b" ",
    b"\x01": b" ",
    b"\x02": b" ",
    b"\x03": b" ",
    b"\x04": b" ",
    b"\x05": b" ",
    b"\x06": b" ",
    b"\x07": b" ",
    b"\x08": b" ",
    b"\t": b" ",
    b"\n": b" ",
    b"\x0b": b" ",
    b"\x0c": b" ",
    b"\r": b" ",
    b"\x0e": b" ",
    b"\x0f": b" ",
    b"\x10": b" ",
    b"\x11": b" ",
    b"\x12": b" ",
    b"\x13": b" ",
    b"\x14": b" ",
    b"\x15": b"\x14",
    b"\x16": b" ",
    b"\x17": b" ",
    b"\x18": b" ",
    b"\x19": b" ",
    b"\x1a": b" ",
    b"\x1b": b" ",
    b"\x1c": b" ",
    b"\x1d": b" ",
    b"\x1e": b" ",
    b"\x1f": b" ",
    b" ": b" ",
    b"!": b".",
    b'"': b"'",
    b"#": b"?",
    b"$": b"#",
    b"%": b"#",
    b"&": b"#",
    b"'": b"?",
    b"(": b"?",
    b")": b"?",
    b"*": b"?",
    b"+": b"*",
    b",": b"?",
    b"-": b"_",
    b".": b",",
    b"/": b"?",
    b"0": b"?",
    b"1": b"0",
    b"2": b"1",
    b"3": b"1",
    b"4": b"1",
    b"5": b"1",
    b"6": b"1",
    b"7": b"1",
    b"8": b"1",
    b"9": b"1",
    b":": b".",
    b";": b":",
    b"<": b"=",
    b"=": b"-",
    b">": b"=",
    b"?": b" ",
    b"@": b"#",
    b"A": b"a",
    b"B": b"b",
    b"C": b"c",
    b"D": b"d",
    b"E": b"e",
    b"F": b"f",
    b"G": b"g",
    b"H": b"h",
    b"I": b"i",
    b"J": b"j",
    b"K": b"k",
    b"L": b"l",
    b"M": b"m",
    b"N": b"n",
    b"O": b"o",
    b"P": b"p",
    b"Q": b"q",
    b"R": b"r",
    b"S": b"s",
    b"T": b"t",
    b"U": b"u",
    b"V": b"v",
    b"W": b"w",
    b"X": b"x",
    b"Y": b"y",
    b"Z": b"z",
    b"[": b"(",
    b"\\": b"|",
    b"]": b")",
    b"^": b"#",
    b"_": b"?",
    b"`": b"'",
    b"a": b"e",
    b"b": b"p",
    b"c": b"t",
    b"d": b"t",
    b"e": b"?",
    b"f": b"t",
    b"g": b"d",
    b"h": b"t",
    b"i": b"e",
    b"j": b"i",
    b"k": b"h",
    b"l": b"r",
    b"m": b"n",
    b"n": b"t",
    b"o": b"a",
    b"p": b"f",
    b"q": b"k",
    b"r": b"t",
    b"s": b"t",
    b"t": b"?",
    b"u": b"o",
    b"v": b"u",
    b"w": b"u",
    b"x": b"s",
    b"y": b"u",
    b"z": b"x",
    b"{": b"(",
    b"|": b"*",
    b"}": b")",
    b"~": b"-",
    b"\x7f": b" ",
    b"\x80": b"?",
    b"\x81": b"\x80",
    b"\x82": b"\x80",
    b"\x83": b"\x80",
    b"\x84": b"\x80",
    b"\x85": b"\x80",
    b"\x86": b"\x80",
    b"\x87": b"\x80",
    b"\x88": b"\x80",
    b"\x89": b"\x80",
    b"\x8a": b"\x80",
    b"\x8b": b"\x80",
    b"\x8c": b"\x80",
    b"\x8d": b"\x80",
    b"\x8e": b"\x80",
    b"\x8f": b"\x80",
    b"\x90": b"\x94",
    b"\x91": b"\x90",
    b"\x92": b"\x90",
    b"\x93": b"\x94",
    b"\x94": b"\x80",
    b"\x95": b"\x90",
    b"\x96": b"\x90",
    b"\x97": b"\x90",
    b"\x98": b"\x90",
    b"\x99": b"\x90",
    b"\x9a": b"\x90",
    b"\x9b": b"\x90",
    b"\x9c": b"\x90",
    b"\x9d": b"\x90",
    b"\x9e": b"\x90",
    b"\x9f": b"\x94",
    b"\xa0": b"\xa9",
    b"\xa1": b"\xa0",
    b"\xa2": b"\xa0",
    b"\xa3": b"\xa0",
    b"\xa4": b"\xa9",
    b"\xa5": b"\xa0",
    b"\xa6": b"\xa0",
    b"\xa7": b"\xa0",
    b"\xa8": b"\xa0",
    b"\xa9": b"\x80",
    b"\xaa": b"\xa0",
    b"\xab": b"\xa0",
    b"\xac": b"\xa0",
    b"\xad": b"\xa9",
    b"\xae": b"\xa0",
    b"\xaf": b"\xa0",
    b"\xb0": b"\xb3",
    b"\xb1": b"\xb3",
    b"\xb2": b"\xb3",
    b"\xb3": b"\x80",
    b"\xb4": b"\xb3",
    b"\xb5": b"\xb3",
    b"\xb6": b"\xb3",
    b"\xb7": b"\xb3",
    b"\xb8": b"\xb3",
    b"\xb9": b"\xb3",
    b"\xba": b"\xb3",
    b"\xbb": b"\xb3",
    b"\xbc": b"\xb3",
    b"\xbd": b"\xb3",
    b"\xbe": b"\xb3",
    b"\xbf": b"\xb3",
    b"\xc0": b"\xc3",
    b"\xc1": b"\xc3",
    b"\xc2": b"\xc3",
    b"\xc3": b"?",
    b"\xc4": b"\xc3",
    b"\xc5": b"\xc3",
    b"\xc6": b"\xc3",
    b"\xc7": b"\xc3",
    b"\xc8": b"\xc3",
    b"\xc9": b"\xc3",
    b"\xca": b"\xc3",
    b"\xcb": b"\xc3",
    b"\xcc": b"\xc3",
    b"\xcd": b"\xc3",
    b"\xce": b"\xc3",
    b"\xcf": b"\xc3",
    b"\xd0": b"\xc3",
    b"\xd1": b"\xd0",
    b"\xd2": b"\xd0",
    b"\xd3": b"\xd0",
    b"\xd4": b"\xd0",
    b"\xd5": b"\xd0",
    b"\xd6": b"\xd0",
    b"\xd7": b"\xd0",
    b"\xd8": b"\xd0",
    b"\xd9": b"\xd0",
    b"\xda": b"\xd0",
    b"\xdb": b"\xd0",
    b"\xdc": b"\xd0",
    b"\xdd": b"\xd0",
    b"\xde": b"\xd0",
    b"\xdf": b"\xd0",
    b"\xe0": b"\xe2",
    b"\xe1": b"\xe0",
    b"\xe2": b"?",
    b"\xe3": b"\xe0",
    b"\xe4": b"\xe0",
    b"\xe5": b"\xe0",
    b"\xe6": b"\xe0",
    b"\xe7": b"\xe0",
    b"\xe8": b"\xe0",
    b"\xe9": b"\xe0",
    b"\xea": b"\xe0",
    b"\xeb": b"\xe0",
    b"\xec": b"\xe0",
    b"\xed": b"\xe0",
    b"\xee": b"\xe0",
    b"\xef": b"\xe0",
    b"\xf0": b"?",
    b"\xf1": b"\xf0",
    b"\xf2": b"\xf0",
    b"\xf3": b"\xf0",
    b"\xf4": b"\xf0",
    b"\xf5": b"\xf0",
    b"\xf6": b"\xf0",
    b"\xf7": b"\xf0",
    b"\xf8": b"\xff",
    b"\xf9": b"\xff",
    b"\xfa": b"\xff",
    b"\xfb": b"\xff",
    b"\xfc": b"\xff",
    b"\xfd": b"\xff",
    b"\xfe": b"\xff",
    b"\xff": b"?",
}
INF = float("inf")


def ancestor(c, groups):
    while c not in groups:
        c = CHARS[c]
    return c


def total_entropy(counts, groups):
    group_counts = [0] * 256

    for c in range(256):
        g = ancestor(bytes([c]), groups)
        group_counts[g[0]] += counts[c]

    entropy = 0

    for c in range(256):
        c_count = counts[c]
        group_count = group_counts[ancestor(bytes([c]), groups)[0]]
        c_entropy = log2(c_count / group_count)
        # print(bytes([c]), c_count, group_count, c_entropy)
        entropy += c_entropy * c_count

    # print(groups, entropy)

    return entropy


def add_char(counts, groups):
    best_entropy = -INF
    best_c = None
    for i in range(256):
        c = bytes([i])
        if c in groups:
            continue
        new_groups = groups + [c]
        entropy = total_entropy(counts, new_groups)
        if entropy > best_entropy:
            best_c = c
            best_entropy = entropy
    return best_c, best_entropy


def add_char2(counts, groups):
    best_entropy = -INF
    best_c = None, None
    for c1 in range(256):
        c1 = bytes([c1])
        if c1 in groups:
            continue
        for c2 in range(c1[0] + 1, 256):
            c2 = bytes([c2])
            if c2 in groups:
                continue
            new_groups = groups + [c1, c2]
            entropy = total_entropy(counts, new_groups)
            if entropy > best_entropy:
                best_c = (c1, c2)
                best_entropy = entropy
    return best_c


def write_token(c: int | bytes) -> str:
    if type(c) is int:
        c = bytes([c])
    try:
        token = c.decode("utf-8")
    except UnicodeError:
        token = [c[0]]
    return token


def write_tokenset(counts, groups, proc="raw", total_size=None):
    group_counts = [0] * 256

    for c in range(256):
        g = ancestor(bytes([c]), groups)
        group_counts[g[0]] += counts[c]

    n = len(groups)
    filename = f"tokens/tokens{n}_{proc}_chars.json"
    print(filename)
    entropy = -total_entropy(counts, groups)
    tokenset = {
        "tokens": [],
        "groups": [],
        "byte_entropy": [],
        "type": "chars",
        "processing": "capswords",
        "stats": {
            "ntokens": n,
            "total_entropy": entropy,
            "entropy_per_byte": entropy / total_size,
        },
    }

    groups = list(sorted(groups))

    for c in groups:
        tokenset["tokens"].append(write_token(c))

    for i in range(256):
        g = ancestor(bytes([i]), groups)
        tokenset["groups"].append(groups.index(g))

        c_count = counts[i]
        group_count = group_counts[g[0]]
        c_entropy = log2(c_count / group_count)
        tokenset["byte_entropy"].append(-c_entropy)

    with open(filename, "w") as f:
        pjson.save_json(tokenset, f)


for i in range(256):
    j = bytes([i])
    visited = []
    while j not in visited:
        visited.append(j)
        j = CHARS[j]
    print(visited)
    assert j == b" "

with open(sys.argv[1]) as f:
    l = f.read()
    counts = list(map(int, l.split()))

for i in range(256):
    counts[i] += 1

for i in range(256):
    c = bytes([i])
    if counts[i] > counts[CHARS[c][0]]:
        print(c, counts[i], CHARS[c], counts[CHARS[c][0]])

groups = [b" ", b"?"]
# groups = [b" "]

# c, entropy = add_char(counts, groups)
# groups.append(c)


if len(sys.argv) > 3:
    total_size = int(sys.argv[3])
else:
    total_size = sum(counts) - 256


while len(groups) <= 128:
    if len(groups) > 1 and (len(groups) - 1) & len(groups) == 0:
        write_tokenset(counts, groups, sys.argv[2], total_size)
        # print(groups)
    c, entropy = add_char(counts, groups)
    groups.append(c)
    # c1, c2 = add_char2(counts, groups)
    # print(c, entropy)
