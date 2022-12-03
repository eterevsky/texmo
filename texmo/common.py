from collections.abc import Iterable
import math


INF = float("inf")
NCHAR = 256


def is_power2(x):
    if x <= 0: return False
    if isinstance(x, int):
        return x & (x - 1) == 0
    log = math.log2(x)
    return abs(log - round(log)) < 1E-10


def is_power2_int(x):
    return isinstance(x, int) and x >= 1 and x & (x - 1) == 0


def total_size(shape):
    prod = 1
    for dim in shape:
        prod *= dim
    return prod


def power2_neighbors(x: int) -> Iterable[int]:
    """Produce neighbor integer powers of 2."""
    if x == 1:
        return (2,)
    else:
        return (x // 2, x * 2)
