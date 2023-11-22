from collections.abc import Iterable
import logging
import math


logging.basicConfig(
        format='%(levelname)s [%(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')


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

_BASES = (
    (1, ""),
    (1000, "k"),
    (1E6, "M"),
    (1E9, "G"),
    (1E12, "T"),
)

def itoa3(x: int) -> str:
    """Convert integer to a string with 3 significant digits."""
    if x < 2000: return str(x)
    for i in range(1, len(_BASES) - 1):
        if x >= _BASES[i][0] and x < _BASES[i + 1][0]:
            break
    
    base, suffix = _BASES[i]
    prev_base, prev_suffix = _BASES[i - 1]

    significand = x / base

    if significand < 1.995:
        return f"{x / prev_base:.0f}{prev_suffix}"
    if significand < 19.95:
        return f"{x / base:.2f}{suffix}"
    elif significand < 199.5:
        return f"{x / base:.1f}{suffix}"
    else:
        return f"{x / base:.0f}{suffix}"


def ttoa3(t: float) -> str:
    """Convert time in seconds to a string with 3 significant digits."""
    if t < 1E-6: return f"{t * 1E9:.0f} ns"
    elif t < 1E-5: return f"{t * 1E6:.2f} μs"
    elif t < 1E-4: return f"{t * 1E6:.1f} μs"
    elif t < 0.001: return f"{t * 1E6:.0f} μs"
    elif t < 0.01: return f"{t * 1E3:.2f} ms"
    elif t < 0.1: return f"{t * 1E3:.1f} ms"
    elif t < 1: return f"{t * 1E3:.0f} ms"
    elif t < 10: return f"{t:.2f} s"
    elif t < 60: return f"{t:.1f} s"
    elif t < 3600:
        m = round(t // 60)
        s = round(t % 60)
        return f"{m} m {s} s"
    else:
        h = round(t // 3600)
        m = (t % 3600) // 60
        s = round(t % 60)
        return f"{h} h {m} m {s} s"