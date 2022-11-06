import math


INF = float("inf")
NCHAR = 256


def is_power2(x):
    if x <= 0: return False
    if type(x) is int:
        return x & (x - 1) == 0
    log = math.log2(x)
    return abs(log - round(log)) < 1E-10
