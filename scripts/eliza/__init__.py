"""Vendored ELIZA with the DOCTOR script -- see README.md.

`load_doctor()` is the whole convenience: the script file ships next
to the code, so no caller has to know where it lives.
"""
import os
import random

from .eliza import Eliza

DOCTOR_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "doctor.txt")


def load_doctor(rng=random) -> Eliza:
    """A fresh DOCTOR session drawing from `rng`.

    Pass a `random.Random(seed)` for a reproducible session: the
    matcher itself is deterministic, and the only draws are the
    greeting, the sign-off and which remark comes back out of memory.
    """
    bot = Eliza(rng)
    bot.load(DOCTOR_SCRIPT)
    return bot
