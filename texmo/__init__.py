"""The texmo package.

Importing it installs the XLA_FLAGS workarounds (see `xla_flags.py`).
That has to happen before the JAX backend initializes, and every entry
point -- `texmo.py`, `scripts/`, the search client, pytest -- imports
something from this package first, so the package `__init__` is the one
place that covers all of them.
"""

from . import xla_flags

xla_flags.apply()
