# Default values for a number of common command-line arguments.

import os

DIR = os.path.dirname(__file__)

DB = os.path.join(DIR, "results/db.sqlite")
DATA = os.path.join(DIR, "data.txt")
LOG = os.path.join(DIR, "log.csv")
CHECKPOINTS = os.path.join(DIR, "checkpoints")

DEFAULT_BATCH = 32
DEFAULT_LR = 0.0625
