# Default values for a number of common command-line arguments.

import os

DATA = "data/data.txt"
TOKENS_DIR = "tokens"

DB = "results/db.sqlite"

# The name of the machine that will by default used in the DB to identify
# runs on the current system.
SYSTEM_NAME = "system"

TRAIN_TIMING = "results/train-timing.jsonl"
SAMPLE_TIMING = "results/sample-timing.jsonl"

SYNC_TOKENS = True