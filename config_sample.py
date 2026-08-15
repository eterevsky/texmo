# Default values for a number of common command-line arguments.

import os

DATA = "data/data.txt"
DATA_CAPS_WORDS = "data/data_capswords.txt"
TOKENS_DIR = "tokens"

DB = "results/db.sqlite"

# The name of the machine that will by default used in the DB to identify
# runs on the current system.
SYSTEM_NAME = "system"

SERVER_HOST = "localhost:5000"

# Bearer token for the authenticated API. Generated with
#   python -c "import secrets; print(secrets.token_urlsafe(16))"
# The texmo server requires this on /select and /add when reached
# via the authenticated endpoint. Clients send it as
# `Authorization: Bearer <key>`. Empty disables the authenticated port
# (every request returns 401), which is fine for LAN-only setups.
API_KEY = ""

# Training backend. 'jax' is the only one; the knob is kept as the
# seam for a future second backend.
BACKEND = 'jax'

# JAX platforms to initialize. Comma-separated list — e.g. 'cpu',
# 'cuda,cpu'. Setting this explicitly avoids the warning about
# failing to load the TPU plugin on Windows.
JAX_PLATFORMS = 'cpu'

# Number of background worker threads the client uses to sample
# training data. Sampling is random-read latency bound; the client
# reads with pread, whose I/O overlaps across threads, so a few
# workers keep even tiny models from starving on input. 1 disables
# the parallelism (single prefetch thread).
SAMPLE_THREADS = 4

# Whether this machine volunteers for loss-model refit jobs from the
# search server (a multi-minute predictor training; see
# docs/loss_prediction.md). Set False on slow machines: a worker
# whose turnaround exceeds the grant interval loses the run-count
# race and its fit is wasted. Only consulted when the server runs
# refits on workers (LOSS_REFIT below).
REFIT = True

# Where loss-model refits run: 'workers' hands the fit to a client
# via /select (the distributed flow), 'server' fits in-process on
# the model thread (needs a fast JAX platform in the server
# process -- see docs/loss_prediction.md).
LOSS_REFIT = 'workers'
