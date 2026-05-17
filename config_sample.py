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

# PyTorch device: 'cuda', 'cpu', or 'auto' (auto picks cuda if available)
TORCH_DEVICE = 'auto'

# Training backend: 'torch' or 'jax'
# PyTorch has fused kernels for standard layers (GRU, LSTM, RNN with
# tanh/relu) but loop-based custom layers are slow.
# JAX compiles custom recurrent layers via lax.scan + jit, making them
# competitive with hand-fused kernels.
BACKEND = 'jax'

# JAX platforms to initialize. Comma-separated list — e.g. 'cpu',
# 'cuda,cpu'. Setting this explicitly avoids the warning about
# failing to load the TPU plugin on Windows.
JAX_PLATFORMS = 'cpu'
