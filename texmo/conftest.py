import os

# torch must load its bundled DLLs before jax: winjax's initialize()
# (any jax import on win32) prepends the pip nvidia dirs to PATH, and
# torch's cudnn_cnn64_9.dll would then resolve its sister cudnn DLLs
# from the wrong copy -- WinError 127 at import. Torch-first is safe
# in both directions.
import torch  # noqa: F401
import jax

from texmo.tokens import set_tokens_dir

# Tests always run on CPU.
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platforms', 'cpu')

# `num_weights` reads a tokenset's `stats.extra_weights` through the
# registry and raises when the set can't be loaded (see
# layers/codec.py), so every test that parses a `tokens.*` spec needs
# the repo's tokens directory registered.
set_tokens_dir(os.path.join(os.path.dirname(__file__), '..', 'tokens'))
