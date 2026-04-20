"""Shared helpers for the predictors (timing, loss)."""

from ..layers.dense import DenseDef
from ..layers.rnn import RnnDef
from ..layers.skip import SkipDef

# Losses above this are treated as "diverged" and not differentiated;
# below MIN_LOSS is a log-validity floor.
MIN_LOSS = 0.1
MAX_LOSS = 10


def layer_type_id(layer) -> str:
    """Stable categorical ID for a layer, used as a feature key.

    Dense and RNN include the activation (`dense.tanh`, `rnn.relu`) so
    different activations get their own model parameters. Skip includes
    the op (`skip.add`, `skip.cat`). Other layers use the bare name.
    """
    if isinstance(layer, DenseDef):
        return f"dense.{layer._activation}"
    if isinstance(layer, RnnDef):
        return f"rnn.{layer._activation}"
    if isinstance(layer, SkipDef):
        return f"skip.{layer.op}"
    return layer.name
