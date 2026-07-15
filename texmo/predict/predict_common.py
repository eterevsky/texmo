"""Shared helpers for the predictors (timing, loss)."""

from ..configuration import Configuration
from ..layers.dense import DenseDef
from ..layers.rnn import RnnDef
from ..layers.split import SplitDef

# Losses above this are treated as "diverged" and not differentiated;
# below MIN_LOSS is a log-validity floor.
MIN_LOSS = 0.1
MAX_LOSS = 10


def layer_type_id(layer) -> str:
    """Stable categorical ID for a layer, used as a feature key.

    Dense and RNN include the activation (`dense.tanh`, `rnn.gelu`) so
    different activations get their own model parameters. Split
    includes the op (`split.mul`). Other layers use the bare name.
    """
    if isinstance(layer, DenseDef):
        # Bare dense (no activation) only occurs as a gate path inside
        # split.mul; give it a clean "dense" key, not "dense.None".
        return f"dense.{layer._activation}" if layer._activation else "dense"
    if isinstance(layer, RnnDef):
        return f"rnn.{layer._activation}"
    if isinstance(layer, SplitDef):
        return f"split.{layer.op}"
    return layer.name


def _flatten_layers(layers: list, out: list) -> None:
    """Flatten a Model2 layer tree into a flat sequence: a split
    becomes a marker step followed by its main branch (branch 0)
    inline; the other/gate branch folds into the marker. Recurses for
    nested splits."""
    for layer in layers:
        out.append(layer)
        if isinstance(layer, SplitDef):
            _flatten_layers(layer.branches[0].layers, out)


def model_layers(conf: Configuration) -> list:
    """The conf's hidden-layer tree flattened into the marker-plus-
    inlined-main-branch form shared by the loss predictors."""
    out: list = []
    _flatten_layers(conf.model.layer_seq.layers, out)
    return out


def discover_type_ids(
    train_data: list[tuple[Configuration, float]],
) -> list[str]:
    """Sorted list of every layer type_id seen in training."""
    seen: set[str] = set()
    for conf, _ in train_data:
        for layer in model_layers(conf):
            seen.add(layer_type_id(layer))
    return sorted(seen)
