from math import log2

from ..common import total_size

_layers = []
_activations = []


def get_layer_cat_features(layer):
    """Generate features for a layer, with some categorical features.

    Features: [layer, activation, input_size, output_size, weights, state size, suffix len]

    Categoricals: [1, 1, 0, 0, 0, 0, 0]
    """
    features = []
    try:
        layer_idx = _layers.index(layer.name)
    except ValueError:
        layer_idx = len(_layers)
        _layers.append(layer.name)
    features.append(layer_idx)

    if layer.name in ("dense", "rec"):
        activation = layer._activation_suffix[1:]
        try:
            activation_idx = _activations.index(activation)
        except ValueError:
            activation_idx = len(_activations)
            _activations.append(activation)
        features.append(activation_idx)
    else:
        features.append(None)

    features.append(log2(layer.input_size))
    features.append(log2(total_size(layer.output_shape)))
    if layer.weights > 0:
        features.append(log2(layer.weights))
    else:
        features.append(None)

    match layer.name:
        case "gru" | "mgru" | "rec":
            features.extend([log2(layer.size), 0])
        case "lstm":
            features.extend([log2(2 * layer.size), 0])
        case "dense":
            features.extend([0, 0])
        case "suffix" | "attn" | "attnmq":
            features.extend([0, log2(layer.length)])
        case _:
            raise ValueError(f"Unknown layer name: {layer.name}")

    return features


_token_types = []
_processing = []

def get_tokens_cat_features(token_set):
    token_type = token_set.token_type
    try:
        type_idx = _token_types.index(token_type)
    except ValueError:
        type_idx = len(_token_types)
        _token_types.append(token_type)

    try:
        processing_idx = _processing.index(token_set.processing)
    except ValueError:
        processing_idx = len(_processing)
        _processing.append(token_set.processing)

    return [
        type_idx, processing_idx, log2(token_set.ntokens), token_set.bytes_per_token, token_set.entropy0
    ]
