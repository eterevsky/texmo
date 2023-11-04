from itertools import chain
from typing import Iterable, Self

from .layers import build_layer
from .layers.dense import Dense
from .layers.input import Input


_SUFFIX_LIKE_LAYERS = ("suffix", "attn", "attnmq")


class Model3(object):
    def __init__(self, spec: str):
        spec_parts = spec.split("|")

        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise AssertionError("Model spec can't contain more than one |")

        self.input = Input.from_spec(input_spec)

        self.layers = []
        shape = self.input.output_shape
        for layer_spec in layers_spec.split("-"):
            layer = build_layer(layer_spec, shape)
            self.layers.append(layer)
            shape = layer.output_shape

        self.output = Dense(self.input.ntokens, input_shape=shape)

    def __str__(self) -> str:
        return str(self.input) + "|" + "-".join(map(str, self.layers))

    def __eq__(self, other: Self) -> bool:
        return str(self) == str(other)

    def is_valid(self) -> bool:
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            if (
                layer1.name in _SUFFIX_LIKE_LAYERS
                and layer2.name in _SUFFIX_LIKE_LAYERS
            ):
                return False
        return all(l.is_valid() for l in self.layers)

    @property
    def weights(self) -> int:
        return (
            self.input.weights
            + sum(l.weights for l in self.layers)
            + self.out_layer.weights
        )

    def _gen_neighbors(self) -> Iterable[str]:
        layers_str = [str(l) for l in self.layers]
        layers_spec = "-".join(layers_str)
        input_spec = str(self.input)

        for input_neighbor in self.input.neighbors():
            yield str(input_neighbor) + "|" + layers_spec

        # Mutate every separate layer
        for i in range(len(layers_str)):
            for layer_neighbor in self.layers[i].neighbors():
                yield input_spec + "|" + "-".join(
                    chain(
                        layers_str[:i],
                        (str(layer_neighbor),),
                        layers_str[i + 1 :],
                    )
                )

        # Add layer to the end
        new_layer_size = min(self.layers[-1].output_size, self.input.ntokens)
        for layer_type in ("dense", "rec"):
            for activation in ("gelu", "relu", "tanh"):
                yield input_spec + "|" + "-".join(
                    chain(layers_str, (f"{layer_type}.{new_layer_size}.{activation}",))
                )

        for layer_type in ("gru", "mgru", "lstm"):
            yield input_spec + "|" + "-".join(
                chain(layers_str, (f"{layer_type}.{new_layer_size}",))
            )

        yield input_spec + "|" + "-".join(chain(layers_str, (f"suffix.2",)))
        yield input_spec + "|" + "-".join(
            chain(layers_str, (f"attn.2.2.{new_layer_size}",))
        )
        yield input_spec + "|" + "-".join(
            chain(layers_str, (f"attnmq.2.2.{new_layer_size}",))
        )

        if self.layers:
            if len(self.layers) == 1:
                prev_size = self.input.output_size
            else:
                prev_size = self.layers[-1].output_size

            output_size = min(prev_size, self.input.ntokens)
            last_layer = self.layers[-1]

            valid_to_remove = False
            if last_layer.name in _SUFFIX_LIKE_LAYERS:
                valid_to_remove = last_layer in (
                    "suffix.2",
                    "attn.2.2.{output_size}",
                    "attnmq.2.2.{output_size}",
                )
            else:
                valid_to_remove = last_layer.output_size == output_size

            if valid_to_remove:
                yield input_spec + "|" + "-".join(layers_str[:-1])

    def neighbors(self):
        for spec in self._gen_neighbors():
            model = build_model(spec)
            if model.is_valid() and model != self:
                yield model


_cache: dict[str, Model3] = {}


def build_model(spec: str) -> Model3:
    model = _cache.get(spec)
    if model is None:
        model = Model3(spec)
        _cache[spec] = model
    return model
