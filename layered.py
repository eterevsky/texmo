from jax.random import split

from layers import LAYERS_BY_NAME
import layers2
from model import Model, NCHAR
import prng


class LayeredModel2(Model):
    """A model composed of a series of layers.

    The layers are such that the first one accepts a single character (NCHAR,)
    input, the input for each following layer matches the shape of the output
    of the previous layer, the last layer outputs (NCHAR,)
    """

    name = "custom2"

    def __init__(self, named_layers):
        """
        Args:
            named_layers: sequence of tuples (layer name, layer spec), for
                example:[('gru1', 'gru.128'), ('gru2', 'gru.512), ('dense_out', 'dense.256')]
        """
        self._layers = []
        shape = (NCHAR,)
        for layer_name, layer in named_layers:
            components = layer.split(".")
            name = components[0]
            params = components[1:]
            args = []
            kwargs = {}
            for param in params:
                try:
                    p = int(param)
                    args.append(p)
                except ValueError:
                    kwargs[param] = True

            layer_cls = layers2.registry[name]
            layer_obj = layer_cls(*args, input_shape=shape, **kwargs)
            shape = layer_obj.output_shape
            self._layers.append((layer_name, layer_obj))

    @staticmethod
    def parse(spec: str):
        layers = spec.split("-")
        named_layers = []
        counter = {}
        for layer in layers:
            name = layer.split(".")[0]
            if name not in counter:
                counter[name] = 0
            counter[name] += 1
            layer_name = f"{name}{counter[name]}"
            named_layers.append((layer_name, layer))
        named_layers.append(("dense_out", f"dense.{NCHAR}"))
        return LayeredModel2(named_layers)

    @staticmethod
    def from_spec(spec):
        return LayeredModel2(spec["layers"])

    @property
    def full_name(self):
        return "-".join(l.full_name for _, l in self._layers[:-1])

    def serialize(self):
        spec = {"name": "layered", "layers": []}
        for name, layer in self._layers:
            spec["layers"].append([name, layer.full_name])
        return spec

    def init_weights(self, key):
        rng = prng.Rng(key)
        weights = {}
        for name, layer in self._layers:
            weights[name] = layer.init_weights(rng)
        return weights

    def init_state(self, weights):
        state = {}
        for name, layer in self._layers:
            state[name] = layer.init_state(weights[name])
        return state

    def step(self, weights, state, input):
        new_state = {}
        v = input
        for name, layer in self._layers:
            layer_state, v = layer.step(weights[name], state[name], v)
            new_state[name] = layer_state
        return new_state, v
