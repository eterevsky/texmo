from copy import deepcopy
from collections.abc import Sequence


registry = {}


def layer_spec(cls):
    assert cls.name not in registry
    registry[cls.name] = cls
    return cls


class LayerSpec(object):
    def __init__(self, size=None, activation=None):
        self._size = size
        self._activation = activation

    @staticmethod
    def parse(spec: str):
        components = spec.split(".")
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

        layer_cls = registry[name]
        return layer_cls(*args, **kwargs)

    def __str__(self):
        raise NotImplementedError

    def output_shape(self, input_shape):
        raise NotImplementedError

    def weights(self, input_shape):
        raise NotImplementedError

    def replace(self, **kwargs):
        copy = deepcopy(self)
        for key, value in kwargs.items():
            if key == "size":
                copy._size = value
            elif key == "activation":
                copy._activation = value
            else:
                raise KeyError
        return copy

    def neighbors(self, vary):
        if "size" in vary:
            for neighbor in self.neighbors_size():
                yield neighbor
        if "activation" in vary or "act" in vary:
            for neighbor in self.neighbors_activation():
                yield neighbor

    def neighbors_size(self):
        if self._size is not None:
            if self._size > 1:
                yield self.replace(size=self.size // 2)
            yield self.replace(size=self.size * 2)

    def neighbors_activation(self):
        if self._activation is None:
            return
        if self._activation is not "tanh":
            yield self.replace(activation="tanh")
        if self._activation is not "relu":
            yield self.replace(activation="relu")

    def is_valid(self):
        return (
            self._size is None or (type(self._size) is int and self._size >= 1)
        ) and self._activation in (None, "tanh", "relu")


class SuffixSpec(LayerSpec):
    def __init__(self, length):
        super().__init__()
        self._length = length

    def output_shape(self, input_shape):
        return (self._length) + input_shape

    def weights(self, input_shape):
        return 0

    def replace(self, **kwargs):
        copy = deepcopy(self)
        for key, value in kwargs.items():
            if key == "length":
                self._length = value
            else:
                raise KeyError
        return copy

    def neighbors_size(self):
        if self._length > 1:
            yield self.replace(length=self._length - 1)
        yield self.replace(length=self._length + 1)

    def is_valid(self):
        return type(self._length) is int and self._length >= 1


class ModelSpec(object):
    def __init__(self, layer_specs: Sequence[LayerSpec]):
        self._layers = list(layer_specs)

    @staticmethod
    def parse(spec: str):
        components = spec.split("-")
        layers = []
        for component in components:
            layer = LayerSpec.parse(component)
            layers.append(layer)
        return ModelSpec(layers)

    def __str__(self):
        return "-".join(map(str, self._layers))
