from copy import deepcopy

from model import NCHAR


registry = {}


def layer_spec(cls):
    assert cls.name not in registry
    registry[cls.name] = cls
    return cls


def total_size(shape):
    prod = 1
    for dim in shape:
        prod *= dim
    return prod


class LayerSpec(object):
    def __init__(self, size=None, tanh=False, relu=False):
        self._size = size
        if tanh:
            self._activation = "tanh"
        elif relu:
            self._activation = "relu"
        else:
            self._activation = None

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
        if self.name == "suffix" and "suffix" in vary:
            for neighbor in self.neighbors_suffix():
                yield neighbor

    def neighbors_size(self):
        if self._size is not None:
            if self._size > 1:
                yield self.replace(size=self._size // 2)
            yield self.replace(size=self._size * 2)

    def neighbors_activation(self):
        if self._activation is None:
            return
        if self._activation != "tanh":
            yield self.replace(activation="tanh")
        if self._activation != "relu":
            yield self.replace(activation="relu")

    def is_valid(self, prev_layer):
        return (
            self._size is None or (type(self._size) is int and self._size >= 1)
        ) and self._activation in (None, "tanh", "relu")


@layer_spec
class SuffixSpec(LayerSpec):
    name = "suffix"

    def __init__(self, length):
        super().__init__()
        self._length = length

    def __str__(self):
        return f"suffix.{self._length}"

    def output_shape(self, input_shape):
        return (self._length,) + input_shape

    def weights(self, input_shape):
        return 0

    def replace(self, **kwargs):
        copy = deepcopy(self)
        for key, value in kwargs.items():
            if key == "length":
                copy._length = value
            else:
                raise KeyError
        return copy

    def neighbors_suffix(self):
        if self._length > 1:
            yield self.replace(length=self._length - 1)
        yield self.replace(length=self._length + 1)

    def is_valid(self, prev_layer):
        return (
            type(self._length) is int
            and self._length >= 2
            and (prev_layer is None or prev_layer.name != "suffix")
        )


@layer_spec
class DenseSpec(LayerSpec):
    name = "dense"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"dense.{self._size}.{self._activation}"

    def output_shape(self, input_shape):
        return (self._size,)

    def weights(self, input_shape):
        return total_size(input_shape) * self._size + self._size


class ModelSpec(object):
    def __init__(self, layer_specs):
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

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def weights(self):
        total = 0
        shape = (NCHAR,)
        for layer in self._layers:
            total += layer.weights(shape)
            shape = layer.output_shape(shape)
        total += total_size(shape) * NCHAR + NCHAR

        return total

    def is_valid(self):
        prev_layer = None
        for layer in self._layers:
            if not layer.is_valid(prev_layer):
                return False
            prev_layer = layer
        return True

    def simplify(self):
        i = 0
        while i < len(self._layers):
            if str(self._layers[i]) == "suffix.1":
                self._layers.pop(i)
            else:
                i += 1
        return self

    def output_shape(self):
        shape = (NCHAR,)
        for layer in self._layers:
            shape = layer.output_shape(shape)
        return shape

    def neighbors(self, vary):
        for i, layer in enumerate(self._layers):
            for mod_layer in layer.neighbors(vary):
                neighbor = ModelSpec(
                    self._layers[:i] + [mod_layer] + self._layers[i + 1 :]
                )
                neighbor.simplify()
                assert neighbor.is_valid()
                yield neighbor

            if (
                "suffix" in vary
                and (i == 0 or self._layers[i - 1].name != "suffix")
                and self._layers[i].name != "suffix"
            ):
                neighbor = ModelSpec(
                    self._layers[:i] + [SuffixSpec(2)] + self._layers[i:]
                )
                assert neighbor.is_valid()
                yield neighbor

        if "suffix" in vary and self._layers[-1].name != "suffix":
            neighbor = ModelSpec(self._layers + [SuffixSpec(2)])
            assert neighbor.is_valid()
            yield neighbor

        if "struct" in vary:
            size = min(total_size(self.output_shape()), NCHAR // 2)
            neighbor = ModelSpec(self._layers + [DenseSpec(size, relu=True)])
            assert neighbor.is_valid()
            yield neighbor


if __name__ == "__main__":
    spec = ModelSpec.parse("suffix.2-dense.128.tanh")
    for neighbor in spec.neighbors(vary=["struct", "suffix", "size", "activation"]):
        print(str(neighbor))