from copy import deepcopy
from unittest import result

from model import NCHAR


class ObsoleteSpec(Exception):
    pass


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
    has_weights = True
    has_size = True
    has_activation = True

    def __init__(self, size=None, tanh=False, relu=False, sigmoid=False):
        if sigmoid:
            raise ObsoleteSpec('sigmoid')

        if self.has_size:
            assert size is not None
            self._size = size
        else:
            assert size is None
            self._size = None

        if tanh:
            self._activation = "tanh"
        elif relu:
            self._activation = "relu"
        else:
            self._activation = None

        assert (
            self.has_activation
            and self._activation is not None
            or not self.has_activation
            and self._activation is None
        )

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

        try:
            layer_cls = registry[name]
        except KeyError as e:
            if name == 'norm':
                raise ObsoleteSpec(e)
        return layer_cls(*args, **kwargs)

    def __str__(self):
        size = f".{self._size}" if self._size is not None else ""
        activation = (
            f".{self._activation}" if self._activation is not None else ""
        )
        return f"{self.name}{size}{activation}"

    def output_shape(self, input_shape):
        return (self._size,)

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

    def neighbors(self, vary, input_shape):
        if "size" in vary:
            for neighbor in self.neighbors_size():
                yield neighbor
        if "activation" in vary or "act" in vary:
            for neighbor in self.neighbors_activation():
                yield neighbor
        if "struct" in vary and self.has_weights or "suffix" in vary and self.name in ("suffix", "attention"):
            for neighbor in self.neighbors_struct(input_shape):
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

    def neighbors_struct(self, input_shape):
        for layer_cls in registry.values():
            if layer_cls.name == 'caru':
                continue
            if not layer_cls.has_weights:
                continue

            if layer_cls.has_activation:
                activations = ((True, False), (False, True))
            else:
                activations = ((False, False),)

            for relu, tanh in activations:
                new_layer = layer_cls(size=self._size, relu=relu, tanh=tanh)
                yield new_layer

                if new_layer.weights(input_shape) < self.weights(input_shape):
                    yield layer_cls(size=2 * self._size, relu=relu, tanh=tanh)

                if (
                    new_layer.weights(input_shape) > self.weights(input_shape)
                    and self._size > 1
                ):
                    yield layer_cls(size=self._size // 2, relu=relu, tanh=tanh)

    def is_valid(self, prev_layer):
        return (
            self._size is None or (type(self._size) is int and self._size >= 1)
        ) and self._activation in (None, "tanh", "relu")


@layer_spec
class SuffixSpec(LayerSpec):
    name = "suffix"
    has_weights = False
    has_size = True
    has_activation = False

    def __init__(self, size):
        super().__init__(size=size)

    def output_shape(self, input_shape):
        return (self._size,) + input_shape

    def weights(self, input_shape):
        return 0

    def is_valid(self, prev_layer):
        return (
            super().is_valid(prev_layer)
            and self._size >= 2
            and self._size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
            and (prev_layer is None or prev_layer.name not in ("suffix", "attention"))
        )

    def neighbors_struct(self, input_shape):
        yield AttentionSpec(self._size, True)
        yield AttentionSpec(self._size, False)


@layer_spec
class AttentionSpec(LayerSpec):
    name = "attention"
    has_weights = False
    has_size = True
    has_activation = False

    def __init__(self, size, pos=False):
        super().__init__(size=size)
        self._pos = pos

    def __str__(self):
        if self._pos:
            return f"attention.{self._size}.pos"
        else:
            return f"attention.{self._size}"

    def output_shape(self, input_shape):
        return input_shape

    def weights(self, input_shape):
        return input_shape[0] * self._size if self._pos else 0

    def is_valid(self, prev_layer):
        return (
            super().is_valid(prev_layer)
            and self._size >= 2
            and (prev_layer is None or prev_layer.name not in ("suffix", "attention"))
        )

    def neighbors_struct(self, input_shape):
        yield SuffixSpec(self._size)
        yield AttentionSpec(self._size, not self._pos)


@layer_spec
class DenseSpec(LayerSpec):
    name = "dense"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weights(self, input_shape):
        return total_size(input_shape) * self._size + self._size


@layer_spec
class RecSpec(LayerSpec):
    name = "rec"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weights(self, input_shape):
        return (
            total_size(input_shape) * self._size
            + self._size * self._size
            + self._size
        )


@layer_spec
class GruSpec(LayerSpec):
    name = "gru"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def weights(self, input_shape):
        return 3 * (
            total_size(input_shape) * self._size
            + self._size * self._size
            + self._size
        )


@layer_spec
class MgruSpec(LayerSpec):
    name = "mgru"
    has_activation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._activation is None

    def weights(self, input_shape):
        return 2 * (
            total_size(input_shape) * self._size
            + self._size * self._size
            + self._size
        )


@layer_spec
class CaruSpec(LayerSpec):
    name = "caru"
    has_activation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._activation is None

    def weights(self, input_shape):
        return (
            2 * total_size(input_shape) * self._size
            + 2 * self._size * self._size
            + 3 * self._size
        )


@layer_spec
class LstmSpec(LayerSpec):
    name = "lstm"
    has_activation = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self._activation is None

    def weights(self, input_shape):
        return 4 * (
            total_size(input_shape) * self._size
            + self._size * self._size
            + self._size
        )


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

    def __lt__(self, other):
        return str(self) < str(other)

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
        if len(self._layers) == 0:
            return False
        for layer in self._layers:
            if not layer.is_valid(prev_layer):
                return False
            prev_layer = layer
        return True

    def simplify(self):
        i = 0
        while i < len(self._layers):
            if str(self._layers[i]) in ("suffix.1", "attention.1", "attention.1.pos"):
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
        assert self.is_valid()
        shape = (NCHAR,)
        for i, layer in enumerate(self._layers):
            for mod_layer in layer.neighbors(vary, shape):
                neighbor = ModelSpec(
                    self._layers[:i] + [mod_layer] + self._layers[i + 1 :]
                )
                neighbor.simplify()
                if not neighbor.is_valid():
                    continue
                assert neighbor.is_valid()
                yield neighbor

            if (
                "suffix" in vary
                and (i == 0 or self._layers[i - 1].name not in ("suffix", "attention"))
                and self._layers[i].name not in ("suffix", "attention")
            ):
                for layer in (SuffixSpec(2), AttentionSpec(2, False), AttentionSpec(2, True)):
                    neighbor = ModelSpec(
                        self._layers[:i] + [layer] + self._layers[i:]
                    )
                    if not neighbor.is_valid():
                        print(neighbor)
                    assert neighbor.is_valid()
                    yield neighbor

            shape = layer.output_shape(shape)

        if "suffix" in vary and self._layers[-1].name not in ("suffix", "attention"):
                for layer in (SuffixSpec(2), AttentionSpec(2, False), AttentionSpec(2, True)):
                    neighbor = ModelSpec(self._layers + [layer])
                    if not neighbor.is_valid():
                        print(self._layers[-1].name)
                        print(neighbor)
                    assert neighbor.is_valid()
                    yield neighbor

        if "struct" in vary:
            size = min(total_size(self.output_shape()), NCHAR // 2)
            neighbor = ModelSpec(self._layers + [DenseSpec(size, relu=True)])
            assert neighbor.is_valid()
            yield neighbor

            if len(self._layers) > 1 and self._layers[-1].has_weights:
                neighbor = ModelSpec(self._layers[:-1])
                assert neighbor.is_valid()
                yield neighbor


if __name__ == "__main__":
    spec = ModelSpec.parse("attention.4-dense.128.relu")
    for neighbor in spec.neighbors(
        vary=["struct", "suffix", "size", "activation"]
    ):
        print(str(neighbor))
