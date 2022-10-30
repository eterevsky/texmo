from copy import deepcopy

from .model import NCHAR


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


def is_power2(x):
    return type(x) is int and x >= 1 and x & (x - 1) == 0


_layer_cache = {}


class LayerSpec(object):
    has_weights = True
    has_size = True
    has_activation = True

    def __init__(self, size=None, tanh=False, relu=False, sigmoid=False):
        if sigmoid:
            raise ObsoleteSpec("sigmoid")

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

        if not (
            self.has_activation
            and self._activation is not None
            or not self.has_activation
            and self._activation is None
        ):
            raise ObsoleteSpec("wrong activation")

        assert (
            self.has_activation
            and self._activation is not None
            or not self.has_activation
            and self._activation is None
        )

        self._str = self.to_str()

    @staticmethod
    def parse(spec: str):
        cached_layer = _layer_cache.get(spec)
        if cached_layer is not None:
            return cached_layer

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
            raise ObsoleteSpec(e)

        layer = layer_cls(*args, **kwargs)
        _layer_cache[spec] = layer
        return layer

    def __str__(self):
        return self._str

    def to_str(self):
        size = f".{self._size}" if self._size is not None else ""
        activation = (
            f".{self._activation}" if self._activation is not None else ""
        )
        return f"{self.name}{size}{activation}"

    def __eq__(self, other):
        return str(self) == str(other)

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
        copy._str = copy.to_str()
        if copy._str in _layer_cache:
            return _layer_cache[copy._str]
        else:
            _layer_cache[copy._str] = copy
            return copy

    def all_neighbors(self, input_shape):
        for neighbor in self.neighbors_size():
            yield neighbor, "size"

        for neighbor in self.neighbors_type(input_shape):
            if neighbor != self:
                yield neighbor, "type"

        if self.name in ("suffix", "attn"):
            for neighbor in self.neighbors_suffix_size():
                yield neighbor, "ssize"
        if self.name == "suffix":
            for neighbor in self.neighbors_suffix_type(input_shape):
                yield neighbor, "attn"
        if self.name == "attn":
            for neighbor in self.neighbors_suffix_type(input_shape):
                yield neighbor, "suffix"

    def neighbors_size(self):
        assert self._size is None or self.has_size or self.name == "attn"
        if self.has_size:
            assert self._size is not None
            if self._size > 1:
                yield self.replace(size=self._size // 2)
            yield self.replace(size=self._size * 2)

    def neighbors_struct(self, input_shape):
        for layer_cls in registry.values():
            if layer_cls.name == "caru":
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

    def neighbors_type(self, input_shape):
        if self.name in ("suffix", "attn"):
            return
        for layer_cls in (DenseSpec, RecSpec, GruSpec, MgruSpec, LstmSpec):
            if layer_cls.name == "gru":
                activations = ((False, True),)
            elif layer_cls.has_activation:
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

    def is_valid(self):
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

    def is_valid(self):
        return super().is_valid() and self._size >= 2 and is_power2(self._size)

    def neighbors_size(self):
        return ()

    def neighbors_type(self, input_shape):
        return ()

    def neighbors_suffix_type(self, input_shape):
        if self._size >= 2:
            yield AttnSpec(self._size, 4, max(total_size(input_shape), 8))

    def neighbors_suffix_size(self):
        if self._size >= 4:
            yield SuffixSpec(self._size // 2)
        yield SuffixSpec(self._size * 2)


# @layer_spec
# class AttentionSpec(LayerSpec):
#     name = "attention"
#     has_weights = False
#     has_size = True
#     has_activation = False

#     def __init__(self, size, pos=False):
#         super().__init__(size=size)
#         self._pos = pos

#     def __str__(self):
#         if self._pos:
#             return f"attention.{self._size}.pos"
#         else:
#             return f"attention.{self._size}"

#     def output_shape(self, input_shape):
#         return input_shape

#     def weights(self, input_shape):
#         return input_shape[0] * self._size if self._pos else 0

#     def is_valid(self):
#         return False


@layer_spec
class AttnSpec(LayerSpec):
    name = "attn"
    has_weights = False
    has_size = False
    has_activation = False

    def __init__(self, length, heads, size):
        super().__init__()
        self._length = length
        self._heads = heads
        self._size = size

    def __str__(self):
        return f"attn.{self._length}.{self._heads}.{self._size}"

    @property
    def comp_size(self):
        return self._size // self._heads

    def output_shape(self, input_shape):
        return (self._size,)

    def weights(self, input_shape):
        input_size = total_size(input_shape)
        return (
            3 * self._heads * self.comp_size * input_size
            + self._length * self._heads * self.comp_size
            + self._heads * self.comp_size
        )

    def is_valid(self):
        return (
            self._length >= 2
            and is_power2(self._length)
            and is_power2(self._heads)
            and is_power2(self._size)
            and self._size % self._heads == 0
        )

    def neighbors_suffix_type(self, input_shape):
        if self._heads == 4 and self._size == max(total_size(input_shape), 8):
            yield SuffixSpec(self._length)

    def neighbors_suffix_size(self):
        if self._length >= 4:
            yield AttnSpec(self._length // 2, self._heads, self._size)
        yield AttnSpec(self._length * 2, self._heads, self._size)

        if self._heads > 1:
            yield AttnSpec(self._length, self._heads // 2, self._size)
        if self._heads < self._size and self._size % (2 * self._heads) == 0:
            yield AttnSpec(self._length, self._heads * 2, self._size)

        if (
            self._heads < self._size
            and self._size >= 2
            and (self._size // 2) % self._heads == 0
        ):
            yield AttnSpec(self._length, self._heads, self._size // 2)
        yield AttnSpec(self._length, self._heads, self._size * 2)


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
    
    def is_valid(self):
        super().is_valid() and self._activation == "tanh"


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

    def is_valid(self):
        return False

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


models_cache = {}


class ModelSpec(object):
    def __init__(self, layer_specs):
        self._layers = list(layer_specs)
        self._str = self.to_str()
        self._hash = hash(self._str)
        self._neighbors = None

    @staticmethod
    def parse(spec: str):
        cached_model = models_cache.get(spec)
        if cached_model is not None:
            return cached_model

        components = spec.split("-")
        layers = []
        for component in components:
            layer = LayerSpec.parse(component)
            layers.append(layer)
        model = ModelSpec(layers)

        models_cache[spec] = model
        return model

    def __str__(self):
        return self._str

    def to_str(self):
        return "-".join(map(str, self._layers))

    def __repr__(self):
        s = str(self)
        return f"ModelSpec('{s}')"

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __hash__(self):
        return self._hash

    def weights(self):
        total = 0
        shape = (NCHAR,)
        for layer in self._layers:
            total += layer.weights(shape)
            shape = layer.output_shape(shape)
        total += total_size(shape) * NCHAR + NCHAR

        return total

    def is_valid(self):
        if len(self._layers) == 0:
            return False
        # has_non_suffix_layer = False
        for layer in self._layers:
            if not layer.is_valid():
                return False
            # if layer.name in ("dense", "rec", "gru", "mgru", "lstm"):
            #     has_non_suffix_layer = True
        return True

    def output_shape(self):
        shape = (NCHAR,)
        for layer in self._layers:
            shape = layer.output_shape(shape)
        return shape

    def _all_neighbors(self):
        assert self.is_valid()
        shape = (NCHAR,)

        suffix2 = LayerSpec.parse("suffix.2")

        for i, layer in enumerate(self._layers):
            for mod_layer, vary in layer.all_neighbors(shape):
                neighbor = ModelSpec(
                    self._layers[:i] + [mod_layer] + self._layers[i + 1 :]
                )
                assert neighbor.is_valid(), f"Invalid neighbor: {self} {layer} {vary} {mod_layer} {neighbor}"
                yield neighbor, vary

            attn_output_size = max(2, total_size(shape))
            attn_layer = AttnSpec(4, 2, attn_output_size)

            if len(self._layers) > 1:
                if layer == suffix2:
                    neighbor = ModelSpec(self._layers[:i] + self._layers[i + 1 :])
                    assert neighbor.is_valid()
                    yield neighbor, "suffix"
                elif layer == attn_layer:
                    neighbor = ModelSpec(self._layers[:i] + self._layers[i + 1 :])
                    assert neighbor.is_valid()
                    yield neighbor, "attn"

            if (
                i == 0
                or self._layers[i - 1].name
                not in ("suffix", "attention", "attn")
            ) and layer.name not in ("suffix", "attention", "attn"):
                neighbor = ModelSpec(
                    self._layers[:i] + [suffix2] + self._layers[i:]
                )
                assert neighbor.is_valid()
                yield neighbor, "suffix"

                neighbor = ModelSpec(
                    self._layers[:i] + [attn_layer] + self._layers[i:]
                )
                assert neighbor.is_valid()
                yield neighbor, "attn"

            shape = layer.output_shape(shape)

        if self._layers[-1].name not in ("suffix", "attention", "attn"):
            neighbor = ModelSpec(self._layers + [suffix2])
            assert neighbor.is_valid()
            yield neighbor, "suffix"

            output_size = max(2, total_size(shape))

            neighbor = ModelSpec(self._layers + [AttnSpec(4, 2, output_size)])
            assert neighbor.is_valid()
            yield neighbor, "suffix"

        for neighbor in self.neighbors_add_layer():
            yield neighbor, "layer"

        neighbor = self.neighbor_remove_layer()
        if neighbor is not None:
            yield neighbor, "layer"

    def all_neighbors(self):
        if self._neighbors is None:
            self._neighbors = list(self._all_neighbors())
        return self._neighbors

    def neighbors_add_layer(self):
        end_size = min(total_size(self.output_shape()), NCHAR // 2)
        for size in (end_size, end_size * 2):
            for layer_str in (
                f"dense.{size}.relu",
                f"dense.{size}.tanh",
                f"rec.{size}.relu",
                f"rec.{size}.tanh",
                f"gru.{size}.tanh",
                f"mgru.{size}",
                f"lstm.{size}",
            ):
                layer = LayerSpec.parse(layer_str)
                neighbor = ModelSpec(self._layers + [layer])
                assert neighbor.is_valid()
                yield neighbor

    def neighbor_remove_layer(self):
        TRANSFORM_LAYERS = ("dense", "rec", "gru", "mgru", "lstm")

        if len(self._layers) < 2:
            return
        if self._layers[-1].name not in TRANSFORM_LAYERS:
            return

        has_another_transform_layer = False
        shape = (NCHAR,)
        for layer in self._layers[:-1]:
            shape = layer.output_shape(shape)
            if layer.name in TRANSFORM_LAYERS:
                has_another_transform_layer = True

        # We can't leave only suffix / attention layers.
        if not has_another_transform_layer:
            return

        # Output size of the last but one layer.
        end_size = min(total_size(shape), NCHAR // 2)

        if self._layers[-1]._size not in (end_size, end_size * 2):
            return

        neighbor = ModelSpec(self._layers[:-1])
        assert neighbor.is_valid()
        return neighbor


def is_reachable_spec(init_spec: ModelSpec, spec: ModelSpec, vary):
    if "suffix" not in vary:
        for layer in spec._layers:
            if layer.name == "suffix":
                return False

    if "attn" not in vary:
        for layer in spec._layers:
            if layer.name == "attn":
                return False

    init_non_suffix_layers = []
    for layer in init_spec._layers:
        if layer.name not in ("suffix", "attention", "attn"):
            init_non_suffix_layers.append(layer)

    spec_non_suffix_layers = []
    for layer in spec._layers:
        if layer.name not in ("suffix", "attention", "attn"):
            spec_non_suffix_layers.append(layer)

    if "layer" not in vary and len(init_non_suffix_layers) != len(
        spec_non_suffix_layers
    ):
        return False

    if "type" not in vary:
        for l1, l2 in zip(init_non_suffix_layers, spec_non_suffix_layers):
            if (
                l1.name != l2.name
                or l1.has_activation
                and l1._activation != l2._activation
            ):
                return False

    if "size" not in vary:
        for l1, l2 in zip(init_non_suffix_layers, spec_non_suffix_layers):
            if l1._size != l2._size:
                return False

    return True


if __name__ == "__main__":
    spec = ModelSpec.parse("rec.8.tanh-suffix.32-dense.256.tanh")
    for neighbor, vary in spec.all_neighbors():
        print(str(neighbor), vary)
