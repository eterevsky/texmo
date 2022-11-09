_registry = {}
_cache = {}


def layer_cls(cls):
    assert cls.name not in _registry
    _registry[cls.name] = cls
    return cls


def build_layer(spec, input_shape):
    key = spec + str(input_shape)
    layer = _cache.get(key)

    if layer is None:
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

        layer_cls = _registry[name]
        layer = layer_cls(*args, input_shape=input_shape, **kwargs)
        _cache[key] = layer

    return layer