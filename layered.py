from jax.random import split

from layers import LAYERS_BY_NAME
from model import Model, NCHAR


class LayeredModel(Model):
    """A straight-forward model composed of a series of layers.

    The layers are such that the first one accepts a single character (NCHAR,)
    input, the input for each following layer matches the shape of the output
    of the previous layer, the last layer outputs (NCHAR,)
    """

    name = 'custom'

    def __init__(self, layers):
        self._layers = []

        layer_names = {}
        layer_objs = []
        for layer in layers:
            components = layer.split('.')
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
            layer_obj = LAYERS_BY_NAME[name](*args, **kwargs)
            layer_names[name] = layer_names.get(name, 0) + 1
            layer_objs.append(layer_obj)

        counter = {}
        for layer_obj in layer_objs:
            name = layer_obj.name
            if layer_names[name] == 1:
                self._layers.append((name, layer_obj))
            else:
                counter[name] = counter.get(name, 0) + 1
                self._layers.append((f'{name}{counter[name]}', layer_obj))

    @property
    def full_name(self):
        return '-'.join(l.full_name for l in self._layers)

    def serialize(self):
        spec = {
            'name': 'layered',
            'layers': []
        }
        for layer in self._layers:
            spec['layers'].append(layer.full_name)

    def init_weights(self, key):
        weights = {}
        keys = split(key, len(self._layers))
        for name, layer in self._layers:
            weights[name] = layer.init_weights(keys.pop())
        return weights

    def init_state(self):
        state = {}
        for name, layer in self._layers:
            state[name] = layer.init_state()

    def step(self, weights, state, input):
        new_state = {}
        v = input
        for name, layer in self._layers:
            layer_state, v = layer.step2(weights[name], state[name], v)
            new_state[name] = layer_state
        return new_state, v