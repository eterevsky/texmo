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
        layer_objs = []
        counter = {}
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
            counter[name] = counter.get(name, 0) + 1
            self._layers.append((f'{name}{counter[name]}', layer_obj))

    @staticmethod
    def parse(spec):
        return LayeredModel(spec.split('-'))

    @property
    def full_name(self):
        return '-'.join(l.full_name for n, l in self._layers)

    def serialize(self):
        spec = {
            'name': 'layered',
            'layers': []
        }
        for name, layer in self._layers:
            spec['layers'].append(layer.full_name)
        return spec

    def init_weights(self, key):
        weights = {}
        for name, layer in self._layers:
            r, key = split(key)
            weights[name] = layer.init_weights(r)
        return weights

    def init_state(self):
        state = {}
        for name, layer in self._layers:
            state[name] = layer.init_state()
        return state

    def step(self, weights, state, input):
        new_state = {}
        v = input
        for name, layer in self._layers:
            layer_state, v = layer.step2(weights[name], state[name], v)
            new_state[name] = layer_state
        return new_state, v