from jax.random import split

from layers import LAYERS_BY_NAME
import layers2
from model import Model, NCHAR
import prng


class LayeredModel(Model):
    """A straight-forward model composed of a series of layers.

    The layers are such that the first one accepts a single character (NCHAR,)
    input, the input for each following layer matches the shape of the output
    of the previous layer, the last layer outputs (NCHAR,)
    """

    name = 'custom'

    def __init__(self, layers, use_layers2=True):
        self._layers = []
        counter = {}
        shape = (NCHAR,)
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

            if use_layers2:
                layer_cls = layers2.registry[name]
                layer_obj = layer_cls(*args, input_shape=shape, **kwargs)
                shape = layer_obj.output_shape
            else:
                layer_obj = LAYERS_BY_NAME[name](*args, **kwargs)
            counter[name] = counter.get(name, 0) + 1
            self._layers.append((f'{name}{counter[name]}', layer_obj))

        if use_layers2:
            final_layer = layers2.Dense(NCHAR, input_shape=shape)
            self._layers.append((f'dense_out', final_layer))

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

    def init_state(self, weights):
        state = {}
        for name, layer in self._layers:
            state[name] = layer.init_state(weights)
        return state

    def step(self, weights, state, input):
        new_state = {}
        v = input
        for name, layer in self._layers:
            layer_state, v = layer.step(weights[name], state[name], v)
            new_state[name] = layer_state
        return new_state, v



class LayeredModel2(Model):
    """A straight-forward model composed of a series of layers.

    The layers are such that the first one accepts a single character (NCHAR,)
    input, the input for each following layer matches the shape of the output
    of the previous layer, the last layer outputs (NCHAR,)
    """

    name = 'custom2'

    def __init__(self, layers):
        self._layers = []
        counter = {}
        shape = (NCHAR,)
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

            layer_cls = layers2.registry[name]
            layer_obj = layer_cls(*args, input_shape=shape, **kwargs)
            shape = layer_obj.output_shape
            counter[name] = counter.get(name, 0) + 1
            self._layers.append((f'{name}{counter[name]}', layer_obj))

        final_layer = layers2.Dense(NCHAR, input_shape=shape)
        self._layers.append((f'dense_out', final_layer))

    @staticmethod
    def parse(spec):
        return LayeredModel2(spec.split('-'))

    @property
    def full_name(self):
        return '-'.join(l.full_name for _, l in self._layers[:-1])

    def serialize(self):
        spec = {
            'name': 'layered',
            'layers': []
        }
        for name, layer in self._layers:
            spec['layers'].append([name, layer.full_name])
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