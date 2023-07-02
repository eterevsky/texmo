from itertools import chain

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.numpy import DeviceArray

from .common import NCHAR
from .layer import Layer, LayerState, LayerWeights
from .layers import build_layer
from .layers.dense import Dense
from .prng import Rng

Weights = list[LayerWeights]
State = list[LayerState]


class Model2(object):
    def __init__(self, ntokens: int, spec: str):
        self.ntokens = ntokens
        self.layers: list[Layer] = []
        shape = (ntokens,)
        for layer_spec in spec.split("-"):
            layer = build_layer(layer_spec, shape)
            self.layers.append(layer)
            shape = layer.output_shape
        self.out_layer: Layer = Dense(ntokens, input_shape=shape)

    def __str__(self):
        return "-".join(map(str, self.layers))

    def __eq__(self, other):
        return self.ntokens == other.ntokens and str(self) == str(other)

    def __lt__(self, other):
        if self.ntokens != other.ntokens:
            return self.ntokens < other.ntokens
        return str(self) < str(other)

    def __hash__(self):
        return hash((str(self), self.ntokens))

    def is_valid(self):
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            if layer1.name in ("suffix", "attn") and layer2.name in (
                "suffix",
                "attn",
            ):
                return False
        return all(l.is_valid() for l in self.layers)

    @property
    def weights(self) -> int:
        return sum(l.weights for l in self.layers) + self.out_layer.weights

    def _gen_neighbors(self):
        layers_str = [str(l) for l in self.layers]

        # Mutate every separate layer
        for i in range(len(layers_str)):
            for layer_neighbor in self.layers[i].neighbors():
                yield "-".join(
                    chain(
                        layers_str[:i],
                        (str(layer_neighbor),),
                        layers_str[i + 1 :],
                    )
                )

        if len(self.layers) > 1:
            # Remove the last layer
            if self.layers[-1] != "suffix.2" and self.layers[
                -1
            ].output_size == min(self.layers[-2].output_size, NCHAR):
                yield "-".join(layers_str[:-1])

            # Remove the intermediate suffix or attn layer
            for i in range(len(layers_str)):
                if layers_str[i] == "suffix.2":
                    yield "-".join(chain(layers_str[:i], layers_str[i + 1 :]))
                if i > 0:
                    size = self.layers[i].output_size
                    heads = min(size, 4)
                    if layers_str[i] == f"attn.4.{heads}.{size}":
                        yield "-".join(
                            chain(layers_str[:i], layers_str[i + 1 :])
                        )

        # Add layer to the end
        new_layer_size = min(self.layers[-1].output_size, NCHAR)
        for layer_type in ("dense", "rec"):
            for activation in ("relu", "tanh"):
                yield str(self) + f"-{layer_type}.{new_layer_size}.{activation}"

        for layer_type in ("gru", "mgru", "lstm"):
            yield str(self) + f"-{layer_type}.{new_layer_size}"

        # Add suffix.2 layers.
        yield "suffix.2-" + str(self)

        for i in range(len(layers_str)):
            # Inserting between layers i and i + 1
            yield "-".join(
                chain(layers_str[: i + 1], ["suffix.2"], layers_str[i + 1 :])
            )

            size = self.layers[i].output_size
            heads = min(size, 4)

            yield "-".join(
                chain(
                    layers_str[: i + 1],
                    [f"attn.4.{heads}.{size}"],
                    layers_str[i + 1 :],
                )
            )

    def neighbors(self):
        for spec in self._gen_neighbors():
            model = build_model(self.ntokens, spec)
            if model.is_valid() and model != self:
                yield model

    def add_layers(self, layers_spec):
        shape = self.layers[-1].output_shape
        for layer_spec in layers_spec.split("-"):
            layer = build_layer(layer_spec, shape)
            self.layers.append(layer)
            shape = layer.output_shape
        self.out_layer = Dense(self.ntokens, input_shape=shape)

    def remove_last_layer(self):
        if len(self.layers) == 1:
            return None
        return build_model(self.ntokens, "-".join(map(str, self.layers[:-1])))

    def init_weights(self, rng: Rng, init_scale: float) -> Weights:
        return [l.init_weights(rng, init_scale) for l in self.layers] + [
            self.out_layer.init_weights(rng, init_scale)
        ]

    def init_state(self, weights: Weights) -> State:
        return [l.init_state(w) for l, w in zip(self.layers, weights)]

    def step(
        self, weights: Weights, state: State, input: jnp.ndarray
    ) -> tuple[State, jnp.ndarray]:
        """Run inference for one input character.

        `input` is a one-hot array representing input character with shape
        (NCHAR,).

        Returns:
            Tuple of new state and an output 1D vector of size NCHAR that after
            softmax will produce character probabilities.
        """
        assert len(self.layers) == len(weights) - 1 == len(state)
        new_state = []
        v = input
        for layer, layer_weights, layer_state in zip(
            self.layers, weights[:-1], state
        ):
            layer_state, v = layer.step(layer_weights, layer_state, v)
            new_state.append(layer_state)
        _, out = self.out_layer.step(weights[-1], None, v)
        return new_state, out

    def step_prob(self, weights, state, c, temperature=1.0):
        """Runs one step of the model and calculate the final probabilities.

        Arguments are the same as in step().

        Returns:
            (new state, a 1D vector of size NCHAR with probabilities of characters)
        """
        state, out = self.step(weights, state, c)
        out_softmax = jax.nn.softmax(out / temperature)
        return state, out_softmax

    def _loss_batch_unpacked(
        self, weights: Weights, batch: jnp.ndarray
    ) -> DeviceArray:
        prefix_len = 1
        for layer in self.layers:
            if layer.name in ("suffix", "attn"):
                prefix_len += layer.length - 1
        batch_size, sample_len, n = batch.shape
        assert n == self.ntokens
        prefix = jnp.ones((batch_size, prefix_len, self.ntokens)) / self.ntokens
        v = jnp.concatenate([prefix, batch], axis=1)[:, :-1, :]
        assert v.shape == (
            batch_size,
            prefix_len + sample_len - 1,
            self.ntokens,
        )

        for layer, layer_weights in zip(self.layers, weights):
            v = layer.forward_batch(layer_weights, v)

        out = self.out_layer.forward_batch(weights[-1], v)
        assert out.shape == batch.shape
        entropy = optax.softmax_cross_entropy(out, batch)
        return entropy

    def loss_batch(self, weights: Weights, batch: jnp.ndarray) -> DeviceArray:
        """Compute the average loss over a batch of training data.

        Args:
            weights:
            batch: an array of shape (batch_size, sample_len, NCHAR)
        """

        entropy = self._loss_batch_unpacked(weights, batch)
        return jnp.average(entropy) / jnp.log(2)

    def loss_batch_masked(
        self, weights: Weights, batch: jnp.ndarray, lengths: jnp.ndarray
    ) -> DeviceArray:
        # print("loss_batch_masked")
        # print(batch.shape)
        # print(batch)
        # print(lengths)
        entropy = self._loss_batch_unpacked(weights, batch)
        # print(entropy)

        mask = jnp.arange(entropy.shape[1]) < lengths[:, np.newaxis]
        # print(lengths)
        # print(mask)
        entropy = entropy * mask

        # print(entropy)

        return np.sum(entropy) / jnp.log(2)


_cache: dict[str, Model2] = {}


def build_model(ntokens: int, spec: str) -> Model2:
    model = _cache.get((ntokens, spec))
    if model is None:
        model = Model2(ntokens, spec)
        _cache[(ntokens, spec)] = model
    return model
