import jax
import jax.numpy as jnp
from itertools import chain
from typing import Iterable, Self
import math
import optax

from .layer import Layer, LayerState, LayerWeights
from .layers import build_layer
from .layers.dense import Dense
from .layers.input import Input
from .prng import Rng


_1_BY_LOG2 = 1.0 / math.log(2.0)
_SUFFIX_LIKE_LAYERS = ("suffix", "attn", "attnmq")


Weights = list[LayerWeights]
State = list[LayerState]


class Model3(object):
    def __init__(self, spec: str):
        spec_parts = spec.split("|")

        if len(spec_parts) == 1:
            input_spec = ""
            layers_spec = spec_parts[0]
        elif len(spec_parts) == 2:
            input_spec, layers_spec = spec_parts
        else:
            raise AssertionError("Model spec can't contain more than one |")

        self.input = Input.from_spec(input_spec)

        self.layers = []
        shape = self.input.output_shape
        if layers_spec:
            for layer_spec in layers_spec.split("-"):
                layer = build_layer(layer_spec, shape)
                self.layers.append(layer)
                shape = layer.output_shape

        self.output = Dense(self.input.ntokens, input_shape=shape)

    def __str__(self) -> str:
        return str(self.input) + "|" + "-".join(map(str, self.layers))

    def __eq__(self, other: Self) -> bool:
        return str(self) == str(other)

    def is_valid(self) -> bool:
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            if (
                layer1.name in _SUFFIX_LIKE_LAYERS
                and layer2.name in _SUFFIX_LIKE_LAYERS
            ):
                return False
        return all(l.is_valid() for l in self.layers)

    @property
    def weights(self) -> int:
        return (
            self.input.weights
            + sum(l.weights for l in self.layers)
            + self.output.weights
        )

    def _gen_neighbors(self) -> Iterable[str]:
        layers_str = [str(l) for l in self.layers]
        layers_spec = "-".join(layers_str)
        input_spec = str(self.input)

        for input_neighbor in self.input.neighbors():
            yield str(input_neighbor) + "|" + layers_spec

        # Mutate every separate layer
        for i in range(len(layers_str)):
            for layer_neighbor in self.layers[i].neighbors():
                yield input_spec + "|" + "-".join(
                    chain(
                        layers_str[:i],
                        (str(layer_neighbor),),
                        layers_str[i + 1 :],
                    )
                )

        # Add layer to the end
        new_layer_size = min(self.layers[-1].output_size, self.input.ntokens)
        for layer_type in ("dense", "rec"):
            for activation in ("gelu", "relu", "tanh"):
                yield input_spec + "|" + "-".join(
                    chain(
                        layers_str,
                        (f"{layer_type}.{new_layer_size}.{activation}",),
                    )
                )

        for layer_type in ("gru", "mgru", "lstm"):
            yield input_spec + "|" + "-".join(
                chain(layers_str, (f"{layer_type}.{new_layer_size}",))
            )

        yield input_spec + "|" + "-".join(chain(layers_str, (f"suffix.2",)))
        if new_layer_size >= 2:
            yield input_spec + "|" + "-".join(
                chain(layers_str, (f"attn.2.2.{new_layer_size}",))
            )
            yield input_spec + "|" + "-".join(
                chain(layers_str, (f"attnmq.2.2.{new_layer_size}",))
            )

        if self.layers:
            if len(self.layers) == 1:
                prev_size = self.input.output_size
            else:
                prev_size = self.layers[-1].output_size

            output_size = min(prev_size, self.input.ntokens)
            last_layer = self.layers[-1]

            valid_to_remove = False
            if last_layer.name in _SUFFIX_LIKE_LAYERS:
                valid_to_remove = last_layer in (
                    "suffix.2",
                    "attn.2.2.{output_size}",
                    "attnmq.2.2.{output_size}",
                )
            else:
                valid_to_remove = last_layer.output_size == output_size

            if valid_to_remove:
                yield input_spec + "|" + "-".join(layers_str[:-1])

    def neighbors(self):
        for spec in self._gen_neighbors():
            model = build_model(spec)
            if model.is_valid() and model != self:
                yield model

    def init_weights(self, rng: Rng, init_scale: float = 1.0):
        return (
            [self.input.init_weights(rng)]
            + [l.init_weights(rng, init_scale) for l in self.layers]
            + [self.output.init_weights(rng, init_scale)]
        )

    def init_state(self, weights: Weights) -> State:
        return [self.input.init_state(weights[0])] + [
            l.init_state(w) for l, w in zip(self.layers, weights[1:-1])
        ]

    def initial_step(self, weights: Weights) -> tuple[State, jax.Array]:
        """Predict the first token."""
        input_state, v = self.input.initial_step(weights[0])
        new_state = [input_state]

        for layer, layer_weights in zip(self.layers, weights[1:-1]):
            init_state = layer.init_state(layer_weights)
            layer_state, v = layer.step(layer_weights, init_state, v)
            new_state.append(layer_state)

        _, out = self.output.step(weights[-1], None, v)
        return new_state, out

    def step(
        self, weights: Weights, state: State, input_char: int
    ) -> tuple[State, jax.Array]:
        """Run inference for one input character.

        Args:
            input: is a single token index

        Returns:
            Tuple of new state and an output 1D vector of size ntokens that
            after softmax will produce character probabilities.
        """
        new_state = []

        input_state, v = self.input.step(weights[0], state[0], input_char)
        new_state.append(input_state)

        for layer, layer_weights, layer_state in zip(
            self.layers, weights[1:-1], state[1:]
        ):
            layer_state, v = layer.step(layer_weights, layer_state, v)
            new_state.append(layer_state)

        _, out = self.output.step(weights[-1], None, v)
        return new_state, out

    def step_prob(
        self,
        weights: Weights,
        state: State,
        input_char: int,
        temperature=1.0,
    ):
        state, out = self.step(weights, state, input_char)
        return state, jax.nn.softmax(out / temperature)

    def step_sample(
        self,
        weights: Weights,
        state: State,
        input_char: int,
        rng: Rng,
        temperature=1.0,
    ):
        """Runs a single step and samples from the output distribution.

        Args:
            weights: model weights
            state: state based on preceding text
            input_char: a token index
            temperature: softmax temperature: 0 would always return the most
                probable token, 1 selects each token with its probability

        Returns:
            (new state, index of the sampled token)
        """
        state, out_softmax = self.step_prob(weights, state, input_char, temperature)
        c_selected = jax.random.choice(rng.gen(), self.input.ntokens, p=out_softmax)
        return state, c_selected

    def _forward_batch(self, weights: Weights, batch: jax.Array) -> jax.Array:
        v = self.input.forward_batch(weights[0], batch[:, :-1], padding_len=1)

        for layer, layer_weights in zip(self.layers, weights[1:-1]):
            v = layer.forward_batch(layer_weights, v)

        out = self.output.forward_batch(weights[-1], v)
        assert out.shape == batch.shape + (self.input.ntokens,), f"batch_shape={batch.shape}, out_shape={out.shape}"
        return out

    def _loss_batch_unpacked(
        self,
        weights: Weights,
        batch: jax.Array,
    ) -> jax.Array:
        """Run model on a batch of inputs and return entropy of each character.

        Args:
            weights: model weights
            batch: an array of shape (batch_size, sample_len) with token indices

        Returns:
            an array of shape (batch_size, sample_len) with cross-entropy
            of each character
        """
        out = self._forward_batch(weights, batch)
        return optax.softmax_cross_entropy_with_integer_labels(out, batch)

    def loss_batch(self, weights: Weights, batch: jax.Array) -> jax.Array:
        """Run model on a batch of inputs and return the average entropy.

        Args:
            weights: model weights
            batch: an array of shape (batch_size, sample_len) with token indices

        Returns:
            average cross-entropy in bits per token
        """
        entropy = self._loss_batch_unpacked(weights, batch)
        return _1_BY_LOG2 * jnp.mean(entropy)

    def loss_batch_masked(
        self, weights: Weights, batch: jax.Array, lengths: jax.Array
    ) -> jax.Array:
        """Run model on a batch of inputs and return "masked" entropy.

        Calculate cross-entropy for the input batch and entropy for the
        first `lengths` tokens in each sample.

        This is useful to evaluate the model loss on the samples of fixed length
        in bytes, which can translate in varying lengths in tokens.

        Args:
            weights: model weights
            batch: an array of shape (batch_size, sample_len) with token indices
            lengths: an array of shape (batch_size,) with lengths of each sample

        Returns:
            total cross-entropy in bits
        """
        entropy = self._loss_batch_unpacked(weights, batch)
        mask = jnp.arange(entropy.shape[1]) < lengths[:, jnp.newaxis]
        return _1_BY_LOG2 * jnp.sum(entropy * mask)


_cache: dict[str, Model3] = {}


def build_model(spec: str) -> Model3:
    model = _cache.get(spec)
    if model is None:
        model = Model3(spec)
        _cache[spec] = model
    return model
