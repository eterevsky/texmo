import jax
from jax import numpy as jnp
from typing import Dict, Union, Tuple

from prng import Rng


ArrayTree = Union[None, jnp.ndarray, Dict[str, "ArrayTree"]]


class Layer2(object):
    name = "OVERRIDE"

    def __init__(
        self,
        input_shape=None,
        init=False,
        sigmoid=False,
        tanh=False,
        relu=False,
    ):
        self._input_shape = input_shape
        self._train_init_state = init
        self._init_suffix = ".init" if self._train_init_state else ""
        self.full_name = "OVERRIDE"
        self.output_shape = None

        if sigmoid:
            self._activation = jax.nn.sigmoid
            self._activation_suffix = ".sigmoid"
            assert not tanh and not relu
        elif tanh:
            self._activation = jnp.tanh
            self._activation_suffix = ".tanh"
            assert not relu
        elif relu:
            self._activation = jax.nn.relu
            self._activation_suffix = ".relu"
        else:
            self._activation = None
            self._activation_suffix = ""

    @property
    def input_size(self):
        prod = 1
        for dim in self._input_shape:
            prod *= dim
        return prod

    def init_weights(self, rng: Rng) -> ArrayTree:
        return None

    def init_state(self, weights: ArrayTree) -> ArrayTree:
        return None

    def step(
        self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray
    ) -> Tuple[ArrayTree, ArrayTree]:
        """Executes the layer function.

        Returns: (new state, output)
        """
        raise NotImplementedError


registry = {}


def layer_cls(cls):
    assert cls.name not in registry
    registry[cls.name] = cls
    return cls


@layer_cls
class Suffix(Layer2):
    name = "suffix"

    def __init__(self, length, **kwargs):
        super().__init__(**kwargs)
        assert self._activation is None
        assert length > 1
        self._length = length
        self.full_name = f"suffix.{length}{self._init_suffix}"
        self.output_shape = (length,) + self._input_shape
        self._state_shape = (length - 1,) + self._input_shape

    def init_weights(self, rng: Rng) -> ArrayTree:
        if self._train_init_state:
            return {"init_state": rng.normal(self._state_shape)}
        else:
            return None

    def init_state(self, weights: ArrayTree) -> ArrayTree:
        if self._train_init_state:
            return weights["init_state"]
        item_size = 1
        for dim in self._state_shape[1:]:
            item_size *= dim
        return jnp.ones(self._state_shape) / item_size

    def step(
        self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray
    ) -> Tuple[ArrayTree, ArrayTree]:
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        return suffix[1:], suffix


@layer_cls
class Dense(Layer2):
    name = "dense"

    def __init__(self, output_size: int, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        assert not self._train_init_state
        self._state = output_size
        self.full_name = f"dense.{self._state}{self._activation_suffix}"
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        return {
            "w": rng.he((self._state, self.input_size)),
            "b": rng.normal((self._state,)),
        }

    def init_state(self, weights) -> None:
        return None

    def step(
        self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray
    ) -> Tuple[ArrayTree, ArrayTree]:
        input = input.flatten()
        out = jnp.dot(weights["w"], input) + weights["b"]
        if self._activation is not None:
            out = self._activation(out)
        return None, out


@layer_cls
class Recurrent(Layer2):
    name = "rec"

    def __init__(self, output_size: int, ss=False, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        self._state = output_size
        self._state_skip = ss
        state_skip_suffix = ".ss" if self._state_skip else ""
        self.full_name = (
            f"rec.{self._state}"
            + self._activation_suffix
            + self._init_suffix
            + state_skip_suffix
        )
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "winput": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "wstate": rng.he(
                (self._state, self._state), input_size=he_input_size
            ),
            "b": rng.normal((self._state,)),
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,))
        return weights

    def init_state(self, weights) -> ArrayTree:
        if self._train_init_state:
            return weights["init_state"]
        else:
            return jnp.zeros((self._state,))

    def step(self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray):
        input = input.flatten()
        new_state = (
            jnp.dot(weights["winput"], input)
            + jnp.dot(weights["wstate"], state)
            + weights["b"]
        )
        if self._state_skip:
            new_state += state
        if self._activation is not None:
            new_state = self._activation(new_state)
        return new_state, new_state


@layer_cls
class Gru(Layer2):
    name = "gru"

    def __init__(self, output_size: int, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        self._state = output_size
        if self._activation is None:
            self._activation = jnp.tanh
            self._activation_suffix = ".tanh"
        self.full_name = (
            f"gru.{self._state}" + self._activation_suffix + self._init_suffix
        )
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wz": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uz": rng.he((self._state, self._state), input_size=he_input_size),
            "bz": rng.normal((self._state,)),
            "wr": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "ur": rng.he((self._state, self._state), input_size=he_input_size),
            "br": rng.normal((self._state,)),
            "wh": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uh": rng.he((self._state, self._state), input_size=he_input_size),
            "bh": rng.normal((self._state,)),
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,))
        return weights

    def init_state(self, weights) -> ArrayTree:
        if self._train_init_state:
            return weights["init_state"]
        else:
            return jnp.zeros((self._state,))

    def step(self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray):
        input = input.flatten()
        z = (
            jnp.dot(weights["wz"], input)
            + jnp.dot(weights["uz"], state)
            + weights["bz"]
        )
        z = jax.nn.sigmoid(z)

        r = (
            jnp.dot(weights["wr"], input)
            + jnp.dot(weights["ur"], state)
            + weights["br"]
        )
        r = jax.nn.sigmoid(r)

        hc = (
            jnp.dot(weights["wh"], input)
            + jnp.dot(weights["uh"], r * state)
            + weights["bh"]
        )
        hc = self._activation(hc)

        state = (1 - z) * state + z * hc
        return state, state


@layer_cls
class Mgru(Layer2):
    name = "mgru"

    def __init__(self, output_size: int, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        assert self._activation is None
        self._state = output_size
        self.full_name = f"mgru.{self._state}" + self._init_suffix
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wf": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uf": rng.he((self._state, self._state), input_size=he_input_size),
            "bf": rng.normal((self._state,)),
            "wh": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uh": rng.he((self._state, self._state), input_size=he_input_size),
            "bh": rng.normal((self._state,)),
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,))
        return weights

    def init_state(self, weights) -> ArrayTree:
        if self._train_init_state:
            return weights["init_state"]
        else:
            return jnp.zeros((self._state,))

    def step(self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray):
        input = input.flatten()
        f = (
            jnp.dot(weights["wf"], input)
            + jnp.dot(weights["uf"], state)
            + weights["bf"]
        )
        f = jax.nn.sigmoid(f)

        hc = (
            jnp.dot(weights["wh"], input)
            + jnp.dot(weights["uh"], f * state)
            + weights["bh"]
        )
        hc = jnp.tanh(hc)

        state = (1 - f) * state + f * hc
        return state, state


@layer_cls
class Caru(Layer2):
    name = "caru"

    def __init__(self, output_size: int, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        assert self._activation is None
        self._state = output_size
        self.full_name = f"caru.{self._state}" + self._init_suffix
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wx": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "bx": rng.normal((self._state,)),
            "wn": rng.he((self._state, self._state), input_size=he_input_size),
            "bn": rng.normal((self._state,)),
            "whz": rng.he((self._state, self._state), input_size=he_input_size),
            "wvz": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "bz": rng.normal((self._state,)),
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,))
        return weights

    def init_state(self, weights) -> ArrayTree:
        if self._train_init_state:
            return weights["init_state"]
        else:
            return jnp.zeros((self._state,))

    def step(self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray):
        input = input.flatten()

        x = jnp.dot(weights["wx"], input) + weights["bx"]

        n = jnp.dot(weights["wn"], state) + weights["bn"] + x
        n = jnp.tanh(n)

        z = (
            jnp.dot(weights["whz"], state)
            + jnp.dot(weights["wvz"], input)
            + weights["bz"]
        )
        z = jax.nn.sigmoid(z)

        l = jax.nn.sigmoid(x) * z

        state = (1 - l) * state + l * n
        return state, state


@layer_cls
class Lstm(Layer2):
    name = "lstm"

    def __init__(self, output_size: int, **kwargs):
        super().__init__(**kwargs)
        assert output_size > 0
        assert self._activation is None
        self._state = output_size
        self.full_name = f"lstm.{self._state}{self._init_suffix}"
        self.output_shape = (output_size,)

    def init_weights(self, rng: Rng) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wf": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uf": rng.he((self._state, self._state), input_size=he_input_size),
            "bf": rng.normal((self._state,)),
            "wi": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "ui": rng.he((self._state, self._state), input_size=he_input_size),
            "bi": rng.normal((self._state,)),
            "wo": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uo": rng.he((self._state, self._state), input_size=he_input_size),
            "bo": rng.normal((self._state,)),
            "wc": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            ),
            "uc": rng.he((self._state, self._state), input_size=he_input_size),
            "bc": rng.normal((self._state,)),
        }
        if self._train_init_state:
            weights["init_state"] = {
                "h": rng.normal((self._state,)),
                "c": rng.normal((self._state,)),
            }
        return weights

    def init_state(self, weights):
        if self._train_init_state:
            return weights["init_state"]
        else:
            return {
                "h": jnp.zeros((self._state,)),
                "c": jnp.zeros((self._state,)),
            }

    def step(self, weights, state, input):
        input = input.flatten()

        h = state["h"]
        c = state["c"]

        f = (
            jnp.dot(weights["wf"], input)
            + jnp.dot(weights["uf"], h)
            + weights["bf"]
        )
        f = jax.nn.sigmoid(f)

        i = (
            jnp.dot(weights["wi"], input)
            + jnp.dot(weights["ui"], h)
            + weights["bi"]
        )
        i = jax.nn.sigmoid(i)

        o = (
            jnp.dot(weights["wo"], input)
            + jnp.dot(weights["uo"], h)
            + weights["bo"]
        )
        o = jax.nn.sigmoid(o)

        cn = (
            jnp.dot(weights["wc"], input)
            + jnp.dot(weights["uc"], h)
            + weights["bc"]
        )
        cn = jax.nn.tanh(cn)

        c = f * c + i * cn
        h = o * c

        return {"h": h, "c": c}, h


@layer_cls
class Normalize(Layer2):
    name = "norm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self._activation is None
        self.full_name = "norm"
        self.output_shape = self._input_shape

    def init_weights(self, rng: Rng):
        return None

    def init_state(self, weights):
        return None

    def step(self, weights, state, input):
        mean = jnp.mean(input)
        stddev = jnp.std(input) + 1e-6

        inv = 1 / stddev

        return None, (input - mean) * inv


@layer_cls
class Convolution(Layer2):
    name = "conv"

    def __init__(self, kernel_size, output_size, **kwargs):
        """Transforms [X, input_size] into [X - kernel_size + 1, output_size]"""
        super().__init__(**kwargs)
        self._kernel = kernel_size
        self._output = output_size
        assert len(self._input_shape) == 2
        assert self._input_shape[0] >= self._kernel
        self.full_name = (
            f"conv.{self._output}.{self._kernel}{self._activation_suffix}"
        )
        self.output_shape = (
            self._input_shape[0] - self._kernel + 1,
            output_size,
        )

    def init_weights(self, rng: Rng):
        return {
            "kernel": rng.he(
                (self._kernel, self._output, self._input_shape[1]),
                self.input_size,
            ),
            "b": rng.normal((self._output,)),
        }

    def init_state(self, weights):
        return None

    def step(self, weights, state, input):
        kernel = weights["kernel"]
        input = jnp.expand_dims(input, axis=0)
        dn = jax.lax.conv_dimension_numbers(
            input.shape, kernel.shape, ("NWC", "WOI", "NWC")
        )
        out = jax.lax.conv_general_dilated(
            input, kernel, (1,), "VALID", (1,), (1,), dn
        )
        out = jnp.squeeze(out)
        out += jnp.expand_dims(weights["b"], axis=0)

        out = self._activation(out)

        return state, out
