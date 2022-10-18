import jax
from jax import numpy as jnp
import math
import numpy as np
from typing import Dict, Union, Tuple

from .prng import Rng


ArrayTree = Union[None, jnp.ndarray, Dict[str, "ArrayTree"]]


def is_power2(x):
    return type(x) is int and x >= 1 and x & (x - 1) == 0


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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        if self._train_init_state:
            return {"init_state": rng.normal(self._state_shape) * init_scale}
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        return {
            "w": rng.he((self._state, self.input_size)) * init_scale,
            "b": rng.normal((self._state,)) * init_scale,
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "winput": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "wstate": rng.he(
                (self._state, self._state), input_size=he_input_size
            )
            * init_scale,
            "b": rng.normal((self._state,)) * init_scale,
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,)) * init_scale
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wz": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "uz": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "bz": rng.normal((self._state,)) * init_scale,
            "wr": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "ur": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "br": rng.normal((self._state,)) * init_scale,
            "wh": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "uh": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "bh": rng.normal((self._state,)) * init_scale,
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,)) * init_scale
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wf": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "uf": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "bf": rng.normal((self._state,)) * init_scale,
            "wh": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "uh": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "bh": rng.normal((self._state,)) * init_scale,
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,)) * init_scale
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wx": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "bx": rng.normal((self._state,)) * init_scale,
            "wn": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "bn": rng.normal((self._state,)) * init_scale,
            "whz": rng.he((self._state, self._state), input_size=he_input_size)
            * init_scale,
            "wvz": rng.he(
                (self._state, self.input_size), input_size=he_input_size
            )
            * init_scale,
            "bz": rng.normal((self._state,)) * init_scale,
        }
        if self._train_init_state:
            weights["init_state"] = rng.normal((self._state,)) * init_scale
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

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        he_input_size = self.input_size + self._state
        weights = {
            "wf": rng.he((self._state, he_input_size), input_size=he_input_size)
            * init_scale,
            "bf": rng.normal((self._state,)) * init_scale,
            "wi": rng.he((self._state, he_input_size), input_size=he_input_size)
            * init_scale,
            "bi": rng.normal((self._state,)) * init_scale,
            "wo": rng.he((self._state, he_input_size), input_size=he_input_size)
            * init_scale,
            "bo": rng.normal((self._state,)) * init_scale,
            "wc": rng.he((self._state, he_input_size), input_size=he_input_size)
            * init_scale,
            "bc": rng.normal((self._state,)) * init_scale,
        }
        if self._train_init_state:
            weights["init_state"] = {
                "h": rng.normal((self._state,)) * init_scale,
                "c": rng.normal((self._state,)) * init_scale,
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

        input_h = jnp.concatenate((input, h))

        f = jnp.dot(weights["wf"], input_h) + weights["bf"]
        f = jax.nn.sigmoid(f)

        i = jnp.dot(weights["wi"], input_h) + weights["bi"]
        i = jax.nn.sigmoid(i)

        o = jnp.dot(weights["wo"], input_h) + weights["bo"]
        o = jax.nn.sigmoid(o)

        cn = jnp.dot(weights["wc"], input_h) + weights["bc"]
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
class Attention(Layer2):
    name = "attention"

    def __init__(self, length, pos=False, **kwargs):
        super().__init__(**kwargs)
        assert self._activation is None
        assert length > 1
        self._length = length
        pos_suffix = ".pos" if pos else ""
        self.full_name = f"attention.{length}{pos_suffix}"
        self.output_shape = self._input_shape
        self._state_shape = (length - 1,) + self._input_shape
        self._dk = 1 / math.sqrt(self._input_shape[0])
        self._use_pos = pos

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        if self._use_pos:
            return {
                "pe": rng.he(
                    (self._length, self.output_shape[0]),
                    self._length * self.output_shape[0],
                )
                * init_scale,
            }
        else:
            return None

    def init_state(self, weights: ArrayTree) -> ArrayTree:
        item_size = 1
        for dim in self._state_shape[1:]:
            item_size *= dim
        return jnp.ones(self._state_shape) / item_size

    def step(
        self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray
    ) -> Tuple[ArrayTree, ArrayTree]:
        suffix = jnp.vstack((state, input.reshape((1, -1))))
        next_suffix = suffix[1:]

        if self._use_pos:
            suffix += weights["pe"]

        input = suffix[-1]

        prod = jnp.matmul(suffix, jnp.transpose(input))
        weight = jax.nn.softmax(prod * self._dk)
        out = jnp.matmul(weight, suffix)

        return next_suffix, out


@layer_cls
class Attn(Layer2):
    name = "attn"

    def __init__(self, length, heads, size, **kwargs):
        """Create a multi-headed attention layer.

        Args:
            length: the length of the suffix that attention will be applied to
            heads: number of independent attention heads
            size: total output size, has to be a multiple of `heads`
        """
        super().__init__(**kwargs)
        assert type(length) is int
        assert length >= 2
        self._length = length
        assert type(heads) is int
        assert heads >= 1
        self._heads = heads
        assert type(size) is int
        assert size >= 1
        assert size % heads == 0
        self._size = size
        # The size of each key/query/value component
        self._comp_size = self._size // heads
        self.full_name = f"attn.{length}.{heads}.{size}"
        self.output_shape = (size,)
        self._score_scale = 1 / math.sqrt(self._comp_size)

    def init_weights(self, rng: Rng, init_scale: float) -> ArrayTree:
        weights = {
            "wkey": rng.he((self._heads, self._comp_size, self.input_size))
            * init_scale,
            "wquery": rng.he((self._heads, self._comp_size, self.input_size))
            * init_scale,
            "wvalue": rng.he((self._heads, self._comp_size, self.input_size))
            * init_scale,
            "bkey": rng.normal((self._length, self._heads, self._comp_size))
            * init_scale
            * 0.1,
            "bquery": rng.normal((self._heads, self._comp_size))
            * init_scale
            * 0.1,
        }
        return weights

    def init_state(self, weights: ArrayTree) -> ArrayTree:
        return {
            "keys": jnp.zeros((self._length - 1, self._heads, self._comp_size)),
            "values": jnp.zeros(
                (self._length - 1, self._heads, self._comp_size)
            ),
        }

    def step(
        self, weights: ArrayTree, state: ArrayTree, input: jnp.ndarray
    ) -> Tuple[ArrayTree, ArrayTree]:
        input = input.flatten()
        key = jnp.dot(weights["wkey"], input)
        query = jnp.dot(weights["wquery"], input) + weights["bquery"]
        value = jnp.dot(weights["wvalue"], input)

        keys = jnp.vstack((state["keys"], key.reshape((1, self._heads, -1))))
        values = jnp.vstack(
            (state["values"], value.reshape((1, self._heads, -1)))
        )

        biased_keys = keys + weights["bkey"]

        scores = self._score_scale * jnp.einsum(
            "hT,phT->hp", query, biased_keys
        )
        weights = jax.nn.softmax(scores)  # head,position -> weight

        attn_value = jnp.einsum("hp,phv->hv", weights, values)
        attn_value = attn_value.flatten()

        next_keys = keys[:-1, :, :]
        next_values = values[:-1, :, :]
        return {"keys": next_keys, "values": next_values}, attn_value
