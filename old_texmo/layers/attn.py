import math
import jax
import jax.numpy as jnp

from ..common import is_power2_int, power2_neighbors
from ..layer import Layer, LayerState, LayerWeights
from .registry import layer_cls


@layer_cls
class Attn(Layer):
    name = "attn"

    def __init__(self, length, heads, size, input_shape=None):
        """Create a multi-headed attention layer.

        Args:
            length: the length of the suffix that attention will be applied to
            heads: number of independent attention heads
            size: total output size, has to be a multiple of `heads`
        """
        super().__init__(input_shape=input_shape)
        assert type(length) is int
        self.length = length
        assert type(heads) is int
        assert heads >= 1
        self.heads = heads
        assert type(size) is int
        assert size >= 1
        assert size >= heads
        self.size = size

        self._comp_size = self.size // heads
        self.output_shape = (size,)
        self._state_shape = (self.length - 1, self.input_size)
        self._score_scale = 1 / math.sqrt(self.size)

    def __str__(self) -> str:
        return f"attn.{self.length}.{self.heads}.{self.size}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.length)
            and self.length > 1
            and self.heads > 1  # Otherwise it's equivalent to attnmq
            and is_power2_int(self.heads)
            and is_power2_int(self.size)
            and self.size % self.heads == 0
        )

    def neighbors(self):
        yield f"att.{self.length}.{self.heads}.{self.size}.mk"
        yield f"attnmq.{self.length}.{self.heads}.{self.size}"
        for l in power2_neighbors(self.length):
            if l >= 2:
                yield f"attn.{l}.{self.heads}.{self.size}"
        for l in power2_neighbors(self.heads):
            if self.size % l == 0 and self.size >= l:
                yield f"attn.{self.length}.{l}.{self.size}"
        for l in power2_neighbors(self.size):
            if l % self.heads == 0 and l >= self.heads:
                yield f"attn.{self.length}.{self.heads}.{l}"

    @property
    def weights(self) -> int:
        return (
            3 * self.heads * self._comp_size * self.input_size
            + self.length * self.heads * self._comp_size
            + self.heads * self._comp_size
        )

    def init_weights(self, rng, init_scale, dtype) -> None:
        return {
            "w": rng.he((self.heads, 3 * self._comp_size, self.input_size), dtype=dtype)
            * init_scale,
            "bkey": rng.normal((self.heads, self.length, self._comp_size), dtype=dtype)
            * init_scale
            * 0.1,
            "bquery": rng.normal((self.heads, self._comp_size), dtype=dtype)
            * init_scale
            * 0.1,
        }

    def init_state(self, _weights, dtype) -> LayerState:
        return {
            "keys": jnp.zeros((self.heads, self.length - 1, self._comp_size), dtype=dtype),
            "values": jnp.zeros((self.heads, self.length - 1, self._comp_size), dtype=dtype),
        }

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()

        kvq = jnp.dot(weights["w"], input)
        key = kvq[:, : self._comp_size]
        value = kvq[:, self._comp_size : 2 * self._comp_size]
        query = kvq[:, 2 * self._comp_size :] + weights["bquery"]

        keys = jnp.concatenate(
            (state["keys"], key.reshape((self.heads, 1, -1))), axis=1
        )
        values = jnp.concatenate(
            (state["values"], value.reshape((self.heads, 1, -1))), axis=1
        )

        biased_keys = keys + weights["bkey"]

        scores = self._score_scale * jnp.einsum(
            "hv,hpv->hp", query, biased_keys
        )
        weights = jax.nn.softmax(scores)  # head,position -> weight
        # softmax by default calculated by the last dim

        attn_value = jnp.einsum("hp,hpv->hv", weights, values)
        attn_value = attn_value.flatten()

        next_keys = keys[:, 1:, :]
        next_values = values[:, 1:, :]
        return {"keys": next_keys, "values": next_values}, attn_value

    def forward(self, weights: LayerWeights, input: jax.Array, dtype) -> jax.Array:
        input_len = input.shape[0]
        input = input.reshape((input_len, -1))
        input = jnp.concatenate(
            [jnp.zeros((self.length - 1, self.input_size), dtype=dtype), input]
        )
        input_len = input.shape[0]

        kvq = jnp.einsum("hoi,pi->pho", weights["w"], input)

        queries = kvq[:, :, 2 * self._comp_size :] + jnp.expand_dims(
            weights["bquery"], 0
        )
        queries = queries[self.length - 1 :]

        kv = kvq[:, :, : 2 * self._comp_size]  # position, head, value
        kv_slices = []
        for offset in range(self.length):
            kv_slices.append(
                kv[offset : input_len - self.length + offset + 1, :, :]
            )
        kv_suffixes = jnp.stack(
            kv_slices, axis=2
        )  # position, head, relative position, value

        keys = kv_suffixes[:, :, :, : self._comp_size] + jnp.expand_dims(
            weights["bkey"], 0
        )
        values = kv_suffixes[:, :, :, self._comp_size :]

        scores = self._score_scale * jnp.einsum("phv,phrv->phr", queries, keys)
        weights = jax.nn.softmax(
            scores, axis=2
        )  # position,head,relative position -> weight

        attn_value = jnp.einsum("phr,phrv->phv", weights, values)
        assert attn_value.shape[0] == input.shape[0] - self.length + 1
        return attn_value.reshape((attn_value.shape[0], -1))



@layer_cls
class AttnMQ(Layer):
    name = "attnmq"

    def __init__(self, length, heads, size, input_shape=None):
        """Create a multi-query attention layer.

        Args:
            length: the length of the suffix that attention will be applied to
            heads: number of independent attention heads
            size: total output size, has to be a multiple of `heads`
        """
        super().__init__(input_shape=input_shape)
        assert type(length) is int
        self.length = length
        assert type(heads) is int
        assert heads >= 1
        self.heads = heads
        assert type(size) is int
        assert size >= 1
        assert size >= heads
        self.size = size

        self._comp_size = self.size // heads
        self.output_shape = (size,)
        self._state_shape = (self.length - 1, self.input_size)
        self._score_scale = 1 / math.sqrt(self.size)

    def __str__(self) -> str:
        return f"attnmq.{self.length}.{self.heads}.{self.size}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.length)
            and self.length > 1
            and is_power2_int(self.heads)
            and is_power2_int(self.size)
            and self.size % self.heads == 0
        )

    def neighbors(self):
        yield f"attn.{self.length}.{self.heads}.{self.size}"
        yield f"att.{self.length}.{self.heads}.{self.size}"
        for l in power2_neighbors(self.length):
            if l >= 2:
                yield f"attnmq.{l}.{self.heads}.{self.size}"
        for l in power2_neighbors(self.heads):
            if self.size % l == 0 and self.size >= l:
                yield f"attnmq.{self.length}.{l}.{self.size}"
        for l in power2_neighbors(self.size):
            if l % self.heads == 0 and l >= self.heads:
                yield f"attnmq.{self.length}.{self.heads}.{l}"

    @property
    def weights(self) -> int:
        return (
            2 * self._comp_size * self.input_size
            + self.heads * self._comp_size * self.input_size
            + self.length * self._comp_size
            + self.heads * self._comp_size
        )

    def init_weights(self, rng, init_scale, dtype) -> None:
        return {
            "wk": rng.he((self._comp_size, self.input_size), dtype=dtype) * init_scale,
            "wv": rng.he((self._comp_size, self.input_size), dtype=dtype) * init_scale,
            "wq": rng.he((self.heads, self._comp_size, self.input_size), dtype=dtype)
            * init_scale,
            "bkey": rng.normal((self.length, self._comp_size), dtype=dtype)
            * init_scale
            * 0.1,
            "bquery": rng.normal((self.heads, self._comp_size), dtype=dtype)
            * init_scale
            * 0.1,
        }

    def init_state(self, _weights, dtype) -> LayerState:
        return {
            "keys": jnp.zeros((self.length - 1, self._comp_size), dtype=dtype),
            "values": jnp.zeros((self.length - 1, self._comp_size), dtype=dtype),
        }

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, step
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()

        #kvq = jnp.dot(weights["w"], input)
        key = jnp.dot(weights["wk"], input)
        value = jnp.dot(weights["wv"], input)
        queries = jnp.dot(weights["wq"], input) + weights["bquery"]

        keys = jnp.concatenate(
            (state["keys"], key.reshape((1, -1))), axis=0
        )
        values = jnp.concatenate(
            (state["values"], value.reshape((1, -1))), axis=0
        )

        biased_key = keys + weights["bkey"]

        scores = self._score_scale * jnp.einsum(
            "hv,pv->hp", queries, biased_key
        )
        weights = jax.nn.softmax(scores)  # head,position -> weight
        # softmax by default calculated by the last dim

        attn_value = jnp.einsum("hp,pv->hv", weights, values)
        attn_value = attn_value.flatten()

        next_keys = keys[1:, :]
        next_values = values[1:, :]
        return {"keys": next_keys, "values": next_values}, attn_value

    def forward(self, weights: LayerWeights, input: jax.Array, dtype) -> jax.Array:
        input_len = input.shape[0]
        input = input.reshape((input_len, -1))
        input = jnp.concatenate(
            [jnp.zeros((self.length - 1, self.input_size), dtype=dtype), input]
        )
        input_len = input.shape[0]

        keys = jnp.einsum("oi,pi->po", weights["wk"], input)
        values = jnp.einsum("oi,pi->po", weights["wv"], input)
        queries = jnp.einsum("hoi,pi->pho", weights["wq"], input) + jnp.expand_dims(
            weights["bquery"], 0
        )
        queries = queries[self.length - 1 :]

        key_slices = []
        for offset in range(self.length):
            key_slices.append(
                keys[offset : input_len - self.length + offset + 1, :]
            )
        key_suffixes = jnp.stack(
            key_slices, axis=1
        )  # position, relative position, value

        value_slices = []
        for offset in range(self.length):
            value_slices.append(
                values[offset : input_len - self.length + offset + 1, :]
            )
        value_suffixes = jnp.stack(
            value_slices, axis=1
        )  # position, relative position, value
        # print("value_suffixes", value_suffixes.shape)

        # print("bkey", weights["bkey"].shape)
        keys = key_suffixes + weights["bkey"]
        values = value_suffixes

        # print("keys", keys.shape)
        # print("values", values.shape)

        scores = self._score_scale * jnp.einsum("phv,prv->phr", queries, keys)
        weights = jax.nn.softmax(
            scores, axis=2
        )  # position,head,relative position -> weight

        attn_value = jnp.einsum("phr,prv->phv", weights, values)
        assert attn_value.shape[0] == input.shape[0] - self.length + 1
        return attn_value.reshape((attn_value.shape[0], -1))
