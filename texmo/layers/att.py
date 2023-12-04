from math import sqrt

import jax
import jax.numpy as jnp

from ..common import is_power2_int, power2_neighbors
from ..layer import Layer, LayerState, LayerWeights
from .registry import layer_cls

@layer_cls
class Attention(Layer):
    name = "att"

    def __init__(self, length: int, heads: int, size: int, mk: bool = False, input_shape=None):
        """Create a multi-headed attention layer.

        Args:
            length: the length of the suffix that attention will be applied to
            heads: number of independent attention heads
            size: total output size, has to be a multiple of `heads`
            mk: if True, each attention head will have its own keys, otherwise
                (default) the keys will all be the same, only queries will
                differ
        """
        super().__init__(input_shape=input_shape)
        assert type(length) is int
        self.length = length
        assert type(heads) is int
        assert heads >= 1
        self.heads = heads
        assert type(size) is int
        assert size >= 1
        assert size % heads == 0
        self.size = size

        self.multi_key: bool = mk

        self._comp_size = self.size // heads
        self.output_shape = (size,)
        self._score_scale = 1 / sqrt(self.size)
        self._masks = {}

    def __str__(self) -> str:
        mk = ".mk" if self.multi_key else ""
        return f"att.{self.length}.{self.heads}.{self.size}{mk}"

    def is_valid(self) -> bool:
        return (
            is_power2_int(self.length)
            and self.length > 1
            and is_power2_int(self.heads)
            and is_power2_int(self.size)
            and self.size % self.heads == 0
        )

    def neighbors(self):
        if self.heads == 4 and self.size == self.input_size:
            yield f"suffix.{self.length}"
        if self.multi_key:
            yield f"att.{self.length}.{self.heads}.{self.size}"
        else:
            yield f"att.{self.length}.{self.heads}.{self.size}.mk"
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
        if self.multi_key:
            return 3 * self.heads * self._comp_size * self.input_size
        else:
            return 2 * self.heads * self._comp_size * self.input_size + self._comp_size * self.input_size

    def init_weights(self, rng, init_scale) -> None:
        weights = {
            "wvalue": rng.he((self.heads, self._comp_size, self.input_size)) * init_scale,
            "wquery": rng.he((self.heads, self._comp_size, self.input_size)) * init_scale,
        }

        if self.multi_key:
            weights["wkey"] = rng.he((self.heads, self._comp_size, self.input_size)) * init_scale,
        else:
            weights["wkey"] = rng.he((self._comp_size, self.input_size)) * init_scale,

    def init_state(self, _weights) -> LayerState:
        return {
            "keys": jnp.zeros((self.heads, self.length - 1, self._comp_size)),
            "values": jnp.zeros((self.heads, self.length - 1, self._comp_size)),
        }

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray
    ) -> tuple[LayerState, jax.Array]:
        input = input.flatten()

        key = jnp.dot(weights["wkey"], input)
        value = jnp.dot(weights["wvalue"], input)
        query = jnp.dot(weights["wquery"], input)

        values = jnp.concatenate(
            (state["values"], value.reshape((self.heads, 1, -1))), axis=1
        )

        if self.multi_key:
            keys = jnp.concatenate(
                (state["keys"], key.reshape((self.heads, 1, -1))), axis=1
            )
            # h = head, v = input vector dimension, p = position
            scores = self._score_scale * jnp.einsum(
                "hv,hpv->hp", query, keys
            )
        else:
            keys = jnp.concatenate(
                (state["keys"], key.reshape((1, -1))), axis=0
            )
            scores = self._score_scale * jnp.einsum(
                "v,hpv->hp", query, keys
            )

        weight = jax.nn.softmax(scores)  # head,position -> weight
        # softmax by default is calculated by the last dim

        attn_value = jnp.einsum("hp,hpv->hv", weight, values)
        attn_value = attn_value.flatten()

        next_keys = keys[:, 1:, :]
        next_values = values[:, 1:, :]
        return {"keys": next_keys, "values": next_values}, attn_value

    def _get_mask(self, input_length: int) -> jax.Array:
        mask = self._masks.get(input_length)
        if mask is not None:
            return mask
        mask = []
        for p in range(input_length):
            row = []
            mask.append(row)
            for q in range(input_length):
                row.append(q <= p and p - q < self.length)
        mask = jnp.array(mask, dtype=jnp.bool_)
        self._masks[input_length] = mask
        return mask

    def forward_batch(self, weights: LayerWeights, input: jax.Array) -> jax.Array:
        """
        Args:
            weights: a single set of weights
            inputs: a batch of inputs with dimensions (batch_size, sample_len, input_shape)

        Returns:
            a batch with dimensions (batch_size, sample_len, output_shape)
        """
        mask = self._get_mask(input.shape[1])

        values = jnp.einsum("hoi,bpi->bhpo", weights["wvalue"], input)
        queries = jnp.einsum("hoi,bpi->bhpo", weights["wquery"], input)

        if self.multi_key:
            keys = jnp.einsum("hoi,bpi->bhpo", weights["wkey"], input)
            scores = jnp.einsum("bhpv,bhqv,pq->bhpq", queries, keys, mask)
        else:
            keys = jnp.einsum("oi,bpi->bpo", weights["key"], input)
            scores = jnp.einsum("bpv,bhqv,pq->bhpq", queries, keys, mask)

        weight = jax.nn.softmax(scores)
        attn_value = jnp.einsum("bhpq,bhqv->bphv", weight, values)

        return attn_value.reshape(input.shape[0], input.shape[1], -1)


