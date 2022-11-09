import math
import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray

from ..common import total_size
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
        assert length >= 2
        self.length = length
        assert type(heads) is int
        assert heads >= 1
        self.heads = heads
        assert type(size) is int
        assert size >= 1
        assert size % heads == 0
        self.size = size

        self._comp_size = self.size // heads
        self.output_shape = (size,)
        self._state_shape = (self.length - 1, self.input_size)
        self._score_scale = 1 / math.sqrt(self._comp_size)

    def __str__(self) -> str:
        return f"attn.{self.length}.{self.heads}.{self.size}"

    @property
    def weights(self) -> int:
        return (
            3 * self.heads * self._comp_size * self.input_size
            + self.length * self.heads * self._comp_size
            + self.heads * self._comp_size
        )

    def init_weights(self, rng, init_scale) -> None:
        return {
            "w": rng.he((self.heads, 3 * self._comp_size, self.input_size))
            * init_scale,
            "bkey": rng.normal((self.heads, self.length, self._comp_size))
            * init_scale * 0.1,
            "bquery": rng.normal((self.heads, self._comp_size))
            * init_scale * 0.1,
        }

    def init_state(self, _weights) -> LayerState:
        return {
            "keys": jnp.zeros((self.heads, self.length - 1, self._comp_size)),
            "values": jnp.zeros(
                (self.heads, self.length - 1, self._comp_size)
            ),
        }

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray
    ) -> tuple[LayerState, DeviceArray]:
        input = input.flatten()

        kqv = jnp.dot(weights["w"], input)
        key = kqv[:,:self._comp_size]
        query = kqv[:,self._comp_size:2*self._comp_size] + weights["bquery"]
        value = kqv[:,2*self._comp_size:]

        keys = jnp.concatenate((state["keys"], key.reshape((self.heads, 1, -1))), axis=1)
        values = jnp.concatenate(
            (state["values"], value.reshape((self.heads, 1, -1))), axis=1
        )

        biased_keys = keys + weights["bkey"]

        scores = self._score_scale * jnp.einsum(
            "hT,hpT->hp", query, biased_keys
        )
        weights = jax.nn.softmax(scores)  # head,position -> weight
                                          # softmax by default calculated by the last dim

        attn_value = jnp.einsum("hp,hpv->hv", weights, values)
        attn_value = attn_value.flatten()

        next_keys = keys[:, :-1, :]
        next_values = values[:, :-1, :]
        return {"keys": next_keys, "values": next_values}, attn_value

    def forward(self, weights: LayerWeights, input: DeviceArray) -> DeviceArray:
        input_len = input.shape[0]
        input = input.reshape((input_len, -1))

        kvq = jnp.einsum("hoi,pi->pho", weights["w"], input)

        queries = kvq[:,:,2*self._comp_size:] + jnp.expand_dims(weights["bquery"], 0)
        queries = queries[self.length - 1:]

        kv = kvq[:,:,:2*self._comp_size]  # position, head, value
        kv_slices = []
        for offset in range(self.length):
            kv_slices.append(kv[offset:input_len - self.length + offset + 1,:,:])
        kv_suffixes = jnp.stack(kv_slices, axis=2)  # position, head, relative position, value

        keys = kv_suffixes[:,:,:,:self._comp_size] + jnp.expand_dims(weights["bkey"], 0)
        values = kv_suffixes[:,:,:,self._comp_size:]

        scores = jnp.einsum("phrv,phv->phr", keys, queries)
        weights = jax.nn.softmax(scores)  # position,head,relative position -> weight

        attn_value = jnp.einsum("phr,phrv->phv", weights, values)
        assert attn_value.shape[0] == input.shape[0] - self.length + 1
        return attn_value.reshape((attn_value.shape[0], -1))
