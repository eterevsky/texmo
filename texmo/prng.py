from jax import numpy as jnp
import jax.random
from random import randrange
from typing import Optional, Sequence


class Rng(object):
    def __init__(self, key = None, seed: Optional[int] = None):
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        if key is None:
            key = jax.random.PRNGKey(randrange(2**32))
        self._key = key

    def gen(self):
        self._key, out = jax.random.split(self._key)
        return out

    def uniform(self, shape: Sequence[int], dtype) -> jax.Array:
        return jax.random.uniform(self.gen(), shape=shape, dtype=dtype)

    def normal(self, shape: Sequence[int], scale=1.0, dtype=jnp.float32) -> jax.Array:
        return scale * jax.random.normal(self.gen(), shape=shape, dtype=dtype)

    def he(self, shape: Sequence[int], input_size: int = 0, dtype = jnp.float32) -> jax.Array:
        """Generate random weights with He initialization.

        The weights follow normal distribution scaled by sqrt(2 / input_size),
        where input size is either taken from `input_size` argument or from
        the last dimension of `shape`.
        """
        if input_size == 0:
            input_size = shape[-1]
        scale = jnp.sqrt(2 / input_size)
        return self.normal(shape, scale, dtype)
