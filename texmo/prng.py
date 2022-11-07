from jax import numpy as jnp
import jax.random
import random
from typing import Sequence


class Rng(object):
    def __init__(self, key: jax.random.PRNGKey = None):
        if key is None:
            key = jax.random.PRNGKey(random.randrange(2**32))
        self._key = key

    def gen(self) -> jax.random.PRNGKey:
        self._key, out = jax.random.split(self._key)
        return out

    def normal(self, shape: Sequence[int], scale=0.1) -> jnp.ndarray:
        return scale * jax.random.normal(self.gen(), shape=shape)

    def he(self, shape: Sequence[int], input_size: int = 0) -> jnp.ndarray:
        """Generate random weights with He initialization.

        The weights follow normal distribution scaled by sqrt(2 / input_size),
        where input size is either taken from `input_size` argument or from
        the last dimension of `shape`.
        """
        if input_size == 0:
            input_size = shape[-1]
        scale = jnp.sqrt(2 / input_size)
        return self.normal(shape, scale)