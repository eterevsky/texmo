import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import Layer, LayerWeights
from ..prng import Rng
from .registry import layer_cls


def _normalize(arr: jax.Array) -> jax.Array:
    return arr * (1 / (jnp.linalg.norm(arr) + 1E-5))


def _normalize_batch(arr: jax.Array) -> jax.Array:
    norms = jnp.linalg.norm(arr, axis=-1)
    coef = 1 / (norms + 1E-5)
    coef = coef.reshape((arr.shape[0], arr.shape[1], 1))
    return arr * coef


@layer_cls
class Latent(Layer):
    """Latent-state recursive layer.

    Inspired by https://arxiv.org/abs/2502.05171
    """
    name = 'latent'

    def __init__(self, size: int, reps: int, tanh=False, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.size = size
        self.reps = reps
        self.output_shape = (size,)

    def __str__(self) -> str:
        return f'latent.{self.size}.{self.reps}.tanh'

    def is_valid(self) -> int:
        return is_power2_int(self.size) and is_power2_int(self.reps) and self.size >= 2 and self.reps >= 2

    def neighbors(self):
        if self.reps == 2:
            yield f'dense.{self.size}.tanh'

        yield f'latent.{self.size}.{self.reps * 2}.tanh'
        if self.reps > 2:
            yield f'latent.{self.size}.{self.reps // 2}.tanh'

        yield f'latent.{self.size * 2}.{self.reps}.tanh'
        if self.size > 2:
            yield f'latent.{self.size // 2}.{self.reps}.tanh'


    @property
    def weights(self) -> int:
        return (self.input_size + self.size + 1) * self.size

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        return {
            "wi": rng.he((self.size, self.input_size), dtype=dtype) * init_scale,
            "wr": rng.he((self.size, self.size), dtype=dtype) * init_scale,
            "b": rng.normal((self.size,), dtype=dtype) * init_scale,
        }

    def step(
        self, weights: LayerWeights, _state: None, input: jax.Array, dtype
    ) -> tuple[None, jax.Array]:
        input = input.flatten()
        input = _normalize(input)

        input_contrib = weights['wi'] @ input + weights['b']

        output, _ = jax.lax.scan(
            lambda state, _: (jnp.tanh(weights['wr'] @ _normalize(state) + input_contrib), None),
            jnp.zeros((self.size,), dtype=dtype),
            length=self.reps)

        return None, output

    def forward_batch(
            self, weights: LayerWeights, input: jax.Array, dtype
    ) -> jax.Array:
        batch_size, length = input.shape[0:2]
        input = jnp.reshape(input, (batch_size, length, -1))
        input = _normalize_batch(input)

        b = jnp.reshape(weights['b'], (1, 1, -1))
        input_contrib = jnp.einsum('oi,bni->bno', weights['wi'], input) + b

        output, _ = jax.lax.scan(
            lambda state, _: (jnp.tanh(jnp.einsum('oi,bni->bno', weights['wr'], _normalize_batch(state)) + input_contrib), None),
            jnp.zeros((batch_size, length, self.size), dtype=dtype),
            length=self.reps)

        return output
