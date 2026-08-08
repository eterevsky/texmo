import jax
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights


class SuffixJax(LayerJax):
    """Sliding window over the last `length` inputs, flattened."""

    def __init__(self, input_size: int, length: int, dtype):
        super().__init__(input_size=input_size, size=length * input_size,
                         dtype=dtype)
        self.length = length

    def init_state(self):
        return jnp.zeros((self.length - 1, self.input_size), dtype=self.dtype)

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # state: (length-1, input_size), x: (input_size,)
        window = jnp.concatenate([state, x[jnp.newaxis, :]], axis=0)
        new_state = window[1:]
        out = window.reshape(-1)
        return new_state, out

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        # inputs: (batch, seq_len, input_size)
        # output: (batch, seq_len - length + 1, length * input_size)
        seq_len = inputs.shape[1]
        out_len = seq_len - self.length + 1
        slices = [inputs[:, offset:offset + out_len]
                  for offset in range(self.length)]
        return jnp.concatenate(slices, axis=-1)


class SuffixDef(LayerDef):
    name = "suffix"

    def __init__(self, length: int, input_size: int):
        super().__init__(input_size=input_size)
        self.length = length
        self.size = length * input_size

    def __str__(self) -> str:
        return f"suffix.{self.length}"

    def is_valid(self) -> bool:
        return is_power2_int(self.length) and self.length > 1

    @property
    def num_weights(self) -> int:
        return 0

    @property
    def projects_input(self) -> bool:
        return False  # weightless window stacking, no matmul

    def build_jax(self, dtype) -> SuffixJax:
        return SuffixJax(self.input_size, self.length, dtype)
