import jax
import jax.nn as jax_nn
import jax.numpy as jnp

from ..common import is_power2_int
from ..layer import LayerDef, LayerJax
from ..layer_jax import LayerWeights, xavier_uniform

_JAX_ACTIVATIONS = {
    "relu": jax_nn.relu,
    "tanh": jnp.tanh,
    "gelu": jax_nn.gelu,
    "silu": jax_nn.silu,
}


class DenseJax(LayerJax):
    def __init__(self, input_size: int, size: int, activation_name: str | None, dtype):
        super().__init__(input_size, size, dtype)
        self.activation = _JAX_ACTIVATIONS.get(activation_name)

    def init_weights(self, rng: jax.Array) -> LayerWeights:
        return {
            'w': xavier_uniform(rng, (self.size, self.input_size), dtype=self.dtype),
            'b': jnp.zeros(self.size, dtype=self.dtype),
        }

    def _apply(self, x, weights):
        out = x @ weights['w'].T + weights['b']
        return self.activation(out) if self.activation else out

    def step(
        self, weights: LayerWeights, state, x: jax.Array
    ) -> tuple[None, jax.Array]:
        return None, self._apply(x, weights)

    def forward(
        self, weights: LayerWeights, inputs: jax.Array
    ) -> jax.Array:
        return self._apply(inputs, weights)


class DenseDef(LayerDef):
    name = "dense"

    def __init__(self, size: int, input_size: int, activation: str | None = None):
        super().__init__(input_size=input_size)
        self.size = size
        # Reject an unknown activation name here rather than silently
        # degrading to a bare linear at build_jax time (DenseJax looks
        # the name up with .get(), since None is the legal bare form).
        if activation is not None and activation not in _JAX_ACTIVATIONS:
            raise KeyError(activation)
        self._activation = activation

    def __str__(self) -> str:
        if self._activation:
            return f"dense.{self.size}.{self._activation}"
        return f"dense.{self.size}"

    def is_valid(self, *, allow_bare: bool = False) -> bool:
        # A bare dense (no activation) is normally rejected: two
        # adjacent linears collapse to one, so it adds nothing on its
        # own. Inside a Split branch it's the value/linear path of a
        # gated unit (GeGLU/SwiGLU: `split.mul(dense.X.gelu,
        # dense.X)`), so SplitDef opts into `allow_bare` for a
        # branch's terminal layer. The power-of-two size discipline
        # still applies either way. relu is retired (2026-07: almost
        # strictly worse than gelu in the DB) -- relu confs still
        # parse and run, but count as invalid like off-whitelist
        # inputs, and their neighbors propose the surviving
        # activations. silu joined the whitelist 2026-08 (it is the
        # `S` in SwiGLU: `split.mul(dense.X.silu, dense.X)`).
        if not is_power2_int(self.size):
            return False
        if self._activation is None:
            return allow_bare
        return self._activation in ("tanh", "gelu", "silu")

    @property
    def num_weights(self) -> int:
        return self.size * self.input_size + self.size

    def build_jax(self, dtype) -> DenseJax:
        return DenseJax(self.input_size, self.size, self._activation, dtype)
