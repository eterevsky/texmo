import jax
import jax.numpy as jnp
from typing import Callable, Optional

from .common import total_size, power2_neighbors
from .prng import Rng


LayerWeights = Optional[dict[str, jnp.ndarray]]
LayerState = Optional[jnp.ndarray | dict[str, jnp.ndarray]]


class Layer(object):
    name = "OVERRIDE"

    def __init__(self, input_shape: tuple[int]):
        # If step_batch() is not overridden, we'll vectorize step() and write it
        # here.
        self._step_batch = None
        self._forward_batch = None
        self._forward_batch_impl: dict[str,
            Callable[[LayerWeights, jnp.ndarray], jnp.ndarray]
        ] = {}
        self._step_batch_impl: Optional[
            Callable[
                [LayerWeights, LayerState, jnp.ndarray],
                tuple[LayerState, jnp.ndarray],
            ]
        ] = None
        self.input_shape: tuple[int] = input_shape
        self.input_size = total_size(self.input_shape)
        self.output_shape: Optional[tuple[int]] = None
        # The number of the output steps from the previous layers on which
        # this layer depends. Should be 1 for dense and various recursive
        # layers, and length of the suffix for suffix and attn layers.
        self.length: int = 1

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def output_size(self):
        return total_size(self.output_shape)

    @property
    def weights(self) -> int:
        raise NotImplementedError

    def is_valid(self) -> bool:
        raise NotImplementedError

    def neighbors(self):
        """Generator layer neighbors.

        Should be overridden by suffix and attn.
        """
        size = self.size

        for name in ("dense", "rec"):
            for activation in ("tanh", "relu", "gelu"):
                yield f"{name}.{size}.{activation}"

        for name in ("gru", "mingru", "mgru", "lstm"):
            yield f"{name}.{size}"

        for neighbor_size in power2_neighbors(size):
            if self.name in ("dense", "rec"):
                yield f"{self.name}.{neighbor_size}{self._activation_suffix}"
            else:
                yield f"{self.name}.{neighbor_size}"

    def init_weights(self, rng: Rng, init_scale: float, dtype) -> LayerWeights:
        return None

    def init_state(self, weights: LayerWeights, dtype) -> LayerState:
        return None

    def step(
        self, weights: LayerWeights, state: LayerState, input: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        """Make a single forward step on the input and return the new state and the output.

        Args:
            input: array matching layer's input_shape

        Returns:
            A pair of the new state and the output.
        """
        raise NotImplementedError

    def _step_batch_from_step(
        self, weights: LayerWeights, states: LayerState, inputs: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        if self._step_batch_impl is None:
            self._step_batch_impl = jax.vmap(
                lambda weights, states, inputs: self.step(weights, states, inputs, dtype=dtype),
                in_axes=(None, 0, 0), out_axes=0
            )

        return self._step_batch_impl(weights, states, inputs)

    def step_batch(
        self, weights: LayerWeights, states: LayerState, inputs: jnp.ndarray, dtype
    ) -> tuple[LayerState, jnp.ndarray]:
        """One character step on a batch of inputs.

        Args:
            weights: a normal set of weights
            states: an object structured as a normal state for the layer,
                but each jax.Array has an extra (first) dimension of
                batch_size
            inputs: a batch of inputs with dimensions (batch_size,) + input_shape

        Returns:
            A pair of state in the same format as input and batch output with
            the shape (batch_size,) + output_shape
        """
        return self._step_batch_from_step(weights, states, inputs, dtype=dtype)

    def _forward_from_step(
        self, weights: LayerWeights, input: jnp.ndarray, dtype
    ) -> jnp.ndarray:
        _, out = jax.lax.scan(
            lambda state, v: self.step(weights, state, v, dtype),
            self.init_state(weights, dtype=dtype),
            input,
        )

        # out = out[self.length - 1 :, :]

        return out

    def forward(self, weights: LayerWeights, input: jnp.ndarray, dtype) -> jnp.ndarray:
        """Make a forward pass on a single full sample.

        This method can be reimplemented in the layer class.

        Args:
            input: array of shape (sample_len, input_shape)

        Returns:
            The output array.
        """
        return self._forward_from_step(weights, input, dtype=dtype)

    def _forward_batch_from_forward(
        self, weights: LayerWeights, inputs: jnp.ndarray, dtype
    ) -> jnp.ndarray:
        if str(dtype) not in self._forward_batch_impl:
            _forward = lambda w, i: self.forward(w, i, dtype)
            self._forward_batch_impl[str(dtype)] = jax.vmap(
                _forward, in_axes=(None, 0), out_axes=0
            )
        return self._forward_batch_impl[str(dtype)](weights, inputs)

    def _forward_batch_from_step(
        self, weights: LayerWeights, inputs: jnp.ndarray, dtype
    ) -> jnp.ndarray:
        batch_size = inputs.shape[0]

        init_state = self.init_state(weights, dtype=dtype)
        init_state = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (batch_size,) + (1,) * len(x.shape)),
            init_state,
        )

        # Change dimensions from (batch, position, ...) to (position, batch, ...)
        inputs_swapped = jnp.swapaxes(inputs, 0, 1)

        _, out_swapped = jax.lax.scan(
            lambda state, v: self.step_batch(weights, state, v, dtype=dtype),
            init_state,
            inputs_swapped,
        )
        out = jnp.swapaxes(out_swapped, 0, 1)
        return out
        # return out[:, self.length - 1 :, :]

    # If True, forward_batch will be executed by using step_batch() recurrently.
    # If False, forward_batch will be executed by batching forward().
    use_step_batch = False

    def forward_batch(
        self, weights: LayerWeights, inputs: jax.Array, dtype
    ) -> jax.Array:
        """Make a forward pass on a batch of inputs.

        Args:
            weights: a single set of weights
            inputs: a batch of inputs with dimensions (batch_size, sample_len, input_shape)

        Returns:
            Output array with the shape (batch_size, sample_len, output_shape)
        """
        assert inputs.dtype == dtype
        if self.use_step_batch:
            return self._forward_batch_from_step(weights, inputs, dtype=dtype)
        else:
            return self._forward_batch_from_forward(weights, inputs, dtype=dtype)

    def _forward_batch_from_step_manual(self, weights: LayerWeights, inputs: jax.Array) -> jax.Array:
        outputs = []
        for sample in inputs:
            sample_out = []
            state = self.init_state(weights)
            for input in sample:
                state, out = self.step(weights, state, input)
                sample_out.append(out)
            outputs.append(sample_out)
        return jnp.array(outputs)
