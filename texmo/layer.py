from .common import total_size
from .prng import Rng

import jax
import jax.numpy as jnp
from jax.numpy import DeviceArray


LayerWeights = None | dict[DeviceArray]
LayerState = None | DeviceArray | dict[DeviceArray]


class Layer(object):
    def __init__(self, input_shape: tuple[int]):
        # If step_batch() is not overridden, we'll vectorize step() and write it
        # here.
        self._step_batch = None
        self._forward_batch = None
        self.input_shape: tuple[int] = input_shape
        self.input_size = total_size(self.input_shape)
        self.output_shape: tuple[int] = None

    @property
    def weights(self) -> int:
        raise NotImplementedError

    def init_weights(self, rng: Rng, init_scale: float = 1.0) -> LayerWeights:
        return None

    def init_state(self, weights: LayerWeights) -> LayerState:
        return None

    def step(
        self, weights: LayerWeights, state: LayerState, input: DeviceArray
    ) -> tuple[LayerState, DeviceArray]:
        """Make a single forward step on the input and return the new state and the output.

        Args:
            input: array matching layer's input_shape

        Returns:
            A pair of the new state and the output.
        """
        raise NotImplementedError

    def step_batch(
        self, weights: LayerWeights, states: LayerState, inputs: DeviceArray
    ) -> tuple[LayerState, DeviceArray]:
        """One character step on a batch of inputs.

        Args:
            weights: a normal set of weights
            states: an object structured as a normal state for the layer,
                but each DeviceArray has an extra (first) dimension of
                batch_size
            inputs: a batch of inputs with dimensions (batch_size,) + input_shape

        Returns:
            A pair of state in the same format as input and batch output with
            the shape (batch_size,) + output_shape
        """
        if self._step_batch is None:
            self._step_batch = jax.vmap(self.step, in_axes=(None, 0, 0), out_axes=0)
        return self._step_batch(weights, states, inputs)

    def forward(self, weights: LayerWeights, input: DeviceArray) -> DeviceArray:
        """Make a forward pass on a single full sample.

        This method can be reimplemented in the layer class.

        Args:
            input: array of shape (sample_len, input_shape)

        Returns:
            The output array.
        """
        _, out = jax.lax.scan(
            lambda state, v: self.step(weights, state, v),
            self.init_state(weights),
            input
        )

        return out

    # If True, forward_batch will be executed by using step_batch() recurrently.
    # If False, forward_batch will be executed by batching forward().
    use_step_batch = False

    def forward_batch(self, weights: LayerWeights, inputs: DeviceArray) -> DeviceArray:
        """Make a forward pass on a batch of inputs.

        Args:
            weights: a single set of weights
            inputs: a batch of inputs with dimensions (batch_size, sample_len, input_shape)

        Returns:
            Output array with the shape (batch_size, sample_len, output_shape)
        """
        batch_size = inputs.shape[0]

        if self.use_step_batch:
            init_state = self.init_state(weights)
            init_state = jax.tree_util.tree_map(
                lambda x: jnp.tile(x, (batch_size,) + (1,) * len(x.shape)),
                init_state
            )
            _, out = jax.lax.scan(
                lambda state, v: self.step_batch(weights, state, v),
                init_state,
                inputs,
            )
            return out
        else:
            if self._forward_batch is None:
                self._forward_batch = jax.vmap(self.forward, in_axes=(None, 0), out_axes=0)
            return self._forward_batch(weights, inputs)

