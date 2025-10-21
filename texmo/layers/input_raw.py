import jax
import jax.numpy as jnp

from ..layer import LayerState, LayerWeights
from ..tokens import get_tokenizer

"""Input layer without tokenization, just with bytes or bits."""


class InputRaw(object):

    @staticmethod
    def from_spec(spec: str) -> InputRaw:
        assert spec == 'bytes'
        return InputRaw()

    def __init__(self):
        self.ntokens = 256
        self.output_size = 256
        self.tokens_name = 'bytes'
        self.output_shape = (self.output_size,)
        self.weights = 0
        self.tokenizer = get_tokenizer('bytes')

    def __str__(self):
        return 'bytes'

    def is_valid(self):
        return True

    def neighbors(self):
        return ()

    def init_weights(self, _rng, _dtype) -> LayerWeights:
        return None

    def init_state(self, weights: LayerWeights, dtype) -> tuple[LayerState, jax.Array]:
        return {}, jnp.zeros(self.output_shape, dtype=dtype)

    def step(self, _weights: LayerWeights, _state: LayerState, input: int, dtype) -> tuple[LayerState, jax.Array]:
        """Consume one token from the input and return new state and output.

        Args:
            weights: embedding weights
            state:
            input: token index

        Returns:
            (new state, output vector)
        """
        return {}, jax.nn.one_hot(input, self.output_size, dtype=dtype)

    def forward_batch(
        self, weights: LayerWeights, input: jax.Array, padding_len: int, dtype
    ) -> jax.Array:
        """Generate output for a batch of full inputs.

        Args:
            weights: embedding weights
            input: a batch of inputs with dimensions (batch_size, sample_len)
                with integer components
            padding_len: extend the sample to the left by this amount of unknown
                tokens

        Returns:
            Output as (batch_size, sample_len + padding_len, output_size),
            where output_size
            is either emb_size, or if it is not defined, the total size of
            one-hot encoding of the token + optionally position.
        """
        batch, sample_len = input.shape
        input_oh = jax.nn.one_hot(input, self.ntokens, dtype=dtype)
        padding = jnp.zeros((batch, padding_len, self.ntokens), dtype=dtype)
        return jnp.concatenate([padding, input_oh], axis=1)