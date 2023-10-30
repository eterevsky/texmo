import jax
import jax.numpy as jnp
from typing import Self, Optional

from ..layer import LayerState, LayerWeights
from ..prng import Rng
from ..tokens import get_tokenizer

_PREFERRED_TOKENSETS = {
    2: "tokens2_raw_bits1",
    4: "tokens4_raw_bits2",
    8: "tokens8_raw_bits2",
    16: "tokens16_raw_bits4",
    32: "tokens32_capswords_bits4",
    64: "tokens64_capswords_bits4",
    128: "tokens128_capswords_bits4",
    256: "tokens256_capswords_bits4",
    512: "tokens512_capswords_bits4",
    1024: "tokens1024_capswords_bits4",
    2048: "tokens2048_capswords_bits4",
    4096: "tokens4096_capswords_bits4",
    8192: "tokens8192_capswords_bits4",
    16384: "tokens16384_capswords_bits4",
}


class InputLayer(object):
    """Input layer controlling tokenization, position and embedding.

    The spec consists of three parts each one of them is optional:

        <tokens spec>-<position spec>-<embedding spec>

    Tokens spec is one of:

        tokens.<number of tokens>
        bytes

    For each number of tokens there is a default tokenset that is used. `bytes`
    are raw bytes, `bits` are raw bits. If token spec is omitted, "bytes" is
    used as a default.

    Position spec is:

        pos.<number of positions>

    If this is present, a position of token in the text modulo
    <number of positions> is added to the input. The positions are absolute, but
    are wrapping after the specified number.

    Embedding spec is one of:

        emb.<dimension>[.norm]
        onehot

    `emb.X` specifies the embedding with a given number of dimensions.
    Embeddings for the token and the position are summed together. If `.norm`
    is added, the embedding is normalized to norm 1.

    `onehot` means that the input will be represented as one-hot encoding and
    one-hot inputs for the token and position will be stacked. This is the
    default if embedding is not specified.

    Examples of specs:
        tokens.256-pos.16-emb.128
        tokens.256-emb.128.norm
        bytes-onehot
    """

    @staticmethod
    def parse(spec: str) -> Self:
        # Defaults
        ntokens = None
        positions = None  # no position encoding
        emb_dim = None  # one-hot
        emb_norm = False

        for comp_spec in spec.split("-"):
            parts = comp_spec.split(".")
            match parts[0]:
                case "tokens":
                    ntokens = int(parts[1])
                case "bytes":
                    ntokens = None
                case "pos":
                    positions = int(parts[1])
                case "onehot":
                    positions = None
                case "emb":
                    emb_dim = int(parts[1])
                    if len(parts) > 2 and parts[2] == "norm":
                        emb_norm = True

        return InputLayer(ntokens, positions, emb_dim, emb_norm)

    def __init__(
        self,
        ntokens: Optional[int],  # "bytes" if None
        positions: Optional[int],
        emb_dim: Optional[int],
        emb_norm: bool,
    ):
        self._raw_bytes = ntokens is None
        token_set = (
            _PREFERRED_TOKENSETS[ntokens] if ntokens else "tokens256_raw_all"
        )
        self.tokenizer = get_tokenizer(token_set)
        self._positions = positions
        self._emb_size = emb_dim
        self._emb_norm = emb_norm

    @property
    def ntokens(self) -> int:
        return self.tokenizer.token_set

    @property
    def output_size(self) -> int:
        if self._emb_size is not None:
            return self._emb_size
        elif self._positions is not None:
            return self._positions + self.ntokens
        else:
            return self.ntokens

    @property
    def weights(self) -> int:
        if self._emb_size is None:
            return 0
        weights = self.ntokens * self._emb_size
        if self._positions is not None:
            weights += self._positions * self._emb_size
        return weights

    def neighbors(self):
        if self._raw_bytes:
            yield InputLayer(
                ntokens=256,
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        else:
            if self.ntokens == 256:
                yield InputLayer(
                    ntokens=None,
                    positions=self._positions,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
            if self.ntokens > 2:
                yield InputLayer(
                    ntokens=self.ntokens // 2,
                    positions=self._positions,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
            if self.ntokens < 16384:
                yield InputLayer(
                    ntokens=self.ntokens * 2,
                    positions=self._positions,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )

        if self._positions is None:
            yield InputLayer(
                ntokens=self.ntokens,
                positions=2,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        else:
            if self._positions == 2:
                yield InputLayer(
                    ntokens=self.ntokens,
                    positions=None,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
            if self._positions > 2:
                yield InputLayer(
                    ntokens=self.ntokens,
                    positions=self._positions // 2,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
            yield InputLayer(
                ntokens=self.ntokens,
                positions=self._positions * 2,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )

        if self._emb_size is None:
            yield InputLayer(
                ntokens=self.ntokens,
                positions=2,
                emb_dim=self.ntokens,
                emb_norm=False,
            )
            yield InputLayer(
                ntokens=self.ntokens,
                positions=2,
                emb_dim=self.ntokens,
                emb_norm=True,
            )
        else:
            if self._emb_size == self.ntokens:
                yield InputLayer(
                    ntokens=self.ntokens,
                    positions=2,
                    emb_dim=None,
                    emb_norm=False,
                )
            if self._emb_size > 1:
                yield InputLayer(
                    ntokens=self.ntokens,
                    positions=2,
                    emb_dim=self._emb_size // 2,
                    emb_norm=self._emb_norm,
                )
            yield InputLayer(
                ntokens=self.ntokens,
                positions=2,
                emb_dim=self._emb_size * 2,
                emb_norm=self._emb_norm,
            )
            yield InputLayer(
                ntokens=self.ntokens,
                positions=2,
                emb_dim=self._emb_size,
                emb_norm=not self._emb_norm,
            )

    def init_weights(self, rng: Rng) -> LayerWeights:
        if self._emb_size is None:
            return {}
        weights = {}

        token_emb = rng.uniform(shape=(self._emb_size, self.ntokens))
        token_emb = token_emb / token_emb.sum(axis=1).reshape(-1, 1)

        weights["tokens"] = token_emb

        if self._positions is not None:
            pos_emb = rng.uniform(shape=(self._emb_size, self._positions))
            pos_emb = pos_emb / pos_emb.sum(axis=1).reshape(-1, 1)
            weights["positions"] = pos_emb
        
        return weights

    def init_state(self, weights: LayerWeights) -> LayerState:
        return {"position": 0}

    def step(self, weights: LayerWeights, state: LayerState, input: int):
        """Consume one token from the input and return new state and output.
        
        Args:
            weights: embedding weights
            state:
            input: token index

        Returns:
            (new state, output vector)
        """
        pos = state["position"]
        new_state = {"position": (pos + 1) % self._positions}

        if self._emb_size:
            emb = weights["tokens"][input]
            if self._positions:
                emb += weights["positions"][pos]

            if self._emb_norm:
                emb = emb / emb.sum()

            return new_state, emb
        else:
            oh = jax.nn.one_hot(input, self.ntokens)
            if self._positions:
                pos_oh = jax.nn.one_hot(pos, self._positions)
                oh = jnp.concatenate([oh, pos_oh])
            return new_state, oh
        
    def forward_batch(self, weights: LayerWeights, input: jax.Array, prefix_len: ) -> jax.Array:
        """Generate output for a batch of full inputs.

        Args:
            weights: embedding weights
            input: a batch of inputs with dimensions (batch_size, sample_len)
                with integer components
            prefix_len: extend the sample to the left by this amount of unknown
                tokens

        Returns:
            Output as (batch_size, sample_len, output_size)
        """
