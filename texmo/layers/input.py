import jax
import jax.numpy as jnp
from typing import Self

from ..layer import LayerState, LayerWeights
from ..prng import Rng
from ..tokens import get_tokenizer


_PROCESSING_NAMES = {"raw": "raw", "cw": "capswords"}
_TOKEN_TYPE_NAMES = {
    "b1": "bits1",
    "b2": "bits2",
    "b4": "bits4",
    "all": "all",
    "bh": "byteshuff",
}
_TOKEN_TYPE_NEIGHBORS = {
    "b1": ["b2"],
    "b2": ["b1", "b4", "bh"],
    "b4": ["b2", "all"],
    "all": ["b4", "bh"],
    "bh": ["b2", "all"],
}
_DEFAULT_EMB_SIZE = {
    2: 1,
    4: 2,
    8: 4,
    16: 4,
    32: 8,
    64: 8,
    128: 16,
    256: 16,
    512: 32,
    1024: 32,
    2048: 64,
    4096: 64,
    8192: 128,
    16384: 128,
    32768: 256,
    65536: 256,
}


class Input(object):
    """Input layer controlling tokenization, position and embedding.

    The spec consists of three parts each one of them is optional:

        <tokens spec>-<position spec>-<embedding spec>

    Tokens spec is one of:

        tokens.<number of tokens>.<processing type>.<token type>

    <processing type> is one of "raw" and "cw" (caps words). <token type> is one
    of "b1" (bits1), "b2" (bits2), "b4" (bits4), "all" , "bh" (byteshuff).

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
        tokens.256.raw.all-pos.16-emb.128
        tokens.256.cw.bh-emb.128.norm
        bytes-onehot
    """

    @staticmethod
    def from_spec(spec: str) -> Self:
        # Defaults
        ntokens = None
        nbits = None
        positions = None  # no position encoding
        emb_dim = None  # one-hot
        emb_norm = False
        token_type = None

        for comp_spec in spec.split("-"):
            parts = comp_spec.split(".")
            match parts[0]:
                case "tokens":
                    ntokens = int(parts[1])
                    processing = parts[2]
                    assert processing in ("raw", "cw")
                    token_type = parts[3]
                    assert token_type in ("b1", "b2", "b4", "all", "bh")
                case "pos":
                    positions = int(parts[1])
                case "onehot":
                    emb_dim = None
                case "emb":
                    emb_dim = int(parts[1])
                    if len(parts) > 2 and parts[2] == "norm":
                        emb_norm = True
                case _:
                    raise ValueError(f"Unknown component spec: {comp_spec}")

        assert (
            ntokens is None
            and nbits is not None
            or ntokens is not None
            and nbits is None
        )

        return Input(
            ntokens=ntokens,
            processing=processing,
            token_type=token_type,
            positions=positions,
            emb_dim=emb_dim,
            emb_norm=emb_norm,
        )

    def __init__(
        self,
        ntokens: int,
        processing: str,
        token_type: str,
        positions: int | None,
        emb_dim: int | None,
        emb_norm: bool,
    ):
        assert isinstance(token_type, str)
        self.ntokens = ntokens
        self._processing = processing
        self._token_type = token_type

        tokenset_name = f"tokens{ntokens}_{_PROCESSING_NAMES[processing]}_{_TOKEN_TYPE_NAMES[token_type]}"
        self.tokenizer = get_tokenizer(tokenset_name)
        self._positions = positions
        self._emb_size = emb_dim
        self._emb_norm = emb_norm

    @property
    def output_size(self) -> int:
        if self._emb_size is not None:
            return self._emb_size
        elif self._positions is not None:
            return self._positions + self.ntokens
        else:
            return self.ntokens

    @property
    def output_shape(self) -> tuple[int]:
        return (self.output_size,)

    @property
    def weights(self) -> int:
        if self._emb_size is None:
            return 0
        # Token # self.ntokens stands for unknown tokens before the beginning
        # of the sample.
        weights = (self.ntokens + 1) * self._emb_size
        if self._positions is not None:
            weights += self._positions * self._emb_size
        return weights

    def is_valid(self) -> bool:
        return self.tokenizer is not None

    def __str__(self):
        parts = [f"tokens.{self.ntokens}.{self._processing}.{self._token_type}"]

        if self._positions:
            parts.append(f"pos.{self._positions}")

        if self._emb_size:
            norm = ".norm" if self._emb_norm else ""
            parts.append(f"emb.{self._emb_size}{norm}")
        return "-".join(parts)

    def neighbors(self):
        for ntokens in (self.ntokens // 2, self.ntokens * 2):
            yield Input(
                ntokens=ntokens,
                processing=self._processing,
                token_type=self._token_type,
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        
        if (self.ntokens, self._token_type) == (2, "b1"):
            yield Input(
                ntokens=4,
                processing=self._processing,
                token_type="b2",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        elif (self.ntokens, self._token_type) == (4, "b2"):
            yield Input(
                ntokens=2,
                processing=self._processing,
                token_type="b1",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
            yield Input(
                ntokens=16,
                processing=self._processing,
                token_type="b4",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        elif (self.ntokens, self._token_type) == (16, "b4"):
            yield Input(
                ntokens=4,
                processing=self._processing,
                token_type="b2",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
            yield Input(
                ntokens=256,
                processing=self._processing,
                token_type="all",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )
        elif (self.ntokens, self._token_type) == (256, "all"):
            yield Input(
                ntokens=16,
                processing=self._processing,
                token_type="b4",
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )

        for processing in ("raw", "cw"):
            if processing != self._processing:
                yield Input(
                    ntokens=self.ntokens,
                    processing=processing,
                    token_type=self._token_type,
                    positions=self._positions,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
        
        for token_type in _TOKEN_TYPE_NEIGHBORS[self._token_type]:
            yield Input(
                ntokens=self.ntokens,
                processing=self._processing,
                token_type=token_type,
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )

        if self._positions is None:
            yield Input(
                ntokens=self.ntokens,
                processing=self._processing,
                token_type=self._token_type,
                positions=2,
                emb_dim=self._emb_size,
                emb_norm=self._emb_norm,
            )            
        else:
            if self._positions == 2:
                yield Input(
                    ntokens=self.ntokens,
                    processing=self._processing,
                    token_type=self._token_type,
                    positions=None,
                    emb_dim=self._emb_size,
                    emb_norm=self._emb_norm,
                )
            for positions in (self._positions // 2, self._positions * 2):
                if positions > 1:
                    yield Input(
                        ntokens=self.ntokens,
                        processing=self._processing,
                        token_type=self._token_type,
                        positions=positions,
                        emb_dim=self._emb_size,
                        emb_norm=self._emb_norm,
                    )

        default_emb_size = _DEFAULT_EMB_SIZE[self.ntokens]
        if self._emb_size is None:
            yield Input(
                ntokens=self.ntokens,
                processing=self._processing,
                token_type=self._token_type,
                positions=self._positions,
                emb_dim=default_emb_size,
                emb_norm=self._emb_norm,
            )
        else:
            if self._emb_size == default_emb_size:
                yield Input(
                    ntokens=self.ntokens,
                    processing=self._processing,
                    token_type=self._token_type,
                    positions=self._positions,
                    emb_dim=None,
                    emb_norm=self._emb_norm,
                )
            for emb_size in (self._emb_size // 2, self._emb_size * 2):
                if emb_size > 0:
                    yield Input(
                        ntokens=self.ntokens,
                        processing=self._processing,
                        token_type=self._token_type,
                        positions=self._positions,
                        emb_dim=emb_size,
                        emb_norm=self._emb_norm,
                    )
        if self._emb_size is not None:
            yield Input(
                ntokens=self.ntokens,
                processing=self._processing,
                token_type=self._token_type,
                positions=self._positions,
                emb_dim=self._emb_size,
                emb_norm=not self._emb_norm,
            )

    def init_weights(self, rng: Rng) -> LayerWeights:
        if self._emb_size is None:
            return {}
        weights = {}

        token_emb = rng.uniform(shape=(self.ntokens + 1, self._emb_size))
        token_emb = token_emb / token_emb.sum(axis=1).reshape(-1, 1)

        weights["tokens"] = token_emb

        if self._positions is not None:
            pos_emb = rng.uniform(shape=(self._positions, self._emb_size))
            pos_emb = pos_emb / pos_emb.sum(axis=1).reshape(-1, 1)
            weights["positions"] = pos_emb

        return weights

    def init_state(self, weights: LayerWeights) -> LayerState:
        if self._positions:
            return {"position": 0}
        else:
            return {}

    def initial_step(self, weights: LayerWeights) -> tuple[LayerState, jax.Array]:
        """Return initial state and output for position 0.

        Args:
            weights: embedding weights

        Returns:
            (initial state, output vector)
        """
        if self._emb_size:
            if self._positions:
                return {"position": 0}, weights["positions"][0]
            else:
                return {}, jnp.zeros(self._emb_size)
        else:
            if self._positions:
                return {"position": 0}, jax.nn.one_hot(
                    self.ntokens, self._positions + self.ntokens
                )
            else:
                return {}, jnp.zeros(self.ntokens)

    def step(
        self, weights: LayerWeights, state: LayerState, input: int
    ) -> tuple[LayerState, jax.Array]:
        """Consume one token from the input and return new state and output.

        Args:
            weights: embedding weights
            state:
            input: token index

        Returns:
            (new state, output vector)
        """
        if self._positions:
            pos = (state["position"] + 1) % self._positions
            new_state = {"position": pos}
        else:
            new_state = {}

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

    def forward_batch(
        self, weights: LayerWeights, input: jax.Array, padding_len: int
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

        if self._emb_size:
            padding = jnp.full(
                (batch, padding_len), fill_value=self.ntokens, dtype=jnp.int32
            )
            input = jnp.concatenate([padding, input], axis=1)
            emb = weights["tokens"][input]
            if self._positions:
                positions = jnp.arange(-padding_len, sample_len) % self._positions
                pos_emb = weights["positions"][positions]
                emb += pos_emb
            if self._emb_norm:
                emb = emb / emb.sum(axis=2).reshape(batch, sample_len + padding_len, 1)
            return emb

        input_oh = jax.nn.one_hot(input, self.ntokens)
        padding = jnp.zeros((batch, padding_len, self.ntokens))
        tokens_oh = jnp.concatenate([padding, input_oh], axis=1)
        if self._positions:
            pos = jnp.arange(-padding_len, sample_len) % self._positions
            pos_oh = jax.nn.one_hot(
                pos,
                self._positions,
            ).reshape(1, -1, self._positions)
            pos_oh = jnp.tile(pos_oh, (batch, 1, 1))
            tokens_oh = jnp.concatenate([tokens_oh, pos_oh], axis=2)

        return tokens_oh
