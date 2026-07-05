"""Tokenized input layer: arbitrary tokenset, one-hot or learned
embeddings.

Spec: `tokens.{ntokens}.{variation}.oh`
   or `tokens.{ntokens}.{variation}.emb.{size}`

`{ntokens}.{variation}` names the tokenset file
`tokens{ntokens}_{variation}.json` (e.g. `tokens.256000.gemma.emb.2560`
-> tokens256000_gemma.json); the sampler resolves it through the
registry via `tokens_name`. The layer itself never loads the file --
the id space is fully described by the spec.

- `.oh`: one-hot vectors of width ntokens, weight-free.
- `.emb.{size}`: a learned (ntokens, size) embedding table -- the
  first input layer with weights (Model2Jax carries them in the
  reserved slot 0 of the weights pytree). No sqrt(d) scaling here:
  reference models that scale embeddings (Gemma) get it folded into
  the table by the weight converter, since a per-lookup scalar
  commutes with the lookup.

The shift-pad initial vector is the uniform-distribution analog of the
one-hot convention: uniform 1/ntokens for `.oh`, and the mean of the
embedding rows for `.emb` (a uniform distribution over tokens, pushed
through the table). Sequence-start semantics for BOS-carrying
tokensets lives in the token stream itself (the tokenizer's add_bos),
not in the padding.

JAX only.
"""
import jax
import jax.numpy as jnp

from ..common import is_power2_int, power2_neighbors
from ..precision import Precision


class TokensInputJax:
    def __init__(self, ntokens: int, size: int, emb: bool, dtype):
        self.ntokens = ntokens
        self.size = size
        self.emb = emb
        self.dtype = dtype

    def init_weights(self, rng: jax.Array):
        if not self.emb:
            return None
        # Unit-variance rows scaled down by sqrt(size), so the initial
        # activations entering the first layer are O(1/sqrt(size)) --
        # in line with the dense layers' Xavier fan-in convention.
        scale = 1.0 / jnp.sqrt(jnp.array(self.size, dtype=self.dtype))
        return {
            'emb': jax.random.normal(
                rng, (self.ntokens, self.size), dtype=self.dtype) * scale,
        }

    def init_state(self):
        return None

    def _initial_vector(self, weights, position: int = -1) -> jax.Array:
        if self.emb:
            # Uniform distribution over tokens through the table.
            return jnp.mean(weights['emb'], axis=0)
        return jnp.full(
            (self.size,), 1.0 / self.ntokens, dtype=self.dtype)

    def step(self, weights, state, token) -> tuple[None, jax.Array]:
        if self.emb:
            return state, weights['emb'][token]
        return state, jax.nn.one_hot(
            token, self.ntokens, dtype=self.dtype)

    def forward(
        self, weights, tokens: jax.Array, padding: int = 0,
    ) -> jax.Array:
        # tokens: (batch, seq_len) int array.
        if self.emb:
            out = weights['emb'][tokens]
        else:
            out = jax.nn.one_hot(tokens, self.ntokens, dtype=self.dtype)
        if padding > 0:
            batch = tokens.shape[0]
            pad = jnp.broadcast_to(
                self._initial_vector(weights),
                (batch, padding, self.size)).astype(self.dtype)
            out = jnp.concatenate([pad, out], axis=1)
        return out


class TokensInputDef:
    """Descriptor for the tokenized input layer."""

    def __init__(
        self, ntokens: int, variation: str, emb_size: int | None,
        precision: Precision = Precision.FP32,
    ):
        self.ntokens = ntokens
        self.variation = variation
        self.emb_size = emb_size  # None => one-hot
        self.size = ntokens if emb_size is None else emb_size
        self.tokens_name = f"tokens{ntokens}_{variation}"
        self.num_weights = 0 if emb_size is None else ntokens * emb_size
        self.num_mults = 0  # table lookup / one-hot: no multiplies
        self.precision = precision

    @staticmethod
    def from_spec(
        spec: str, precision: Precision = Precision.FP32,
    ) -> 'TokensInputDef':
        parts = spec.split('.')
        assert parts[0] == 'tokens', spec
        ntokens = int(parts[1])
        variation = parts[2]
        if parts[3] == 'oh':
            assert len(parts) == 4, spec
            return TokensInputDef(ntokens, variation, None, precision)
        assert parts[3] == 'emb' and len(parts) == 5, spec
        return TokensInputDef(
            ntokens, variation, int(parts[4]), precision)

    def __str__(self):
        mode = 'oh' if self.emb_size is None else f'emb.{self.emb_size}'
        return f'tokens.{self.ntokens}.{self.variation}.{mode}'

    def is_valid(self):
        # ntokens is data-determined (whatever the tokenset has); the
        # embedding width follows the usual power-of-2 rule, so e.g.
        # tokens.256000.gemma.emb.2560 is load-only while
        # tokens.256000.gemma.emb.64 is search-valid.
        if self.ntokens < 2:
            return False
        if self.emb_size is not None:
            return is_power2_int(self.emb_size)
        return True

    def neighbors(self):
        if self.emb_size is None:
            return ()
        return tuple(
            f'tokens.{self.ntokens}.{self.variation}.emb.{s}'
            for s in power2_neighbors(self.emb_size)
        )

    def build_jax(self) -> TokensInputJax:
        return TokensInputJax(
            self.ntokens, self.size, self.emb_size is not None,
            self.precision.jax_dtype)
