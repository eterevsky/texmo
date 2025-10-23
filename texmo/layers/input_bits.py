import jax
import jax.numpy as jnp

from ..layer import LayerState, LayerWeights
from ..tokens import get_tokenizer


_BP = {1: 3, 2: 2, 4: 1}


class InputBits:
    """Input layer encoding groups of bits either as one-hot or with one bit per bit.

    Possible specs:

        bits.1
        bits.1+bp
        bits.1+pos
        bits.2
        bits.2.oh
        bits.2+bp
        bits.2+pos
        bits.2.oh+bp
        bits.2.oh+pos
        bits.4
        bits.4.oh
        bits.4+bp
        bits.4+pos
        bits.4.oh+bp
        bits.4.oh+pos
        bits.8
        bits.8-oh (== bytes)
    """

    @staticmethod
    def from_spec(spec: str) -> InputBits:
        if spec == 'bytes':
            spec = 'bits.8.oh'

        parts = spec.split('+')
        if len(parts) > 1:
            assert len(parts) == 2
            assert parts[1] in ('bp', 'pos')
            pos = parts[1]
        else:
            pos = ''

        spec = parts[0]
        parts = spec.split('.')

        assert parts[0] == 'bits'
        n = int(parts[1])
        assert n in (1, 2, 4, 8)

        if len(parts) > 2:
            assert len(parts) == 3
            assert parts[2] == 'oh'
            oh = True
        else:
            oh = False

        return InputBits(n, oh, pos)

    def __init__(self, nbits: int, one_hot: bool, pos: str):
        self.nbits = nbits
        self.one_hot = one_hot
        self.pos = pos
        self.ntokens = 2**nbits
        if one_hot:
            self.output_size = 2**nbits
        else:
            self.output_size = nbits
        if pos == 'pos':
            self.output_size += 8 // nbits
        elif pos == 'bp':
            self.output_size += _BP[nbits]
        self.tokens_name = f'bits.{nbits}'
        self.output_shape = (self.output_size,)
        self.weights = 0
        self.tokenizer = get_tokenizer(self.tokens_name)
        encodings = []
        for i in range(2**nbits):
            if self.one_hot:
                encodings.append(jax.nn.one_hot(i, 2**nbits))
            else:
                enc = []
                for j in range(nbits):
                    enc.append(i % 2)
                    i //= 2
                encodings.append(enc)
        self.encodings = jnp.array(encodings)
        self.positions = 8 // self.nbits

    def __str__(self):
        if self.nbits == 8 and self.one_hot:
            return 'bytes'
        return f'bits.{self.nbits}' + ('.oh' if self.one_hot else '') + (f'+{self.pos}' if self.pos else '')

    def is_valid(self):
        return (self.nbits in (1, 2, 4, 8) and
                (not self.one_hot or self.nbits > 1) and
                (not self.pos or self.nbits < 8))

    def _neighbors(self):
        if self.nbits > 1:
            yield InputBits(self.nbits // 2, self.one_hot, self.pos)
        if self.nbits < 8:
            yield InputBits(self.nbits * 2, self.one_hot, self.pos)

        yield InputBits(self.nbits, not self.one_hot, self.pos)

        if self.pos != '':
            yield InputBits(self.nbits, self.one_hot, '')
        if self.pos != 'pos' and self.nbits < 8:
            yield InputBits(self.nbits, self.one_hot, 'pos')
        if self.pos != 'bp' and self.nbits < 8:
            yield InputBits(self.nbits, self.one_hot, 'bp')

    def neighbors(self):
        for neighbor in self._neighbors():
            if neighbor.is_valid():
                yield neighbor

    def init_weights(self, _rng, _dtype) -> LayerWeights:
        return None

    def init_state(self, weights: LayerWeights, dtype) -> LayerState:
        return {'pos': 0}

    def step(self, _weights: LayerWeights, state: LayerState, input: int, dtype) -> tuple[LayerState, jax.Array]:
        """Consume one token from the input and return new state and output.

        Args:
            weights: embedding weights
            state:
            input: token index

        Returns:
            (new state, output vector)
        """
        if self.one_hot:
            out = jax.nn.one_hot(input, 2**self.nbits)
        else:
            i = input
            out = []
            for j in range(self.nbits):
                out.append(i % 2)
                i //= 2
            out = jnp.array(out)
        pos = state['pos']

        if self.pos == 'pos':
            out = jnp.concatenate([out, jax.nn.one_hot(state['pos'], self.positions)], dtype=dtype)
        elif self.pos == 'bp':
            bits = []
            p = state['pos']
            for i in range(_BP[self.nbits]):
                bits.append(p % 2)
                p //= 2
            out = jnp.concatenate([out, jnp.array(bits)], dtype=dtype)

        return {'pos': (pos + 1) % self.positions}, out

    def forward_batch(
        self, _weights: LayerWeights, input: jax.Array, padding_len: int, dtype
    ) -> jax.Array:
        """Generate output for a batch of full inputs.

        Args:
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
        if self.one_hot:
            out = jax.nn.one_hot(input, 2**self.nbits)
        else:
            digits = []
            for i in range(self.nbits):
                d = input // 2**i % 2
                digits.append(d)
            out = jnp.stack(digits, axis=-1)
        if self.pos == 'pos':
            pos = jnp.arange(sample_len) % self.positions
            pos = jax.nn.one_hot(pos, self.positions)
            pos = jnp.broadcast_to(pos, (batch, sample_len, self.positions))
            out = jnp.concatenate([out, pos], axis=2)
        elif self.pos == 'bp':
            pos = jnp.arange(sample_len)
            if self.nbits == 1:
                pos = jnp.stack([pos % 2, pos // 2 % 2, pos // 4 % 2], axis=-1)
                dim = 3
            elif self.nbits == 2:
                pos = jnp.stack([pos % 2, pos // 2 % 2], axis=-1)
                dim = 2
            elif self.nbits == 4:
                pos = jnp.stack([pos % 2], axis=-1)
                dim = 1
            pos = jnp.broadcast_to(pos, (batch, sample_len, dim))
            out = jnp.concatenate([out, pos], axis=2)

        padding = jnp.zeros((batch, padding_len, self.output_size), dtype=dtype)
        return jnp.concatenate([padding, out], axis=1, dtype=dtype)