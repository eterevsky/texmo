import torch
import torch.nn as nn
from torch import Tensor


# Number of bits used to encode the bit chunk position within a byte.
_BP = {1: 3, 2: 2, 4: 1, 8: 0}

_INPUT_NEIGHBORS = {
    'bits.1': ('bits.1+bp', 'bits.2', 'bits.2.oh'),
    'bits.1+bp': ('bits.1', 'bits.2+bp', 'bits.2.oh+bp'),
    'bits.2': ('bits.1', 'bits.2+bp', 'bits.2.oh', 'bits.4'),
    'bits.2+bp': ('bits.1+bp', 'bits.2', 'bits.2.oh+bp', 'bits.4+bp'),
    'bits.2.oh': ('bits.1', 'bits.2', 'bits.2.oh+bp', 'bits.4.oh'),
    'bits.2.oh+bp': ('bits.1+bp', 'bits.2+bp', 'bits.2.oh', 'bits.4.oh+bp'),
    'bits.4': ('bits.2', 'bits.4+bp', 'bits.4.oh', 'bits.8'),
    'bits.4+bp': ('bits.2+bp', 'bits.4', 'bits.4.oh+bp', 'bits.8'),
    'bits.4.oh': ('bits.2.oh', 'bits.4', 'bits.4.oh+bp', 'bytes'),
    'bits.4.oh+bp': ('bits.2.oh+bp', 'bits.4+bp', 'bits.4.oh', 'bytes'),
    'bits.8': ('bits.4', 'bits.4+bp', 'bytes'),
}


def _to_bit_array(n: int, nbits: int) -> list[float]:
    enc = []
    for i in range(nbits):
        enc.append(float(n % 2))
        n //= 2
    return enc


class InputBitsModule(nn.Module):
    """Input module that encodes bit-chunks as one-hot or raw bits, optionally
    with binary position encoding."""

    def __init__(self, nbits: int, one_hot: bool, bp: bool,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = 2 ** nbits
        self.dtype = dtype

        if one_hot:
            enc_size = self.ntokens
        else:
            enc_size = nbits
        self.size = enc_size
        if bp:
            self.size += _BP[nbits]

        # Build encoding table
        encodings = []
        for i in range(self.ntokens):
            if one_hot:
                row = [0.0] * self.ntokens
                row[i] = 1.0
            else:
                row = _to_bit_array(i, nbits)
            encodings.append(row)
        self.register_buffer('encodings', torch.tensor(encodings))

        # Position encodings for bp mode
        if bp:
            self.positions = 8 // nbits
            bp_bits = _BP[nbits]
            pos_encodings = []
            for i in range(self.positions):
                pos_encodings.append(_to_bit_array(i, bp_bits))
            self.register_buffer('pos_encodings', torch.tensor(pos_encodings))

    def init_state(self) -> int | None:
        """Return initial state: bit-chunk position within the byte, or None
        if position encoding is not used."""
        if self.bp:
            return 0
        return None

    def step(self, state, token: int,
             device: torch.device | None = None) -> tuple:
        """Encode a single token, returning (new_state, output)."""
        out = self.encodings[int(token)]

        if self.bp:
            pos = state
            pos_enc = self.pos_encodings[pos]
            out = torch.cat([out, pos_enc])
            state = (pos + 1) % self.positions

        return state, out.to(self.dtype)

    def initial_vector(self, device: torch.device | None = None,
                        position: int = -1) -> Tensor:
        """Return the input vector for 'no token observed yet'.

        Args:
            device: target device
            position: position in the byte for bp encoding (default -1,
                      i.e. the last position in the cycle)
        """
        if self.one_hot:
            # Uniform 1/n: max entropy over tokens
            v = torch.full((self.ntokens,), 1.0 / self.ntokens,
                           dtype=self.dtype, device=device)
        else:
            # 0.5 for value bits (max entropy for unknown bits)
            v = torch.full((self.nbits,), 0.5, dtype=self.dtype,
                           device=device)
        if self.bp:
            bp_enc = self.pos_encodings[position % self.positions]
            v = torch.cat([v, bp_enc.to(dtype=self.dtype, device=device)])
        return v

    def forward(self, tokens: Tensor, padding: int = 0) -> Tensor:
        """Encode a batch of token sequences.

        Args:
            tokens: (batch, seq_len) int64 token indices
            padding: number of initial positions to prepend (uninformative
                     input with correct position encoding)

        Returns:
            (batch, seq_len + padding, size) float tensor
        """
        batch_size, seq_len = tokens.shape

        out = self.encodings[tokens]  # (batch, seq_len, enc_size)

        if self.bp:
            reps = (seq_len + self.positions - 1) // self.positions
            pos = self.pos_encodings.repeat(reps, 1)[:seq_len]
            pos = pos.unsqueeze(0).expand(batch_size, -1, -1)
            out = torch.cat([out, pos], dim=-1)

        if padding > 0:
            pad_vecs = [self.initial_vector(device=tokens.device, position=-padding + i)
                        for i in range(padding)]
            pad = torch.stack(pad_vecs).unsqueeze(0).expand(batch_size, -1, -1)
            out = torch.cat([pad, out], dim=1)

        return out.to(self.dtype)


class InputBitsDef:
    """Descriptor for the bits input layer."""

    def __init__(self, nbits: int, one_hot: bool, bp: bool,
                 dtype: torch.dtype = torch.float32):
        self.nbits = nbits
        self.one_hot = one_hot
        self.bp = bp
        self.ntokens = 2 ** nbits
        self.dtype = dtype
        self.tokens_name = f'bits.{nbits}'
        self.num_weights = 0

        if one_hot:
            self.size = self.ntokens
        else:
            self.size = nbits
        if bp:
            self.size += _BP[nbits]

    @staticmethod
    def from_spec(spec: str, dtype: torch.dtype = torch.float32) -> 'InputBitsDef':
        if spec == 'bytes':
            spec = 'bits.8.oh'

        parts = spec.split('+')
        bp = False
        if len(parts) > 1:
            assert len(parts) == 2
            assert parts[1] == 'bp'
            bp = True

        main = parts[0].split('.')
        assert main[0] == 'bits'
        nbits = int(main[1])
        assert nbits in (1, 2, 4, 8)

        one_hot = len(main) > 2 and main[2] == 'oh'

        return InputBitsDef(nbits, one_hot, bp, dtype)

    def __str__(self):
        if self.nbits == 8 and self.one_hot and not self.bp:
            return 'bytes'
        s = f'bits.{self.nbits}'
        if self.one_hot:
            s += '.oh'
        if self.bp:
            s += '+bp'
        return s

    def is_valid(self):
        if self.nbits not in (1, 2, 4, 8):
            return False
        if self.bp and self.nbits >= 8:
            return False
        if self.one_hot and self.nbits == 1:
            return False
        return True

    def neighbors(self):
        spec = str(self)
        return _INPUT_NEIGHBORS.get(spec, ())

    def build_module(self) -> InputBitsModule:
        return InputBitsModule(self.nbits, self.one_hot, self.bp, self.dtype)
