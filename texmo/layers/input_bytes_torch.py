import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InputBytesModule(nn.Module):
    """Input module that converts byte token indices to one-hot vectors."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ntokens = 256
        self.output_size = 256
        self.dtype = dtype

    def init_state(self):
        return None

    def step(self, state, token: int, device: torch.device | None = None) -> tuple[None, Tensor]:
        """Convert a single token index to a one-hot vector.

        Args:
            state: unused, always None
            token: integer token index (0-255)
            device: target device for the output tensor

        Returns:
            (state, (256,) one-hot tensor)
        """
        return state, F.one_hot(torch.tensor(int(token), dtype=torch.long, device=device), self.ntokens).to(self.dtype)

    def initial_vector(self, device: torch.device | None = None) -> Tensor:
        """Return the input vector for 'no token observed yet'."""
        return torch.full((self.output_size,), 1.0 / self.ntokens,
                          dtype=self.dtype, device=device)

    def forward(self, tokens: Tensor, padding: int = 0) -> Tensor:
        """Convert a batch of token sequences to one-hot.

        Args:
            tokens: (batch, seq_len) int64 token indices
            padding: number of uniform-distribution positions to prepend

        Returns:
            (batch, seq_len + padding, 256) one-hot float tensor
        """
        out = F.one_hot(tokens, self.ntokens).to(self.dtype)
        if padding > 0:
            batch_size = tokens.shape[0]
            pad = torch.full((batch_size, padding, self.ntokens),
                             1.0 / self.ntokens,
                             dtype=self.dtype, device=tokens.device)
            out = torch.cat([pad, out], dim=1)
        return out


class InputBytesDef(object):
    """Descriptor for the bytes input layer."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.ntokens = 256
        self.output_size = 256
        self.tokens_name = 'bytes'
        self.num_weights = 0
        self.dtype = dtype

    def __str__(self):
        return 'bytes'

    def is_valid(self):
        return True

    def neighbors(self):
        return ()

    def build_module(self) -> InputBytesModule:
        return InputBytesModule(dtype=self.dtype)
