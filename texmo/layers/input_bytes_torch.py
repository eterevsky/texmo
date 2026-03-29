import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InputBytesModule(nn.Module):
    """Input module that converts byte token indices to one-hot vectors."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ntokens = 256
        self.output_dtype = dtype

    def step(self, token: int) -> Tensor:
        """Convert a single token index to a one-hot vector.

        Args:
            token: integer token index (0-255)

        Returns:
            (256,) one-hot tensor
        """
        return F.one_hot(torch.tensor(token), self.ntokens).to(self.output_dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        """Convert a batch of token sequences to one-hot.

        Args:
            tokens: (batch, seq_len) integer token indices

        Returns:
            (batch, seq_len, 256) one-hot float tensor
        """
        return F.one_hot(tokens, self.ntokens).to(self.output_dtype)


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
