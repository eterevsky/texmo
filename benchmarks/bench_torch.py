"""Standalone PyTorch training benchmark (a hand-written reference
model, not a texmo `Model2`).

torch is no longer a texmo dependency -- the torch backend and the
per-layer `nn.Module` implementations were removed 2026-08-04 (see
docs/backends.md). This script is kept as the cross-framework
reference point behind the `PyTorch + ...` rows in docs/bench.txt; run
it with torch supplied ad hoc:

    uv run --with torch python scripts/bench_torch.py

On Windows, torch's bundled cudnn DLLs and the winjax CUDA plugin
fight over PATH, so prefer a separate throwaway env there.
"""
import argparse
import math
import os
import sys
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# texmo is not an installed package; scripts/ must put the repo root
# on the path itself.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.dataset import DataSet

_1_BY_LOG2 = 1.0 / math.log(2.0)


class GRUModel(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) integer tokens
        oh = F.one_hot(x, self.vocab_size).float()  # (batch, seq_len, 256)
        # Shift right: prepend zeros, drop last
        oh = F.pad(oh[:, :-1], (0, 0, 1, 0))  # (batch, seq_len, 256)
        mid, _ = self.gru(oh)  # (batch, seq_len, hidden)
        return self.output(mid)  # (batch, seq_len, 256)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=256, num_heads=4, head_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        d_model = num_heads * head_dim  # 256
        self.qkv = nn.Linear(vocab_size, 3 * d_model)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.proj = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) integer tokens
        B, T = x.shape
        oh = F.one_hot(x, self.vocab_size).float()
        oh = F.pad(oh[:, :-1], (0, 0, 1, 0))  # shift right

        qkv = self.qkv(oh)  # (B, T, 3 * d_model)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d_model)
        out = self.proj(out)
        return self.output(out)  # (B, T, vocab_size)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if device_str == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device_str == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(device_str)


def train(model, name, args):
    device = _resolve_device(args.device)
    print(f'\n=== {name} ===')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  {torch.cuda.get_device_name()}')

    dataset = DataSet(path=args.data)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(args.steps):
        if step == 1:
            start = perf_counter()

        data = dataset.sample_tokens(ntokens=256, batch=256, tokenset_name='bytes')
        data = torch.from_numpy(np.array(data, dtype=np.int64)).to(device)

        logits = model(data)  # (batch, seq_len, 256)
        loss = loss_fn(logits.reshape(-1, 256), data.reshape(-1))
        loss_bits = loss.item() * _1_BY_LOG2

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step < 10 or step % 10 == 0:
            print(f'{step} {loss_bits:.4f}')

    finish = perf_counter()
    elapsed = finish - start
    ms_per_step = elapsed / (args.steps - 1) * 1000
    print(f'Latency: {elapsed:.2f}s ({ms_per_step:.1f} ms/step)')


def main():
    parser = argparse.ArgumentParser(description='PyTorch benchmarks')
    parser.add_argument('-d', '--data', type=str, default='data/books3.txt',
                        help='path to data file')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of training steps')
    parser.add_argument('--lr', type=float, default=0.0078125,
                        help='learning rate (default: 1/128)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=('auto', 'cpu', 'cuda', 'mps'),
                        help='torch device (default: auto)')
    args = parser.parse_args()
    train(GRUModel(), 'GRU.256', args)
    train(TransformerModel(), 'Transformer 4h64d', args)


if __name__ == '__main__':
    main()
