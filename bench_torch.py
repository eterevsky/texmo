import argparse
import math
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'  {torch.cuda.get_device_name()}')

    dataset = DataSet(path=args.data)
    model = GRUModel(vocab_size=256, hidden_size=256).to(device)

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
    parser = argparse.ArgumentParser(description='PyTorch GRU benchmark')
    parser.add_argument('-d', '--data', type=str, default='data/books3.txt',
                        help='path to data file')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of training steps')
    parser.add_argument('--lr', type=float, default=0.0078125,
                        help='learning rate (default: 1/128)')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
