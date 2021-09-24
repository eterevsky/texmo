from copy import deepcopy
import numpy as np
import random
import re
import sys
import torch
from torch import nn

def load_samples(path):
    text = open(path, 'rb').read()
    text  = re.sub(rb'\r\n', rb'\n', text)
    text = re.sub(rb'(?<=\w)\n(?=\w)', b' ', text)
    return text


def one_hot(text):
    enc = np.zeros((len(text), 256), dtype=np.float32)
    for i, c in enumerate(text):
        enc[i, c] = 1
    return enc


def extend_with_0(s, sample_len):
    return s + b'\0' * (sample_len - len(s))


def gen_batch(text, sample_len=32, batch_size=32):
    input = np.zeros((batch_size, sample_len, 256), dtype=np.float32)
    target = []

    for isample in range(batch_size):
        start = random.randrange(0, len(text))
        sample = extend_with_0(text[start:start + sample_len + 1], sample_len + 1)
        input[isample,:,:] = one_hot(sample[:-1])
        target.append(sample[1:])

    input_seq = torch.from_numpy(input)
    target_seq = torch.Tensor(target).long()
    return input_seq, target_seq


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.hidden_dim = 128
        self.n_layers = 2

        self.rnn = nn.RNN(input_size, self.hidden_dim, self.n_layers, batch_first=True)
        self.lstm = nn.LSTM(input)
        self.output_layer = nn.Linear(self.hidden_dim, output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.output_layer(out)

        return out, hidden


def predict(model, character, hidden=None):
    character = np.expand_dims(one_hot(character), 0)
    character = torch.from_numpy(character)

    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data
    char_ind = torch.max(prob, dim=0)[1].item()

    return char_ind, hidden


def sample(model, out_len, start=b'hey'):
    model.eval()
    cont = []

    chars = start
    hidden = None
    for _ in range(out_len):
        c, hidden = predict(model, chars, hidden)
        cont.append(c)
        chars = bytes([c])
    return start + bytes(cont)


def main(text_file):
    data = load_samples(text_file)
    device = torch.device('cpu')

    model = Model(input_size=256, output_size=256)
    model.to(device)

    sample_len = 128
    batch_size = 64

    n_epochs = 10000
    lr = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)
    best_loss = 10000
    best_model = None

    for epoch in range(n_epochs):
        input_seq, target_seq = gen_batch(data, sample_len, batch_size)

        optimizer.zero_grad()
        input_seq.to(device)
        output, _ = model(input_seq)
        loss = criterion(output, target_seq.view(-1))
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(model)

        print("Epoch {}/{}. Loss: {}".format(epoch, n_epochs, loss.item()))

    print(sample(best_model, 128))


if __name__ == '__main__':
    main(sys.argv[1])