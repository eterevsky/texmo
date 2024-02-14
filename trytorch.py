import torch
from torch import nn

from texmo.dataset import DataSet
from texmo.tokens import set_tokens_dir


set_tokens_dir("tokens")
dataset = DataSet(path="data/books3.txt", path_processed="data/books3_caps_words.txt")

batch = 512
# batch = 4
length = 128
lr = 0.0078
steps = 1024
device = "cuda"

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru0 = nn.GRU(256, 256,  1, batch_first=True)
        self.gru1 = nn.GRU(256, 512,  1, batch_first=True)
        self.output = nn.Linear(512, 256)

    def forward(self, x, hidden):
        x, hidden0 = self.gru0(x, hidden[0])
        x, hidden1 = self.gru1(x, hidden[1])
        x = self.output(x)
        return x, (hidden0, hidden1)
    
def init_hidden(batch_size, device):
    hidden0 = torch.zeros(1, batch_size, 256, device=device)
    hidden1 = torch.zeros(1, batch_size, 512, device=device)
    return hidden0, hidden1
    

initial_hidden = init_hidden(batch, device)
model = Model().to(device)


sample = dataset.sample_tokens(length, batch, "tokens256_raw_bytes")
x = torch.tensor(sample, dtype=torch.int64, device=device)
x = nn.functional.one_hot(x, num_classes=256).to(torch.float32)

model = torch.jit.trace(model, (x, initial_hidden))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0078)

for step in range(steps):
    sample = dataset.sample_tokens(length, batch, "tokens256_raw_bytes")
    optimizer.zero_grad()

    x = torch.tensor(sample, dtype=torch.int64, device=device)
    x = nn.functional.one_hot(x, num_classes=256).to(torch.float32)

    y, _ = model(x, initial_hidden)

    loss = loss_fn(y[:,:-1,:], x[:,1:,:])
    loss.backward()
    optimizer.step()
    print(f"Step {step} loss: {loss.item()}")
