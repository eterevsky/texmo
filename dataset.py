import numpy as np
import os
import random


class DataSet(object):
    def __init__(self, dir):
        texts = []
        count = 0
        for dir, _, files in os.walk(dir):
            for filename in files:
                count += 1
                path = os.path.join(dir, filename)
                with open(path, 'rb') as f:
                    texts.append(f.read())

        self.all = b'\n\n'.join(texts)
        total = len(self.all) / 1E9

        print(f'Dataset loaded: {count} texts, {total:.2f} GB')

    def sample(self, length, batch_size):
        batch = []
        for _ in range(batch_size):
            start = random.randrange(len(self.all) - length)
            sample = self.all[start:start + length]
            batch.append(np.frombuffer(sample, dtype=np.ubyte))
        batch = np.array(batch, dtype=np.ubyte)
        return batch

    @property
    def total_size(self):
        return len(self.all)

    def warmup(self):
        _ = self.sample(1024, 1024)


def build_dataset(data):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()
    return dataset
