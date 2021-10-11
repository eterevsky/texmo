import numpy as np
import os
import random


class DataSet(object):
    def __init__(self, dir):
        self.texts = []
        self.cum_weights = []
        total = 0
        for dir, _, files in os.walk(dir):
            for filename in files:
                path = os.path.join(dir, filename)
                print(f'Loading {path}')
                with open(path, 'rb') as f:
                    text = f.read()
                self.texts.append(text)
                total += len(text)
                self.cum_weights.append(total)

    def sample(self, length, size):
        texts_batch = random.choices(self.texts, cum_weights=self.cum_weights, k=size)
        batch = []
        for text in texts_batch:
            start = random.randrange(len(text))
            sample = text[start:start + length]
            sample += b'\0' * (length - len(sample))
            batch.append(list(sample))
        batch = np.array(batch, dtype=np.ubyte)
        return batch


