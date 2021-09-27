import argparse
import jax
import jax.numpy as jnp
import numpy as np
import os
import random


class TrainSet(object):
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

    def sample(self, length, batch_size):
        texts_batch = random.choices(self.texts, cum_weights=self.cum_weights, k=batch_size)
        batch = []
        for text in texts_batch:
            start = random.randrange(len(text))
            sample = text[start:start + length]
            sample += b'\0' * (length - len(sample))
            batch.append(list(sample))
            print(sample)
        batch = np.array(batch, dtype=np.ubyte)
        print(batch)
        print(batch.shape)
        one_hot = jax.nn.one_hot(batch, 256)
        print(one_hot)
        print(one_hot.shape)
        return batch


def main(dir):
    print(f'Training data: {dir}')
    train_set = TrainSet(dir)
    train_set.sample(16, 16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, help='directory with training data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.dir)
