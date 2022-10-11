import latency
import mmap
import numpy as np
import os
import random


class DataSet(object):
    def __init__(self, file_or_dir):
        if os.path.isdir(file_or_dir):
            dir = file_or_dir

            texts = []
            count = 0
            for dir, _, files in os.walk(dir):
                for filename in files:
                    count += 1
                    path = os.path.join(dir, filename)
                    with open(path, "rb") as f:
                        texts.append(f.read())

            self.all = b"\n\n".join(texts)
            total = len(self.all) / 1e9

            print(f"Dataset loaded: {count} texts, {total:.2f} GB")
        else:
            self.file = open(file_or_dir, "rb")
            self.all = mmap.mmap(
                self.file.fileno(),
                0,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ,
            )

            total = len(self.all) / 1E9

            print(f"Dataset mmap'ed: {total:.2f} GB")

    def sample(self, length, batch_size):
        with latency.timer(f"dataset-sample-{length}"):
            batch = []
            for _ in range(batch_size):
                start = random.randrange(len(self.all) - length)
                sample = self.all[start : start + length]
                batch.append(np.frombuffer(sample, dtype=np.ubyte))
            batch = np.array(batch, dtype=np.ubyte)
            return batch

    @property
    def total_size(self):
        return len(self.all)

    def warmup(self):
        _ = self.sample(1024, 1024)

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(self.all)


def build_dataset(data):
    print(f"Loading dataset from {data}")
    dataset = DataSet(data)
    dataset.warmup()
    return dataset


if __name__ == "__main__":
    dataset = DataSet("data")
    dataset.save("data.txt")
