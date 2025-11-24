import numpy as np
from rich import print as pprint


class BitsTokenSet:
    processing = 'raw'

    def __init__(self, nbits):
        self.nbits = nbits
        if nbits < 8:
            self.name = f'bits.{nbits}'
        else:
            self.name = 'bytes'

        self.avg_proc_bytes_per_token = nbits / 8
        self.avg_bytes_per_token = nbits / 8


class BitsTokenizer1:
    def __init__(self):
        self.tokenset = BitsTokenSet(1)

    def tokenize(self, chunk: bytes) -> np.ndarray:
        chunk = np.frombuffer(chunk, dtype=np.uint8)
        return np.unpackbits(chunk, bitorder='little')

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        return bytes(np.packbits(tokens, bitorder='little'))


class BitsTokenizer2:
    def __init__(self):
        self.tokenset = BitsTokenSet(2)

    def tokenize(self, chunk: bytes) -> np.ndarray:
        chunk = np.frombuffer(chunk, dtype=np.uint8)

        chunk0 = chunk % 4
        chunk1 = chunk % 16 >> 2
        chunk2 = chunk % 64 >> 4
        chunk3 = chunk >> 6
    
        return np.vstack((chunk0, chunk1, chunk2, chunk3)).reshape((-1,), order='F')

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        tokens = np.array(tokens)
        chunk0 = tokens[0::4]
        chunk1 = tokens[1::4]
        chunk2 = tokens[2::4]
        chunk3 = tokens[3::4]

        chunk = chunk0 + (chunk1 << 2) + (chunk2 << 4) + (chunk3 << 6)

        return bytes(chunk.astype(np.uint8))


class BitsTokenizer4:
    def __init__(self):
        self.tokenset = BitsTokenSet(4)

    def tokenize(self, chunk: bytes) -> np.ndarray:
        chunk = np.frombuffer(chunk, dtype=np.uint8)

        chunk0 = chunk % 16
        chunk1 = chunk >> 4

        return np.vstack((chunk0, chunk1)).reshape((-1,), order='F')

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        tokens = np.array(tokens)
        chunk0 = tokens[0::2]
        chunk1 = tokens[1::2]
        chunk = chunk0 + (chunk1 << 4)
        return bytes(chunk.astype(np.uint8))


class BytesTokenizer:
    def __init__(self):
        self.tokenset = BitsTokenSet(8)

    def tokenize(self, chunk: bytes) -> np.ndarray:
        return np.frombuffer(chunk, dtype=np.uint8)

    def tokenize_processed(self, chunk: bytes) -> np.ndarray:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        return bytes(tokens)
