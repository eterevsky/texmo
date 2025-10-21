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


class BitsTokenizer:
    def __init__(self, nbits: int):
        self.nbits = nbits
        self.tokenset = BitsTokenSet(nbits)

    def tokenize(self, chunk: bytes) -> list[int]:
        radix = 2**self.nbits

        out = []
        for b in chunk:
            for i in range(8 // self.nbits):
                out.append(b % radix)
                b //= radix

        return out

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        return self.tokenize(chunk)

    def untokenize(self, tokens: list[int]) -> bytes:
        out = []
        b = 0
        p = 1
        for token in tokens:
            b += token * p
            p *= 2**self.nbits
            if p == 256:
                out.append(b)
                b = 0
                p = 1
