import mmap
from typing import Optional

import numpy as np

from .tokenizer import process, unprocess
from .tokenset import CharsTokenSet


class CharsTokenizer(object):
    def __init__(self, token_set: CharsTokenSet):
        self.token_set = token_set

    def tokenize_ids(
        self,
        string: Optional[bytes | mmap.mmap],
        start=0,
        max_tokens=None,
        max_bytes=None,
    ) -> list[int]:
        if max_bytes is not None:
            end = start + max_bytes
        elif max_tokens is not None:
            end = start + max_tokens
        else:
            end = len(string)
        
        while end < len(string) and 128 <= string[end] < 192:
            end += 1

        string = string[start:end]

        if self.token_set.processing:
            string = process(string, True, True)
        else:
            string = string
        
        if max_tokens is not None:
            string = string[:max_tokens]

        byte_array = np.frombuffer(string, dtype=np.byte)

        entropy = np.sum(self.token_set.residual_entropy[byte_array])
        return self.token_set.groups[byte_array], entropy
    
    def untokenize(self, tokens: list[int]) -> bytes:
        chars = []
        for token in tokens:
            chars.append(self.token_set.tokens[token])
        print(chars)
        s = b"".join(chars)
        return unprocess(s)
