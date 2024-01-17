import mmap
import numpy as np
from typing import Optional

from .tokenset import CharsTokenSet
from .tokenizer import process


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
        print(start, max_tokens, max_bytes, self.token_set.processing)

        if max_bytes is not None:
            string = string[start : start + max_bytes]
        else:
            string = string[start:]

        if max_tokens is not None:
            string = string[:max_tokens]

        if self.token_set.processing:
            string = process(string, True, True)
        else:
            string = string
        
        if max_tokens is not None:
            string = string[:max_tokens]

        byte_array = np.frombuffer(string, dtype=np.byte)
        return self.token_set.groups[byte_array]
