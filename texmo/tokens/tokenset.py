import json
import numpy as np
from typing import Optional


def parse_token_set_name(name: str) -> tuple[int, str, str]:
    tokens_n, token_processing, token_type = name.split("_")
    ntokens = int(tokens_n.removeprefix("tokens"))
    return ntokens, token_processing, token_type


class Token(object):
    def __init__(self, id: int, string: bytes | None, value: int | None = None):
        self.id: int = id
        self.string: bytes | None = string
        self.value: int | None = value

    def __str__(self):
        if self.string:
            try:
                return repr(self.string.decode("utf-8"))[1:-1]
            except UnicodeDecodeError:
                return repr(self.string)
        else:
            return f"\\{self.value}"

    def __repr__(self):
        if self.string is not None:
            return f"Token({id}, {self.string})"
        else:
            return f"Token({id}, {self.value})"


def _parse_token(string: str|list[int]|int) -> bytes|int:
    if isinstance(string, str):
        return string.encode("utf-8")
    if isinstance(string, list):
        return bytes(string)
    assert isinstance(string, int)
    return string


class TokenSet(object):
    @staticmethod
    def from_json_file(filename: str):
        with open(filename, "rb") as file:
            tokens_dict = json.load(file)
            return TokenSet.from_json(tokens_dict)

    @staticmethod
    def from_json(tokens_dict: dict):
        fallback_bits = tokens_dict.get("fallback_bits")
        if fallback_bits is not None:
            fallback_bits = int(fallback_bits)
        token_set = TokenSet(
            tokens_dict["type"],
            tokens_dict["processing"],
            tokens_dict.get("stats"),
        )
        
        for token_str in tokens_dict["tokens"]:
            token = _parse_token(token_str)
            if isinstance(token, bytes):
                token_set.add_token(token)
            else:
                assert isinstance(token, int)
                token_set.add_ext_token(token)
        
        for seq in tokens_dict.get("sequences", []):
            string = _parse_token(seq["string"])
            assert isinstance(string, bytes)
            tokens = []
            for token_str in seq["tokens"]:
                token_str = _parse_token(token_str)
                token = token_set.get_token(token_str)
                tokens.append(token)
            
            token_set.add_sequence(string, tokens)
        
        return token_set

    def __init__(
        self,
        token_type: str,
        processing: str,
        stats: dict,
    ):
        self.type = token_type
        self.processing = processing
        self.stats = stats
        self.tokens: list[Token] = []
        self.tokens_by_str: dict[bytes|int, Token] = {}
        self.sequences: dict[bytes, list[Token]] = {}

    def byte_loss(self, token_loss: float) -> float:
        raise NotImplementedError()

    @property
    def avg_bytes_per_token(self):
        return self.stats["bytes_per_token"]
    
    @property
    def avg_proc_bytes_per_token(self):
        return self.stats["scanned_bytes"] / self.stats["total_tokens"]

    @property
    def ntokens(self):
        return len(self.tokens)

    @property
    def name(self):
        return f"tokens{self.ntokens}_{self.processing}_{self.type}"

    def get_token(self, token_str: bytes|int) -> Token:
        return self.tokens_by_str[token_str]

    def add_ext_token(self, n: int):
        assert n == len(self.tokens)
        token = Token(len(self.tokens), None, n)
        self.tokens.append(token)
        self.tokens_by_str[n] = token

    def add_token(self, string: bytes):
        assert isinstance(string, bytes)
        new_token = Token(len(self.tokens), string)

        self.tokens.append(new_token)
        self.tokens_by_str[string] = new_token

    def add_sequence(self, string: bytes, tokens: list[Token]):
        self.sequences[string] = tokens
