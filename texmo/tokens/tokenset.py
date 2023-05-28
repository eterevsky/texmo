import json
from typing import Self


class Token(object):
    def __init__(self, string: bytes, value: int = None):
        self.string: bytes = string
        self.value: int = value
        self.suffix: Self | int = None


class TokenSet(object):
    @staticmethod
    def from_json_file(filename: str) -> Self:
        with open(filename) as file:
            tokens_dict = json.load(file)
        return TokenSet.from_json(tokens_dict)

    @staticmethod
    def from_json(tokens_dict: dict) -> Self:
        token_set = TokenSet(
            tokens_dict["type"],
            tokens_dict["processing"],
            tokens_dict.get("fallback_bits"),
            tokens_dict.get("literal_count"),
        )
        for i, token in enumerate(tokens_dict["tokens"]):
            if type(token) is str:
                b = token.encode("utf-8")
                token_set.add_token(b)
            elif type(token) is list:
                b = bytes(token)
                token_set.add_token(b)
            elif type(token) is int:
                token_set._add_special_token(token)
        return token_set

    def __init__(
        self, token_set_type, processing, fallback_bits, literal_count
    ):
        self.type = token_set_type
        self.processing = processing
        self.fallback_bits = fallback_bits
        self.literal_count = literal_count
        self.tokens = []
        self.tokens_by_str = {}

    @property
    def ntokens(self):
        return len(self.tokens)

    def _add_special_token(self, n: int):
        assert n == len(self.tokens)
        self.tokens.append(Token(None, n))

    def add_token(self, string: bytes):
        assert isinstance(string, bytes)
        new_token = Token(string)

        if len(string) > 1:
            new_token.suffix = string[-1]  # int

            for start in range(1, len(string)):
                suffix_token = self.tokens_by_str.get(string[start:])
                if suffix_token is not None:
                    new_token.suffix = suffix_token
                    break

        for token in self.tokens:
            if token.string is None:
                continue
            if not token.string.endswith(string):
                continue
            assert string != token.string
            if isinstance(token.suffix, int) or len(token.suffix.string) < len(
                string
            ):
                token.suffix = new_token

        self.tokens.append(new_token)
        self.tokens_by_str[string] = new_token
