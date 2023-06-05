import json
from typing import Self


class Token(object):
    def __init__(self, id: int, string: bytes, value: int = None):
        self.id: int = id
        self.string: bytes = string
        self.value: int = value
        self.suffix: Self | int = None

    def __str__(self):
        if self.string:
            try:
                return repr(self.string.decode("utf-8"))[1:-1]
            except UnicodeDecodeError:
                return repr(self.string)
        else:
            return f"\{self.value}"

    def __repr__(self):
        if self.string is not None:
            return f"Token({id}, {self.string})"
        else:
            return f"Token({id}, {self.value})"


class TokenSet(object):
    @staticmethod
    def from_json_file(filename: str) -> Self:
        with open(filename) as file:
            tokens_dict = json.load(file)
        return TokenSet.from_json(tokens_dict)

    @staticmethod
    def build_with_fallback_bits(fallback_bits: int) -> Self:
        token_set = TokenSet("str_with_fallback_bits", [], fallback_bits, None)
        for i in range(2**fallback_bits):
            token_set._add_special_token(i)
        return token_set

    @staticmethod
    def build_with_fallback_dist() -> Self:
        token_set = TokenSet(
            "str_with_fallback_distribution", [], None, [1] * 256
        )
        token_set._add_special_token(0)
        return token_set

    @staticmethod
    def from_json(tokens_dict: dict) -> Self:
        token_set = TokenSet(
            tokens_dict["type"],
            tokens_dict["processing"],
            tokens_dict.get("fallback_bits"),
            tokens_dict.get("literal_count"),
            tokens_dict.get("stats"),
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
        self, token_set_type, processing, fallback_bits, literal_count, stats
    ):
        self.type = token_set_type
        self.processing = processing
        self.fallback_bits = fallback_bits
        self.literal_count = literal_count
        self.stats = stats
        self.tokens = []
        self.tokens_by_str = {}

    @property
    def tokens_in_literal(self) -> int:
        match self.type:
            case "fallback_distribution" | "str_with_fallback_distribution":
                return 1
            case "fallback_bits":
                return 8 // self.fallback_bits

    @property
    def bytes_per_token(self) -> float:
        return self.stats["initial_size"] / (
            self.stats["total_tokens"]
            + self.tokens_in_literal * self.stats["total_literals"]
        )

    @property
    def entropy0(self) -> float:
        return self.stats["literal_dist_entropy"]

    def byte_loss(self, token_loss: float) -> float:
        loss = token_loss / self.bytes_per_token
        if self.type in (
            "fallback_distribution",
            "str_with_fallback_distribution",
        ):
            loss += (
                self.stats["literal_dist_entropy"]
                * self.stats["total_literals"]
                / (self.stats["total_literals"] + self.stats["total_tokens"])
            )
        return loss

    @property
    def ntokens(self):
        return len(self.tokens)

    def _add_special_token(self, n: int):
        assert n == len(self.tokens)
        self.tokens.append(Token(len(self.tokens), None, n))

    def add_token(self, string: bytes):
        assert isinstance(string, bytes)
        new_token = Token(len(self.tokens), string)

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

    def literal_encodings(self):
        match self.type:
            case "fallback_distribution" | "str_with_fallback_distribution":
                return [[self.tokens[0]] for _ in range(256)]
            case "fallback16":
                fallbacks = []
                for i in range(256):
                    s = f"{i:02x}".encode("utf-8")
                    fallbacks.append(
                        [
                            self.tokens_by_str[s[1:]],
                            self.tokens_by_str[s[:1]],
                            self.tokens_by_str[bytes([0x10])],
                        ]
                    )
                return fallbacks
            case "str_with_fallback_bits" | "fallback_bits":
                fallbacks = []
                for value in range(256):
                    literal = []
                    for i in range(8 // self.fallback_bits):
                        digit = value % 2**self.fallback_bits
                        literal.append(self.tokens[digit])
                        value //= 2**self.fallback_bits
                    # print(literal)
                    fallbacks.append(literal)
                # print(fallbacks)
                return fallbacks
