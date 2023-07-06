import json
from typing import Optional


def parse_token_set_name(name: str) -> tuple[int, str, str]:
    tokens_n, token_processing, token_type = name.split("_")
    ntokens = int(tokens_n.removeprefix("tokens"))
    return ntokens, token_processing, token_type


class Token(object):
    def __init__(self, id: int, string: bytes, value: Optional[int] = None):
        self.id: int = id
        self.string: bytes = string
        self.value: Optional[int] = value
        self.suffix = None

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
            fallback_bits,
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

    @staticmethod
    def build(
        type: str,
        fallback_bits: Optional[int],
        processing: str,
        tokens: list[bytes],
        stats: dict,
        bytes_per_token: Optional[float] = None,
    ):
        assert isinstance(fallback_bits, Optional[int])
        token_set = TokenSet(type, processing, fallback_bits, stats, bytes_per_token=bytes_per_token)
        if type == "fallback_distribution":
            reserved_tokens = 1
        elif type == "fallback_bits":
            reserved_tokens = 2**fallback_bits
        else:
            reserved_tokens = 0
        for i in range(reserved_tokens):
            token_set._add_special_token(i)
        for t in tokens:
            token_set.add_token(t)
        return token_set

    # TODO: Instead of `stats` argument provide important stats as separate
    # arguments.
    def __init__(
        self,
        token_set_type,
        processing,
        fallback_bits,
        stats,
        bytes_per_token=None,
    ):
        self.type = token_set_type
        self.processing = processing
        self.fallback_bits = fallback_bits
        self.stats = stats
        self.tokens = []
        self.tokens_by_str = {}

        if bytes_per_token is not None:
            self.bytes_per_token = bytes_per_token
        else:
            self.bytes_per_token = stats["initial_size"] / (
                stats["total_tokens"]
                + self.tokens_in_literal * stats["total_literals"]
            )

    @property
    def token_type(self) -> str:
        if self.type in (
            "dist",
            "fallback_distribution",
            "str_with_fallback_distribution",
        ):
            cost = self.stats["literal_cost"]
            return f"dist{cost}"
        elif self.type in ("bits", "fallback_bits"):
            return f"bits{self.fallback_bits}"
        elif self.type == "all":
            return "all"

    @property
    def tokens_in_literal(self) -> int:
        match self.type:
            case "fallback_distribution" | "str_with_fallback_distribution" | "all" | "dist2" | "dist4" | "dist8":
                return 1
            case "fallback_bits":
                return 8 // self.fallback_bits
            case "bits1":
                return 8
            case "bits2":
                return 4
            case "bits4":
                return 2

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
