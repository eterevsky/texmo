import json


def parse_token_set_name(name: str) -> tuple[int, str, str]:
    tokens_n, token_processing, token_type = name.split("_")
    ntokens = int(tokens_n.removeprefix("tokens"))
    return ntokens, token_processing, token_type


class Token(object):
    def __init__(
        self, id: int, string: bytes | None, value: int | None = None,
        is_byte: bool = False,
    ):
        self.id: int = id
        self.string: bytes | None = string
        self.value: int | None = value
        # True for byte-fallback tokens (stored as [int] lists in the
        # JSON). Several ids can share one byte string (e.g. the "\n"
        # piece and the <0x0A> byte token); this flag is how the BPE
        # encoder knows which one is the fallback.
        self.is_byte: bool = is_byte

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


def _parse_merges(merges: list) -> list:
    """Normalize a tokenset's `merges` list.

    Two on-disk shapes exist and both stay supported:

    - `[left_id, right_id, merged_id]` INT TRIPLES (converted
      SentencePiece vocabs, e.g. tokens.256000.gemma) -- passed through
      unchanged; `BpeTokenizer` reads the ids directly.
    - `[left_string, right_string]` STRING PAIRS (hexbpe sets) --
      parsed into `(bytes, bytes)` tuples, in rank order. Ids are NOT
      resolved here: the merged piece is the concatenation of the pair
      and the tokenizer owns the string -> id maps.
    """
    if not merges or len(merges[0]) != 2:
        return merges
    out = []
    for left, right in merges:
        left = _parse_token(left)
        right = _parse_token(right)
        assert isinstance(left, bytes) and isinstance(right, bytes), (
            f"merge pair must be strings: {left!r}, {right!r}")
        out.append((left, right))
    return out


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
        token_set.algorithm = tokens_dict.get("algorithm", "dp")
        token_set.add_bos = tokens_dict.get("add_bos", False)
        token_set.bos_id = tokens_dict.get("bos_id")
        token_set.eos_id = tokens_dict.get("eos_id")
        token_set.pad_id = tokens_dict.get("pad_id")
        token_set.unk_id = tokens_dict.get("unk_id")
        token_set.merges = _parse_merges(tokens_dict.get("merges", []))
        # {id, name, char}: control tokens with their private-use chars.
        token_set.specials = tokens_dict.get("specials", [])
        # Ids matched verbatim in text before the BPE bulk.
        token_set.priority_tokens = tokens_dict.get("priority_tokens", [])
        # Fold sets: P(head byte | token), keyed by the token's
        # character. See fold_tokenizer.py for the accounting model.
        token_set.head_freq = tokens_dict.get("head_freq", {})

        for token_str in tokens_dict["tokens"]:
            token = _parse_token(token_str)
            if isinstance(token, bytes):
                # List-form entries are byte-fallback tokens; string
                # form are ordinary pieces. The distinction only exists
                # in the JSON, so record it here.
                token_set.add_token(
                    token, is_byte=isinstance(token_str, list))
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
        # NOTE: several ids may share one byte string (BPE sets); this
        # map keeps the FIRST id for a string, which is only used by
        # `sequences` resolution in legacy sets. The BPE tokenizer
        # builds its own maps with explicit preference rules.
        self.tokens_by_str: dict[bytes|int, Token] = {}
        self.sequences: dict[bytes, list[Token]] = {}
        # BPE-set extras (populated by from_json when present).
        self.algorithm: str = "dp"
        self.add_bos: bool = False
        self.bos_id: int | None = None
        self.eos_id: int | None = None
        self.pad_id: int | None = None
        self.unk_id: int | None = None
        self.merges: list = []
        self.specials: list = []
        self.priority_tokens: list = []
        self.head_freq: dict[str, float] = {}

    @property
    def residual_bits_per_byte(self) -> float:
        """Corpus-average charge for the information a lossy (fold)
        tokenset destroys; 0 for lossless sets. Baked into the
        tokenset at build time from corpus byte counts."""
        return (self.stats or {}).get("residual_bits_per_byte", 0.0)

    def byte_loss(self, token_loss: float) -> float:
        return (token_loss / self.avg_bytes_per_token
                + self.residual_bits_per_byte)

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

    def add_token(self, string: bytes, is_byte: bool = False):
        assert isinstance(string, bytes)
        new_token = Token(len(self.tokens), string, is_byte=is_byte)

        self.tokens.append(new_token)
        # Keep the first id on duplicates (see __init__ note).
        self.tokens_by_str.setdefault(string, new_token)

    def add_sequence(self, string: bytes, tokens: list[Token]):
        self.sequences[string] = tokens
