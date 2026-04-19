from .processing import process, unprocess
from .tokenset import TokenSet


class Span(object):
    def __init__(self, string: bytes, tokens: list[int]):
        self.string = string
        self.len = len(string)
        self.reversed_tokens = list(reversed(tokens))
        self.cost = len(tokens)
        self.suffix_span = None


class SuffixState(object):
    def __init__(self, string: bytes, span: Span):
        self.string = string
        self.span: Span = span
        self.next: list[SuffixState | None] = [None] * 256


def _find_suffix_span(spans: dict[bytes, Span], string: bytes) -> Span:
    """Find the longest span which is the suffix of the string.

        The full string is considered its own suffix.
    """
    for start in range(0, len(string) + 1):
        suffix = string[start:]
        suffix_span = spans.get(suffix)
        if suffix_span is not None:
            return suffix_span
    assert False, f"Unreachable: {string}"


def _build_suffix_states(spans: dict[bytes, Span]) -> dict[bytes, SuffixState]:
    states = {}
    states[b""] = SuffixState(b"", spans[b""])

    for span in spans.values():
        for end in range(1, len(span.string) + 1):
            prefix = span.string[:end]
            if prefix not in states:
                states[prefix] = SuffixState(prefix, _find_suffix_span(spans, prefix))
    
    for state in states.values():
        for byte in range(256):
            string = state.string + bytes([byte])
            for start in range(len(string) + 1):
                suffix = string[start:]
                next_state = states.get(suffix)
                if next_state is not None:
                    state.next[byte] = next_state
                    break
    
    return states


class DecoderState(object):
    def __init__(self, tokens: list[int], string: bytes):
        self.tokens: tuple = tokens
        self.string = string
        self.next: list[tuple[DecoderState, bytes]] = []


class Decoder(object):
    def __init__(self, tokenset: TokenSet):
        self.tokens_to_string: dict[tuple, bytes] = {}
        self.has_longer: set[tuple] = set()

        for token in tokenset.tokens:
            if token.string is not None:
                self.tokens_to_string[(token.id,)] = token.string

        for string, tokens in tokenset.sequences.items():
            tokens = tuple(t.id for t in tokens)
            self.tokens_to_string[tokens] = string
            for end in range(1, len(tokens)):
                self.has_longer.add(tuple(tokens[:end]))

    def decode(self, tokens: list[int]) -> bytes:
        out = b""
        fragment = []
        for c in tokens:
            fragment.append(c)
            if tuple(fragment) in self.has_longer:
                continue
            tail = []
            while tuple(fragment) not in self.tokens_to_string:
                tail.append(fragment.pop())
            out += self.tokens_to_string[tuple(fragment)]
            fragment = tail
            fragment.reverse()

        if fragment:
            while fragment and (tuple(fragment) not in self.tokens_to_string):
                fragment.pop()
            if fragment:
                out += self.tokens_to_string[tuple(fragment)]
        return out


class Tokenizer(object):
    def __init__(self, tokenset: TokenSet):
        self.tokenset: TokenSet = tokenset
        self.spans: dict[bytes, Span] = {
            b"": Span(b"", []),
        }

        for token in self.tokenset.tokens:
            if token.string is not None:
                span = Span(token.string, [token.id])
                self.spans[span.string] = span
        
        for string, tokens in self.tokenset.sequences.items():
            span = Span(string, list(t.id for t in tokens))
            self.spans[string] = span
        
        for span in self.spans.values():
            if span.string:
                span.suffix_span = _find_suffix_span(self.spans, span.string[1:])
        
        self._suffixes = _build_suffix_states(self.spans)
        self._decoder = Decoder(tokenset)

    def tokenize(self, chunk: bytes) -> list[int]:
        if self.tokenset.processing == "capswords":
            chunk = process(chunk)
        return self.tokenize_processed(chunk)

    def tokenize_processed_impl(self, chunk: bytes) -> list[int]:
        state = self._suffixes[b""]
        cost = [0]
        spans = [self.spans[b""]]

        for c in chunk:
            state = state.next[c]
            span = state.span

            best_span = span
            best_cost = cost[-span.len] + span.cost

            span = span.suffix_span
            while span.len > 0:
                new_cost = cost[-span.len] + span.cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_span = span
                span = span.suffix_span
            
            cost.append(best_cost)
            spans.append(best_span)
        
        tokens = []
        pos = len(spans) - 1
        while pos > 0:
            span = spans[pos]
            tokens.extend(span.reversed_tokens)
            pos -= span.len

        tokens.reverse()
        return tokens

    def tokenize_processed(self, chunk: bytes) -> list[int]:
        # with latency.timer("Tokenizer.tokenize_processed"):
        return self.tokenize_processed_impl(chunk)

    def untokenize(self, token_ids: list[int]) -> bytes:
        text = self._decoder.decode(token_ids)
        if self.tokenset.processing == "capswords":
            text = unprocess(text)
        
        return text
