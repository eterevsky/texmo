from .tokenset import CharsTokenSet


class CharsTokenizer(object):
    def __init__(self, token_set: CharsTokenSet):
        self.token_set = token_set