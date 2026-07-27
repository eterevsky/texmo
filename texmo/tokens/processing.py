import regex

WORD_BOUNDARY = regex.compile(r"(?<=\P{L})(?=\p{L})|(?<=\p{L})(?=\P{L})")
CAPITALIZED_MARKER = "\x14"
ALLCAPS_MARKER = "\x15"
WORD_MARKER = "\x16"
# capswords2 only: the next letter is uppercase. Covers the mixed-case
# words ("McDonald", "iPhone") that capswords passes through verbatim,
# forcing every tokenset to carry uppercase letters it otherwise would
# not need. CAPITALIZED_MARKER stays even though this could express it
# at the same cost -- a leading capital is highly predictable after
# ". " while a mid-word one is not, and merging them would blur a
# signal the model can use.
UPPER_MARKER = "\x17"

# Markers that begin a word, so `unprocess2` knows a WORD_MARKER before
# one of them stood for an elided space.
_WORD_START_MARKERS = (CAPITALIZED_MARKER, ALLCAPS_MARKER, UPPER_MARKER)


def process(s: bytes, mark_caps: bool = True, mark_words: bool = True) -> bytes:
    try:
        s = s.decode("utf-8")
    except UnicodeDecodeError:
        print(s[:1000])
        raise

    out = []
    i = 0
    words = WORD_BOUNDARY.split(s)

    for i, word in enumerate(words):
        if not word:
            continue
        if word.isalpha():
            if (
                mark_caps
                and word[0].isupper()
                and (len(word) == 1 or word[1:].islower())
            ):
                out.append(CAPITALIZED_MARKER)
                out.append(word.lower())
            elif mark_caps and word.isupper():
                out.append(ALLCAPS_MARKER)
                out.append(word.lower())
            else:
                out.append(word)
            if mark_words:
                out.append(WORD_MARKER)
        else:
            if mark_words and word == " " and 0 < i < len(words) - 1:
                pass
            else:
                out.append(word)

    return "".join(out).encode("utf-8")


def unprocess(text: bytes) -> bytes:
    try:
        text = text.decode("utf-8")
    except UnicodeDecodeError:
        return text

    words = WORD_BOUNDARY.split(text)

    for i in range(len(words)):
        word = words[i]
        if not word.isalpha():
            continue

        if i > 0 and words[i - 1][-1] == "\x14":
            words[i - 1] = words[i - 1][:-1]
            words[i] = word.capitalize()
        elif i > 0 and words[i - 1][-1] == "\x15":
            words[i - 1] = words[i - 1][:-1]
            words[i] = word.upper()

        if i + 1 < len(words):
            if words[i + 1] == WORD_MARKER:
                words[i + 1] = " "
            elif words[i + 1].startswith(WORD_MARKER) and words[i + 1][
                1
            ] in (CAPITALIZED_MARKER, ALLCAPS_MARKER):
                words[i + 1] = " " + words[i + 1][1:]
            elif words[i + 1].startswith(WORD_MARKER):
                words[i + 1] = words[i + 1][1:]

    return "".join(words).encode("utf-8")


# --- capswords2 -------------------------------------------------------
#
# Same shape as capswords (word terminator, single inter-word space
# elided, whole-word case markers) plus UPPER_MARKER for mid-word
# capitals. Written as an explicit state machine rather than a regex
# split so it mirrors `src/processing.rs` character for character --
# the Rust generator and the Python tokenizer must agree exactly or
# tokensets decode to garbage.


def _add_word2(out: list, word: str) -> None:
    first, rest = word[0], word[1:]
    if first.isupper() and (not rest or rest.islower()):
        out.append(CAPITALIZED_MARKER)
        out.append(word.lower())
    elif first.isupper() and rest and rest.isupper():
        out.append(ALLCAPS_MARKER)
        out.append(word.lower())
    elif any(c.isupper() for c in word):
        for c in word:
            if c.isupper():
                out.append(UPPER_MARKER)
                out.append(c.lower())
            else:
                out.append(c)
    else:
        out.append(word)
    out.append(WORD_MARKER)


def process2(s: bytes) -> bytes:
    text = s.decode("utf-8")
    out: list[str] = []
    word: list[str] = []
    # 'word' | 'space_after_word' | 'nonword'
    state = "nonword"

    for ch in text:
        if ch.isalpha():
            if state == "word":
                word.append(ch)
            else:
                # A pending space before a word is elided: WORD_MARKER
                # on the previous word already implies the boundary.
                word = [ch]
                state = "word"
        elif ch == " ":
            if state == "word":
                _add_word2(out, "".join(word))
                word = []
                state = "space_after_word"
            elif state == "space_after_word":
                out.append(" ")
                out.append(ch)
                state = "nonword"
            else:
                out.append(ch)
        else:
            if state == "word":
                _add_word2(out, "".join(word))
                word = []
            elif state == "space_after_word":
                out.append(" ")
            out.append(ch)
            state = "nonword"

    if state == "word":
        _add_word2(out, "".join(word))
    elif state == "space_after_word":
        out.append(" ")

    return "".join(out).encode("utf-8")


def unprocess2(text: bytes) -> bytes:
    try:
        s = text.decode("utf-8")
    except UnicodeDecodeError:
        return text

    out: list[str] = []
    # Pending case transform for the letters that follow.
    upper_next = False     # UPPER_MARKER: exactly one letter
    upper_word = False     # ALLCAPS_MARKER: until WORD_MARKER
    i = 0
    for i, ch in enumerate(s):
        if ch == CAPITALIZED_MARKER or ch == UPPER_MARKER:
            upper_next = True
        elif ch == ALLCAPS_MARKER:
            upper_word = True
        elif ch == WORD_MARKER:
            upper_word = False
            upper_next = False
            # The space between two words was dropped by process2, so
            # put it back when another word starts here.
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt and (nxt.isalpha() or nxt in _WORD_START_MARKERS):
                out.append(" ")
        elif upper_next or upper_word:
            out.append(ch.upper())
            upper_next = False
        else:
            out.append(ch)

    return "".join(out).encode("utf-8")


# Processing pipelines by the name stored in a tokenset's `processing`
# field. 'raw' means the bytes are used as they are.
PROCESSORS = {
    "capswords": (process, unprocess),
    "capswords2": (process2, unprocess2),
}


def apply_processing(name: str, data: bytes) -> bytes:
    fns = PROCESSORS.get(name)
    return data if fns is None else fns[0](data)


def undo_processing(name: str, data: bytes) -> bytes:
    fns = PROCESSORS.get(name)
    return data if fns is None else fns[1](data)
