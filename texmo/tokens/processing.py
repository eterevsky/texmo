import regex


WORD_BOUNDARY = regex.compile(r"(?<=\P{L})(?=\p{L})|(?<=\p{L})(?=\P{L})")
CAPITALIZED_MARKER = "\x14"
ALLCAPS_MARKER = "\x15"
WORD_MARKER = "\x16"


def process(s: bytes, mark_caps: bool, mark_words: bool) -> str:
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


def unprocess(text: bytes) -> str:
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

    return "".join(words)
