import os
import sys
from typing import Iterable


def encode_word(word: str) -> str:
    if word[0].isupper() and all(c.islower() for c in word[1:]):
        return '\uE000\uE002' + word.lower() + '\uE003'
    
    if all(c.isupper() for c in word):
        return '\uE001\uE002' + word.lower() + '\uE003'

    return '\uE002' + word + '\uE003'
    

def encode_words(stream: Iterable[str]) -> Iterable[str]:
    word = ""
    for char in stream:
        if char.isalpha():
            word += char
        else:
            if word:
                for c in encode_word(word):
                    yield c                    
                word = ''
            yield char
    if word:
        for c in word:
            yield c


def encode(stream: Iterable[str]) -> Iterable[str]:
    suffix = ""
    for c in encode_words(stream):
        if suffix == "" and c == "\uE003":
            suffix = "\uE003"
        elif suffix == "\uE003" and c == " ":
            suffix = "\uE003 "
        elif suffix == "\uE003 " and c in ("\uE000", "\uE001", "\uE002"):
            yield "\uE003"
            yield c
            suffix = ""
        else:
            for s in suffix:
                yield s
            yield c
            suffix = ""
    for s in suffix:
        yield s


def decode_spaces(stream: Iterable[str]) -> Iterable[str]:
    end_word = False
    for char in stream:
        if end_word:
            if char == "\uE002":
                yield " "
            elif char in ("\uE000", "\uE001"):
                yield " "
                yield char
            else:
                yield char
            end_word = False
        else:
            if char == "\uE003":
                end_word = True
            elif char != "\uE002":
                yield char


def decode(stream: Iterable[str]) -> Iterable[str]:
    word = ""
    capitalize = False
    all_caps = False
    for char in decode_spaces(stream):
        assert char != "\uE002"
        assert char != "\uE003"
        if char.isalpha():
            word += char
        else:
            if word:
                if capitalize:
                    yield word[0].upper()
                    for c in word[1:]:
                        yield c
                elif all_caps:
                    for c in word.upper():
                        yield c
                else:
                    for c in word:
                        yield c
                word = ""
                capitalize = False
                all_caps = False
            if char == "\uE000":
                capitalize = True
                all_caps = False
            elif char == "\uE001":
                capitalize = False
                all_caps = True
            else:
                yield char
                capitalize = False
                all_caps = False

    if word:
        if capitalize:
            yield word[0].upper()
            for c in word[1:]:
                yield c
        elif all_caps:
            for c in word.upper():
                yield c
        else:
            for c in word:
                yield c


def read_file(filename: str) -> Iterable[str]:
    with open(filename, encoding="utf-8", newline='') as f:
        for line in f:
            for c in line:
                yield c

word = ""

with open(sys.argv[2], "w", encoding="utf-8", newline='') as out:
    for c in decode(read_file(sys.argv[1])):
        out.write(c)
