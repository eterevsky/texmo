import os
import sys

word = ""

with open(sys.argv[1], encoding="utf-8") as f, open(
    sys.argv[2], "w", encoding="utf-8"
) as out:
    for line in f:
        for c in line:
            if c.isalpha():
                word += c
            else:
                if word:
                    if word[0].isupper() and all(c.islower() for c in word[1:]):
                        out.write("\uE000" + word.lower() + c)
                    elif all(c.isupper() for c in word):
                        out.write("\uE001" + word.lower() + c)
                    else:
                        for l in word:
                            if l.islower():
                                out.write(l)
                            else:
                                out.write("\uE002" + l.lower())
                        out.write(c)
                    word = ""
                else:
                    out.write(c)
