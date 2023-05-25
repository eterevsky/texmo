#!/bin/sh

cargo build --release

# target/release/texmo -d data/books3.txt optimize-tokens -n 128 -f 1 --processing= --initial-size=108362047381 -i tokens/tokens64_raw_f1.json -o tokens/tokens128_raw_f1.json
target/release/texmo -d data/books3.txt optimize-tokens -n 128 -f 2 --processing= --initial-size=108362047381 -i tokens/tokens64_raw_f2.json -o tokens/tokens128_raw_f2.json
target/release/texmo -d data/books3.txt optimize-tokens -n 128 -f 4 --processing= --initial-size=108362047381 -i tokens/tokens64_raw_f4.json -o tokens/tokens128_raw_f4.json

# target/release/texmo -d data/books3_caps.txt optimize-tokens -n 128 -f 1 --processing=caps --initial-size=108362047381 -i tokens/tokens64_caps_f1.json -o tokens/tokens128_caps_f1.json
# target/release/texmo -d data/books3_caps.txt optimize-tokens -n 128 -f 2 --processing=caps --initial-size=108362047381 -i tokens/tokens64_caps_f2.json -o tokens/tokens128_caps_f2.json
# target/release/texmo -d data/books3_caps.txt optimize-tokens -n 128 -f 4 --processing=caps --initial-size=108362047381 -i tokens/tokens64_caps_f4.json -o tokens/tokens128_caps_f4.json

# target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 128 -f 1 --processing=caps,words --initial-size=108362047381 -i tokens/tokens64_capswords_f1.json -o tokens/tokens128_capswords_f1.json
# target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 128 -f 2 --processing=caps,words --initial-size=108362047381 -i tokens/tokens64_capswords_f2.json -o tokens/tokens128_capswords_f2.json
# target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 128 -f 4 --processing=caps,words --initial-size=108362047381 -i tokens/tokens64_capswords_f4.json -o tokens/tokens128_capswords_f4.json

