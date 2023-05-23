#!/bin/sh

cargo build --release

target/release/texmo -d data/books3_caps.txt optimize-tokens -n 16 --fallback=4 --initial-size=108362047381 -o tokens/tokens16_caps4.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 16 --fallback=4 --initial-size=108362047381 -o tokens/tokens16_capswords4.json --processing=caps,words

target/release/texmo -d data/books3.txt optimize-tokens -n 32 --initial-size=108362047381 -i tokens/tokens16_raw4.json -o tokens/tokens32_raw4.json --processing=
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 32 --initial-size=108362047381 -i tokens/tokens16_caps4.json -o tokens/tokens32_caps4.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 32 --initial-size=108362047381 -i tokens/tokens16_capswords4.json -o tokens/tokens32_capswords4.json --processing=caps,words

target/release/texmo -d data/books3.txt optimize-tokens -n 64 --initial-size=108362047381 -i tokens/tokens32_raw4.json -o tokens/tokens64_raw4.json --processing=
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 64 --initial-size=108362047381 -i tokens/tokens32_caps4.json -o tokens/tokens64_caps4.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 64 --initial-size=108362047381 -i tokens/tokens32_capswords4.json -o tokens/tokens64_capswords4.json --processing=caps,words


