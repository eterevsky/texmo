#!/bin/sh

cargo build --release

target/release/texmo -d data/books3.txt optimize-tokens -n 128  --initial-size=108362047381 -i tokens/tokens64_raw16.json -o tokens/tokens128_raw16.json --processing=
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 128  --initial-size=108362047381 -i tokens/tokens64_caps16.json -o tokens/tokens128_caps16.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 128  --initial-size=108362047381 -i tokens/tokens64_capswords16.json -o tokens/tokens128_capswords16.json --processing=caps,words

