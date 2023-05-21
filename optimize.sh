#!/bin/sh

cargo build --release

target/release/texmo -d data/books3.txt optimize-tokens --fallback2 -n 32  --initial-size=108362047381 -o tokens/tokens32_raw2.json --processing=
target/release/texmo -d data/books3.txt optimize-tokens -n 32  --initial-size=108362047381 -o tokens/tokens32_raw16.json --processing=
target/release/texmo -d data/books3_caps.txt optimize-tokens --fallback2 -n 32  --initial-size=108362047381 -o tokens/tokens32_caps2.json --processing=caps
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 32  --initial-size=108362047381 -o tokens/tokens32_caps16.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens --fallback2 -n 32  --initial-size=108362047381 -o tokens/tokens32_capswords2.json --processing=caps,words
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 32  --initial-size=108362047381 -o tokens/tokens32_capswords16.json --processing=caps,words

target/release/texmo -d data/books3.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_raw2.json -o tokens/tokens64_raw2.json --processing=
target/release/texmo -d data/books3.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_raw16.json -o tokens/tokens64_raw16.json --processing=
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_caps2.json -o tokens/tokens64_caps2.json --processing=caps
target/release/texmo -d data/books3_caps.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_caps16.json -o tokens/tokens64_caps16.json --processing=caps
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_capswords2.json -o tokens/tokens64_capswords2.json --processing=caps,words
target/release/texmo -d data/books3_caps_words.txt optimize-tokens -n 64  --initial-size=108362047381 -i tokens/tokens32_capswords16.json -o tokens/tokens64_capswords16.json --processing=caps,words

