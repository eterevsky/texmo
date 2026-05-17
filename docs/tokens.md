# Tokens and tokenization

> **Note.** Tokenization is currently parked. The architecture-search work
> in this repo deliberately trains directly on bytes / bit-prefixes
> (see the [README](../README.md) for the reasoning). This doc captures
> the tokenization machinery that exists in the repo for later.

## Training data

The training corpus is a set of books from
[The Pile](https://pile.eleuther.ai/) (~100 GB). All texts are
concatenated into a single file. Text is encoded in UTF-8 with `\n` as
line ends and `\n\n` as paragraph separators.

## Token sets

The tokenization is designed to work with token sets that could have
both small (<256) and big number of tokens.

- In a typical token set each token stands for a byte sequence. Not
  every byte has to have a token (otherwise the number of tokens would
  necessarily be ≥256). A byte in the input text that can't be covered
  by a normal token is represented by 3 token sequence: `\x10` and two
  hexadecimal digits (`0`..`f`). For this to work the token set has to
  have at least 17 tokens, including `\x10` and 16 hexadecimal digits.

- Alternatively bytes not covered by the token set could be encoded as
  bit sequences with bits represented by two other markers: `\x11` for
  0 and `\x12` for 1. In experiments these kinds of token sets ended
  up always producing less optimal tokenizations than other options.

- In token sets with 2, 4 or 16 tokens each token stands for 1, 2 or 4
  bits in the input text, thus encoding a single byte as 8, 4 or 2
  tokens.

- A token set with 8 tokens contains 4 2-bit sequences and 4 tokens
  standing for the most common 4-bit sequences (aligned to 4 bits):
  0001, 0010, 0110, 0111.

## Pre-processing the text before tokenization

Before splitting the text into tokens, it could be processed in 2 ways
to simplify extracting words as tokens:

1. Encode capital letters. This allows reusing the same tokens for
   capitalized and lower-case versions of the same word.

  - A word (a sequence of Unicode letters) with an uppercase first
    letter and lowercase subsequent letters is encoded as `\x14` +
    letters converted to lowercase.
  - A word written in all uppercase letters is encoded as `\x15` +
    letters converted to lowercase.

2. Add a word-end marker `\x16` at the end of each word (sequence of
   letter characters) and drop single spaces appearing between words.
   More formally:
   a) A sequence `letter` `non-letter` is converted to `letter` `\x16`
      `non-letter`.
   b) A sequence `letter` `\x16` `" "` `letter` is converted to `letter`
      `\x16` `letter`.

Both these conversions together allow using one token per word in a
typical text. Without these processing stages one of two things will
happen: either a token would include word + following or preceding
space — and this token wouldn't be used for the same word followed or
preceded by punctuation — or each space between words has to be
encoded as an additional token.

Empirically these processing stages increase the average number of
bytes per token. For token sets with 32 tokens the optimal tokenization
executes just the first processing step (encoding capitalized words).
For token sets with 64 or more tokens, the optimal tokenization
involves both processing steps.

## Tokenization algorithm

Given a set of byte sequences as tokens, a naive tokenization algorithm
would always greedily select the longest token matching the current
location in text. This however is not optimal. Consider a text
`abcdef` and tokens `ab` and `bcdef` (+ single letters). If token `ab`
is selected in the beginning of the text, then the rest of the text
will be encoded with one token per letter. The optimal tokenization in
this case would be: `a` + `bcdef`.

The optimal tokenization could be achieved using a dynamic programming
algorithm: an array is maintained with the optimal number of tokens up
to any given position in text. In each new position all the possible
end tokens are tried, choosing the one that results in the lowest
possible number of tokens up to a given location.

To optimize this algorithm we need a way to quickly iterate through all
possible tokens ending at a given position in the text. This is done
through two tricks:

1. Each token maintains a pointer to the longest other token which is
   a suffix of the current token. Using this, once we know the
   *longest* token at the current position, we could iterate through
   other matching tokens as a linked list.

2. Before scanning the text we construct a finite-state machine for
   different "interesting" suffixes. We include one state for each
   string which is a prefix of any token in our set. Each such state
   includes a pointer to the longest token which is a suffix of the
   string in the current state. Each state also includes an array of
   256 next states, depending on the next input byte.

It is easy to prove that this finite-state machine would be able to
always produce the longest token matching the current suffix.

This algorithm produces a provably optimal tokenization in terms of
the number of tokens. The Rust implementation can scan 50-100 MB of
text per second in a single thread (depending on the number of tokens
in the token set).

## Optimizing the set of tokens

The set of tokens is optimized roughly following the BPE algorithm:

1. The set of tokens is initialized either with a smaller token set or
   with a set of mandatory tokens (`\x10`, `0`..`9`, `a`..`f` if we
   fall back to hexadecimal encoding of missing bytes).

2. We tokenize the whole corpus with the current set of tokens,
   counting the occurrences of token bigrams (pairs of subsequent
   tokens) and non-token characters encoded as token sequences. The
   most common bigram or non-token is added as a new token. (The count
   of non-tokens is multiplied by the number of tokens in its
   encoding − 1.)

3. If the number of tokens in the set is greater than the number of
   tokens we are aiming for, search for a token we could remove from
   the set without increasing the number of tokens too much.

   a. Tokenize the dataset with the new set of tokens and iterate over
      tokens in the order of increasing number of occurrences of the
      token in the text.

   b. For each token, try removing it, re-tokenize the text, count the
      text and if it's lower than the number of the tokens before step
      2, then remove this token and go back to step 2.

4. After the last iteration in which no token-removal decreased the
   number of tokens in the tokenization, stop.

## Implementation

The optimization algorithm is implemented in Rust, since it requires
high performance and benefits from parallelization. On a MacBook Pro
working in 10 threads it scans 800-900 MB of input data per second,
resulting in a single tokenization of all the training corpus in
around 2 minutes.

To run the optimization:

```
texmo -d <training data> optimize-tokens -n <desired number of tokens> [-i <starting token set>] -o <output token set> [--fallback2]
```

If `--fallback2` is given and no starting token set is provided, the
algorithm encodes non-token characters as 8 bits, using `\x11` and
`\x12` characters. Otherwise it encodes them as hexadecimal, using
`\x10` and two hexadecimal digits.
