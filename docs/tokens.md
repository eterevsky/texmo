# Tokens and tokenization

This is the **machinery** reference: how a tokenset is represented on
disk, which tokenizer implementation reads it, how text is turned into
token ids, and how new tokensets are built.

It is not the record of *which* tokensets exist. Custom tokensets are
central to the search — `tokens.32.fold`, `tokens.{32,64}.shift`,
`tokens.{32,64,128,256}.hexbpe` and `tokens.128.fold` occupy much of
the Pareto frontier — and the families, their design rationale, their
lossy-vs-lossless accounting and their search neighbor edges all live
in [`io.md`](io.md), which is kept current. Read that first; come here
for the mechanics.

## Training data

The training corpus is `books3` from
[The Pile](https://pile.eleuther.ai/) (`config.DATA`, by default
`data/books3.txt`): all texts concatenated into a single file, encoded
in UTF-8 with `\n` as line ends and `\n\n` as paragraph separators.
Byte-frequency tables used by the tokenset generators (`freq.txt`,
`freq_capswords2.txt`) are produced from it by the Rust `count-bytes`
command.

## What a tokenset is

A tokenset maps byte strings to integer ids and back. The design goal
is one mechanism that works from 32 tokens to 256,000, so the pieces
below are all optional and combine freely:

- **Tokens** — a token normally stands for a byte sequence. Not every
  byte needs its own token, which is what lets a set be smaller than
  256.
- **Numbered ("ext") tokens** — tokens with no string of their own,
  written as bare integers in the JSON. Markers and escapes (a shift
  marker, a hex escape, Gemma's `<bos>`) are numbered tokens: control
  *characters* belong to text→text preprocessing, numbered tokens to
  constructs that live inside the tokenizer.
- **Sequences** — a byte string spelled as a fixed list of tokens.
  This is how a set covers bytes it has no token for.

Several fallback conventions are in use for the bytes a set does not
cover directly; which one a set uses is a property of how it was
built, not a runtime switch:

- **Hex escape** — `\x10` followed by two hexadecimal digits
  (`0`..`f`). Needs at least 17 tokens in the set.
- **Nibble pairs** — the hexbpe schema reserves ids 0–15 as nibble
  tokens, so any uncovered byte spells as exactly two of them. No
  escape marker at all.
- **Bit escapes** — uncovered bytes spelled as 8 bits with `\x11` for
  0 and `\x12` for 1. Historical: in every experiment this produced
  worse tokenizations than the alternatives.
- **Shift markers** — a numbered marker token plus a related host
  token (`A` = marker + `a`), used by the shift sets.
- **Lossy folding** — several bytes collapse onto one token and the
  difference is simply lost, paid for at eval time by a stored
  residual charge. See [`io.md`](io.md) for that accounting.

Sub-byte "tokensets" are **not** JSON files. Splitting a byte into
1/2/4-bit chunks is handled by built-in tokenizers registered under
`bits.1`, `bits.2`, `bits.4` and `bits.8` / `bytes`
(`tokens/bits_tokenizer.py`), which are pure array reshapes with no
vocabulary to store.

## The tokenset JSON

Files live in the directory passed as `--tokens-dir` (default
`tokens/`) and are named after their spec form:
`tokens.32.hexbpe` → `tokens/tokens.32.hexbpe.json`. The legacy
underscore basename (`tokens32_capswords_hexbpe`) is still accepted by
`registry._normalize_name` for old callers; the sets under
`tokens/asoif/` and `tokens/pride/` (small-corpus experiments) still
use it.

`TokenSet.from_json` (`tokens/tokenset.py`) reads:

| field | meaning |
| --- | --- |
| `type` | dispatch key for the tokenizer implementation — `fold`, `shift`, `shift_bucket`, `hexbpe`, `bytes`, ... |
| `processing` | name of the text→text pass to apply first (`raw`, `capswords`, `capswords2`, `gemma`) |
| `tokens` | ordered list; a **string** is a normal piece, a **list of ints** a raw byte-fallback piece, a bare **int** a numbered/ext token |
| `sequences` | `{string, tokens}` entries spelling a byte string as a fixed token list |
| `merges` | either `[left_id, right_id, merged_id]` int triples (converted SentencePiece vocabs) or `[left_string, right_string]` pairs in rank order (hexbpe) |
| `algorithm` | `dp` (default) or `bpe` |
| `stats` | `bytes_per_token` (against **raw** corpus bytes), `scanned_bytes`, `total_tokens`, `residual_bits_per_byte`, `extra_weights` |
| `head_freq` | fold sets only: P(head byte \| token), keyed by the token's character |
| `specials` | control tokens with a private-use char so id streams round-trip through plain strings |
| `priority_tokens` | ids matched verbatim before the BPE bulk (pieces with no merge path) |
| `add_bos`, `bos_id`, `eos_id`, `pad_id`, `unk_id` | pretrained-vocabulary metadata |

`stats.bytes_per_token` and `stats.residual_bits_per_byte` are what
`TokenSet.byte_loss` uses to convert per-token loss to bits per byte;
`stats.extra_weights` is the tokenset's parameter surcharge. Both are
explained in [`io.md`](io.md).

## Tokenizer implementations

`registry.get_tokenizer(name)` loads the JSON once per process and
picks an implementation from `type` and `algorithm`:

- **`Tokenizer`** (`tokens/tokenizer.py`) — the generic dynamic-
  programming tokenizer over tokens + sequences. The default; also what
  `hexbpe` sets use at sampling time, and what handles `type: shift`
  (shift-64 is an ordinary tokens+sequences set — every byte has
  exactly one spelling, so the DP has nothing to choose and reproduces
  the fixed encoding).
- **`FoldTokenizer`** — `type: fold`. Byte → token is a 256-entry
  table lookup; the generic decoder can't represent many-to-one groups.
- **`ShiftBucketTokenizer`** — `type: shift_bucket`. Part lossless
  (the shift pairs), part forgetting (the uniform buckets); encoding is
  still a plain 256-entry lookup, since every byte has exactly one
  spelling.
- **`BpeTokenizer`** — `algorithm: bpe` for converted SentencePiece
  vocabularies. Runs the merge loop; the DP tokenizer's dense suffix
  automaton would be prohibitively large at 256k pieces.
- **`HexBpeTokenizer`** — builder-exact BPE by merge rank. Kept for
  scripts that must reproduce the builder's counts; the sampler
  deliberately uses the generic DP instead (16x faster, 0.5% more
  compact — see [`io.md`](io.md)).
- **`BitsTokenizer{1,2,4}` / `BytesTokenizer`** — the built-in
  sub-byte chunkers, registered eagerly and never file-backed.

## Pre-processing the text before tokenization

Before splitting the text into tokens, it can be run through a
text→text pass (`tokens/processing.py`, selected by the tokenset's
`processing` field) that makes words easier to capture as single
tokens. `raw` does nothing.

**`capswords`** does two things:

1. Encode capital letters, so the same token serves the capitalized
   and lowercase forms of a word.

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

Together these allow one token per word in typical text. Without them
one of two things happens: either a token includes the word plus a
following or preceding space — and then goes unused when the same word
is next to punctuation — or every inter-word space costs an extra
token.

**`capswords2`** is `capswords` plus `\x17`, a next-letter-uppercase
marker for the mixed-case words (`iPhone`, `McDonald`) that `capswords`
passes through verbatim, forcing every tokenset to carry uppercase
letters it otherwise would not need. `\x14` survives alongside it: a
leading capital is highly predictable after `". "` while a mid-word one
is not, and merging the two would blur a signal the model can use.
`capswords2` is written as an explicit state machine rather than a
regex split so it mirrors `src/processing.rs` character for character —
the Rust generator and the Python tokenizer must agree exactly or
tokensets decode to garbage.

Empirically these stages increase the average number of bytes per
token. For 32-token sets the optimal tokenization runs only the
capitalization step; for 64 or more tokens, both. Single-character-
token sets need neither (see shift-32 in [`io.md`](io.md)) — the word
and case markers exist *for multi-character tokens*.

`gemma` is the converted-vocabulary quirk pass, not a compression aid:
input U+2581 is replaced by a space before encoding, reproducing the
reference tokenizer's space/U+2581 conflation.

## Tokenization algorithm

Given a set of byte sequences as tokens, a naive tokenization algorithm
would always greedily select the longest token matching the current
location in text. This however is not optimal. Consider a text
`abcdef` and tokens `ab` and `bcdef` (+ single letters). If token `ab`
is selected in the beginning of the text, then the rest of the text
will be encoded with one token per letter. The optimal tokenization in
this case would be: `a` + `bcdef`.

The optimal tokenization is achieved with dynamic programming: an array
holds the optimal number of tokens up to any given position in the
text. At each new position all possible end tokens are tried, choosing
the one that results in the lowest total token count up to that point.

To make this fast we need a way to quickly iterate through all possible
tokens ending at a given position. Two tricks do it:

1. Each token maintains a pointer to the longest other token which is
   a suffix of the current token. Once we know the *longest* token at
   the current position, we can walk the other matching tokens as a
   linked list.

2. Before scanning the text we construct a finite-state machine over
   the "interesting" suffixes: one state per string that is a prefix of
   any token in the set, each carrying a pointer to the longest token
   that is a suffix of that state's string, plus an array of 256 next
   states indexed by the next input byte.

It is easy to prove that this finite-state machine always produces the
longest token matching the current suffix.

The algorithm gives a provably optimal tokenization *in terms of the
number of tokens*. Note that minimizing token count is not the same as
minimizing loss, and the literature finds the difference immaterial at
these vocabulary sizes (see the hexbpe discussion in
[`io.md`](io.md)). The Rust implementation scans 50–100 MB of text per
second in a single thread, depending on vocabulary size.

Decoding is the inverse walk (`Decoder` in `tokens/tokenizer.py`),
followed by the inverse of the processing pass. It is defensive: a
*sampled* id stream can misuse a marker or escape (a shift token
followed by a non-letter), so when nothing decodable starts at the
current position the head token is dropped and the rest reprocessed.

## Building a tokenset

Two generators, both in Rust (`src/`, built with `cargo`), because the
work is corpus-scale and parallelizes well.

### `optimize-tokens` — the BPE-style optimizer

    texmo optimize-tokens -d <training data> -n <ntokens> \
        [-i <starting token set>] [-o <output token set>] \
        [--literals dist8] [--processing capswords2]

1. The set is initialized either from a smaller token set or from the
   mandatory tokens the chosen literal encoding needs (`\x10` and
   `0`..`9`, `a`..`f` for the hex fallback).

2. The whole corpus is tokenized with the current set, counting token
   bigrams (pairs of subsequent tokens) and non-token characters
   encoded as token sequences. The most common bigram or non-token is
   added as a new token. (The count of a non-token is multiplied by the
   number of tokens in its encoding − 1.)

3. If the set is now larger than the target, look for a token that can
   be removed without increasing the tokenization length too much:

   a. Tokenize the dataset with the new set and iterate over tokens in
      order of increasing occurrence count.

   b. For each, try removing it, re-tokenize, and if the total token
      count is lower than it was before step 2, drop the token and go
      back to step 2.

4. Stop after the first iteration in which no removal decreased the
   token count.

`--literals` selects the fallback encoding for uncovered bytes:
`hex` (the `\x10` + two hex digits form), `bits1`/`bits2`/`bits4` (the
`\x11`/`\x12` bit escapes), `dist2`/`dist4`/`dist8` (a single
unknown-byte token charged as 2/4/8 normal tokens; `dist8` is the
default), or `all` when every byte has its own token.

### `build-hexbpe` — the current schema

    texmo build-hexbpe -d <training data> -n <ntokens> -o <out.json>

Reserves 16 nibble tokens, then greedily interleaves single-byte
selections and BPE merges in one currency (emitted tokens saved). It
samples the raw corpus and applies `capswords2` on the fly, so no
preprocessed copy of the corpus is needed. The schema and the reasons
for it are in [`io.md`](io.md#hexbpe-tokensets-one-schema-for-every-size).

The Python-side generators for the fold / shift families —
`scripts/make_fold32.py`, `make_shift32.py`, `make_shift64.py`,
`make_bucket.py` — compute their mappings, stored frequencies and
residuals from the byte-count tables rather than by corpus search.

Supporting Rust commands: `count-bytes` / `count-chars` (frequency
tables), `process` (write a preprocessed copy of the corpus),
`convert-tokens`, `optimize` and `optimize-all` (sweep a range of
sizes into a tokens directory).

On a 10-thread laptop the Rust scanner reads 800–900 MB/s, so one full
tokenization pass over the training corpus takes about two minutes.

## Converting a HuggingFace tokenizer

`uv run python -m texmo.tokens.convert_hf <tokenizer.json> <out.json>`
turns a SentencePiece-BPE `tokenizer.json` (as shipped with Gemma /
RecurrentGemma) into the tokenset JSON above. The conversion choices:

- Pieces are rewritten with the SentencePiece whitespace marker U+2581
  replaced by a literal space, so tokens carry their spaces directly
  and decoding is plain concatenation. Safe because no piece contains a
  real space; the runtime keeps the conflation quirk as processing
  `gemma`.
- `<0xXX>` byte-fallback pieces become raw single-byte tokens, stored
  as one-element int lists.
- Control tokens (HF added tokens with `special=true`) become numbered
  tokens, listed in `specials` with a private-use char (U+F0000 + id).
- Non-special added tokens (newline runs, space runs, HTML tags) keep
  their strings and go in `priority_tokens`: the encoder must match
  them with a longest-match scan *before* running BPE, since some have
  no merge path.
- `merges` is stored as id triples rather than string pairs, because
  converted pieces may contain spaces and several distinct ids can
  share one byte string.

The resulting `stats` block is synthetic (~4 bytes/token); it only
seeds the sampler's chunk sizing, which self-corrects.
