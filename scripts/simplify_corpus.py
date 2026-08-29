"""LLM-simplified variants of the User/Bot SODA training corpus.

Milestone 3 of the tiny chatbot program (`docs/roadmap.md`,
"Tranches"): the question is whether deliberately dumbed-down dialog
data trains a better nano-sized chatbot than natural SODA at equal
bytes, measured on the milestone-2 scripted-examiner eval. This
script produces the dumbed-down corpora.

Three target corpora, all in the same `Name: utterance` format as
`data/soda/soda_train_ub.txt`:

    s1  User turns ORIGINAL, Bot turns SIMPLIFIED -- same rough
        meaning, at most 2 short sentences, A2 (beginner learner)
        vocabulary.
    s2  User turns ORIGINAL, Bot turns TRIVIAL -- one everyday
        phrase carrying only the gist ("Yes.", "I don't know.",
        "That sounds fun.").
    s3  User turns SIMPLIFIED (the s1 Bot rule, applied to the User
        side), Bot turns TRIVIAL.
    s4  s3, except that a Bot turn answering a yes/no question says
        "Yes." or "No." -- see "Polar answers" below.

The first three differ only in *selection*, so generation runs once
and renders three times:

    uv run --with pyarrow python scripts/simplify_corpus.py generate \\
        --n-dialogs 1000 [--start 0] [--parallel 4]
    uv run --with pyarrow python scripts/simplify_corpus.py render \\
        --variant s1 [--n-dialogs 30000]

`generate` asks the LLM for BOTH rewrites of every turn in one
request and appends one JSONL record per dialog to
`data/simplified/simplify-log.jsonl`; `render` selects fields out of
that log per the table above. One generation pass, three corpora, and
a new variant is a render away rather than a regeneration.

`--n-dialogs` on `render` truncates the log to its first N records --
its *file* order, which is the order they were rendered in. A log that
has grown since an earlier corpus was written can still reproduce it,
and a new variant can cover exactly the same dialogs.

## Polar answers (s4)

Measured on the milestone-2 eval, an 8k model trained on s3 is 92%
grammatical but only 26% responsive -- exactly the floor of a null
model drawing its own top-20 phrases at random. The simplification
prompt steered *away* from bare "Yes."/"No." (it asked for varied
phrases), so the one place where a purely structural cue determines a
content-appropriate short answer is underused.

s4 puts it back. A User turn that ends in an auxiliary-initial
question ("Do you...", "Are you...", optionally after a short lead-in:
`is_yes_no_question`) is a cue a tiny model can plausibly detect, and
the right answer to it is one of two words. The `classify` subcommand
asks a second, smaller LLM what the ORIGINAL Bot reply meant as an
answer to that question --

    uv run python scripts/simplify_corpus.py classify \\
        [--n-dialogs 30000] [--parallel 16]

-- where "as an answer" is read broadly: a request or an offer
("Can you help me?", "Do you want some?") is a yes/no question for
this purpose, granted or refused, because a tiny model cannot tell it
from a true polar question either and "Yes." answers all of them.
Narrowing the classifier to true polar questions was tried first and
labelled only 20% of the detected turns, which is too thin a signal to
learn from; the broad reading labels ~80%.

The subcommand logs `{dialog_idx, turn, label}` (yes / no / other) to
`data/simplified/polar-log.jsonl`, resumably, keyed by that pair.
`render --variant s4` then writes "Yes." / "No." for the labelled
turns and the s3 trivial phrase everywhere else, so s4 minus s3 is
exactly the polar answers. A classification that failed is recorded as
`other`, which is the s3 behaviour: no hole, no retry.

The substitutes are bare and unvaried on purpose. "Yes, I do." would
be more natural and would blunt the point, which is a single cue with
a single answer to learn.

## Source and dialog identity

Dialogs come from `data/soda/train.parquet` through the *same* filter
`scripts/make_soda_txt.py --names user-bot` applies: exactly two
distinct speakers, no length mismatch between `dialogue` and
`speakers`, no blank utterance. `dialog_idx` counts the surviving
dialogs in file order, so it indexes `data/soda/soda_train_ub.txt`
directly -- dialog 7 here is dialog 7 there, and `--start` /
`--n-dialogs` slice the same sequence the baseline corpus was built
from.

That filter is re-implemented here rather than imported:
`make_soda_txt.py` imports pyarrow unconditionally, and both this
module's tests and its `render` subcommand must import without it.
The rules are small and must stay in step with that script.

pyarrow is not a texmo dependency and must not become one, hence
`uv run --with pyarrow` above (only `generate` reads parquet; `render`
runs without it).

## The request

One request per dialog, temperature 0.3, with the whole dialog
numbered by turn. The reply must be a strict JSON array with one
object per turn:

    {"i": 3, "side": "Bot", "simple": "...", "trivial": "..."}

`trivial` is required on Bot turns and null on User turns -- s3 takes
the User side from `simple`, so no User turn ever needs a trivial
phrase. The array is validated hard (every index covered, sides
matching the source, non-empty strings, Bot trivials present); a
failure gets one retry with an error-correcting suffix and is then
recorded as `{dialog_idx, error}` and skipped. Both prompt examples
are worked end to end; the second one deliberately holds a long,
ornate Bot turn (the 2-sentence reduction is only interesting where
there was something to cut) and a Bot turn that *asks* something (see
`_PROMPT_VERSION`).

## Throughput

`--parallel N` sets llama-server's `-np N` (N decode slots) *and*
drives N concurrent in-flight requests from a thread pool, which is
what actually fills them; one at a time leaves most of the GPU idle.
`--parallel-requests` splits the two, for a `--url` endpoint whose
slot count is somebody else's business. Records are written by the
single collector loop as futures complete, so the log is in
completion order -- `render` sorts by `dialog_idx`.

Measured 2026-08-27 on a 32GB consumer GPU already carrying the
search server (~20GB), gemma-4-12b-it-Q6_K at `-c 16384 -np 4`:
~1.2-2.2s per dialog, i.e. ~7-12h for 20k dialogs, and the card sits
at ~31.9/32.6GB. `-np 8` (at `-c 24576`, to keep 3k tokens per slot)
bought no throughput at all -- 4 slots already saturate a GPU that is
sharing -- so raising `--parallel` only buys OOM risk here.

Resumable: `dialog_idx`es already present in the log (successes and
failures alike) are skipped, so an interrupted run resumes by
re-running the same command.

Every llama.cpp request carries
`{"chat_template_kwargs": {"enable_thinking": false}}`: 2026 instruct
models reason by default and return empty `content` otherwise.

## Degenerate output

s2 and s3 are only worth training on if the trivial phrases vary. The
`generate` summary therefore prints the mean character counts per
side and variant, the number of distinct trivial phrases, and the
share held by the most common one -- warning loudly above 40%, since
an all-"Yes." corpus would teach a model exactly one thing.
"""
import argparse
import collections
import concurrent.futures
import json
import logging
import os
import re
import sys
import threading
import time

import chat_eval as ce
import dialog_harness as dh
import requests

# Only `generate` reads parquet. The import is guarded so that
# `render`, and the test module, work in an environment without
# pyarrow (it is deliberately not a texmo dependency).
try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by the CLI, not tests
    pq = None

_DEFAULT_SOURCE = os.path.join("data", "soda", "train.parquet")
_DEFAULT_LOG = os.path.join("data", "simplified", "simplify-log.jsonl")
_DEFAULT_POLAR_LOG = os.path.join("data", "simplified", "polar-log.jsonl")
_CORPUS_DIR = os.path.join("data", "soda")
_DEFAULT_MODEL = "C:/Users/oleg/models/gemma-4-12b-it-Q6_K.gguf"
_DEFAULT_POLAR_MODEL = "C:/Users/oleg/models/Qwen3-8B-Q6_K.gguf"

_COLUMNS = ["dialogue", "speakers"]
_BATCH_SIZE = 4096
_USER, _BOT = "User", "Bot"
_SIDES = (_USER, _BOT)

# Bumped whenever the prompt changes in a way that changes the data;
# recorded with every dialog so a corpus always says what made it.
# v2 (2026-08-27): the only repeated validation failure in the 44-dialog
# trial was a Bot turn that *asks* something -- Gemma answered
# `"trivial": null`, twice, for "And how much was in your account
# then?". The rules now say a Bot question gets a short question back
# and that no Bot turn may be null, and example 2 ends on one.
_PROMPT_VERSION = 2

_DEFAULT_PARALLEL = 4
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 2048
# Total across the slots: llama-server splits `-c` by `-np`, so 4
# slots of 4096 tokens each. A dialog request is ~1-2k tokens.
_DEFAULT_CTX = 16384
_DEFAULT_STARTUP_TIMEOUT = 600
_DEFAULT_REQUEST_TIMEOUT = 600.0

# The classification pass is the opposite shape: a cached ~350-token
# system prefix, two short lines and a one-word answer. Many small
# slots, then; `-c` is the total llama-server splits across them, so
# 16 slots get 1024 tokens each -- far more than a pair ever needs.
_DEFAULT_POLAR_PARALLEL = 16
_DEFAULT_POLAR_CTX = 16384
# One word, with room for an empty <think> block a template may leak.
_DEFAULT_POLAR_MAX_TOKENS = 32

# Loud warning threshold for the most common trivial phrase.
_TRIVIAL_TOP_WARN = 0.40

# House rule for llama.cpp seats: without it 2026 instruct models burn
# the budget in <think> and return empty content.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

# A whitespace run containing a newline: an utterance must stay on one
# line for `Name: text` to parse back.
_NEWLINE_RUN = re.compile(r"\s*\n\s*")
# Last-resort extraction when the model wraps the array in prose.
_ARRAY_RE = re.compile(r"\[.*\]", re.S)

_SYSTEM_PROMPT = """\
You rewrite short conversations into very simple English, for a \
beginner learner (CEFR level A2). You are given a conversation \
between "User" and "Bot", one turn per line, numbered from 0.

For EVERY turn write a "simple" version:
- the same rough meaning and intent as the original;
- at most 2 short sentences;
- only words a beginner learner knows: replace rare words with \
common ones, and drop the details that would need rare words;
- keep proper names, and keep the kind of turn (a question stays a \
question, a greeting stays a greeting).

For every "Bot" turn ALSO write a "trivial" version: ONE very short \
everyday phrase that keeps only the gist of the turn, such as \
"Yes.", "No.", "I don't know.", "I like it.", "Sure, why not?", \
"That sounds fun.", "My name is Bot.", "Not really.", "Me too.", \
"Thank you!". Pick the phrase that actually fits that turn, and vary \
them through the conversation -- answering "Yes." to everything is \
wrong. When the Bot turn asks something, its trivial version is a \
short question: "Why?", "How much?", "Really?", "Yes, why?". EVERY \
"Bot" turn needs a trivial phrase, however short or dull the turn \
already is; never write null for a "Bot" turn. For "User" turns \
write null instead.

Every string must be a single line of plain text: no line breaks, no \
surrounding quotation marks, no speaker labels, no markdown.

Reply with STRICT JSON and nothing else -- no fences, no explanation \
before or after: an array holding exactly one object per turn, in \
order,

[{"i": <turn number>, "side": "User" or "Bot", "simple": "...", \
"trivial": "..." or null}]

Example 1.

Conversation:
0 User: Hey Marcus, are you coming to the potluck on Saturday?
1 Bot: I'd love to, but I promised my sister I'd help her move \
apartments that afternoon.
2 User: That's a shame. Could you drop by afterwards?
3 Bot: If we finish before eight I'll swing by. I'll text you when \
we're done with the last boxes, and I'll bring the leftover lasagna \
so it doesn't go to waste.

Answer:
[{"i": 0, "side": "User", "simple": "Hi Marcus, are you coming to \
the party on Saturday?", "trivial": null},
{"i": 1, "side": "Bot", "simple": "I want to come. But I must help \
my sister move.", "trivial": "Sorry, I can't."},
{"i": 2, "side": "User", "simple": "That is sad. Can you come \
later?", "trivial": null},
{"i": 3, "side": "Bot", "simple": "If we finish early, I will come. \
I will bring food.", "trivial": "Maybe later."}]

Example 2.

Conversation:
0 User: What did you think of the museum?
1 Bot: Honestly, the contemporary wing was underwhelming -- the \
curation felt scattered, and half the installations were roped off \
for maintenance. The medieval manuscripts upstairs, though, were \
extraordinary; I spent nearly an hour in front of one illuminated \
psalter.
2 User: I had no idea you cared about that sort of thing.
3 Bot: Have you ever been up to the manuscript room yourself?

Answer:
[{"i": 0, "side": "User", "simple": "What did you think about the \
museum?", "trivial": null},
{"i": 1, "side": "Bot", "simple": "The new art was boring. But I \
loved the old books upstairs.", "trivial": "I liked it."},
{"i": 2, "side": "User", "simple": "I did not know you liked that.", \
"trivial": null},
{"i": 3, "side": "Bot", "simple": "Have you seen the old books \
upstairs?", "trivial": "Have you been there?"}]"""

_USER_PROMPT = """\
Conversation ({n} turns, numbered 0 to {last}). Return exactly {n} \
objects, one per turn, in order.

{dialog}"""

_RETRY_SUFFIX = """\
Your previous answer was not usable ({error}). Reply again with the \
JSON array only: exactly {n} objects, one per turn, with the keys \
"i", "side", "simple", "trivial". No fences, no prose."""


# --------------------------------------------------------- source text


def collapse(text: str) -> tuple[str, bool]:
    """One-line, stripped form of `text`, and whether it changed.

    The corpus format puts one utterance on one line, so a newline
    anywhere in an utterance -- from SODA or from the LLM -- has to
    go. The flag is what the callers count.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" not in text:
        return text, False
    return _NEWLINE_RUN.sub(" ", text), True


class SourceStats:
    """Tallies over the parquet rows read by `iter_dialogs`."""

    def __init__(self):
        self.rows = 0
        self.dialogs = 0
        self.skip_mismatch = 0
        self.skip_empty_dialog = 0
        self.skip_empty_utterance = 0
        self.skip_speakers = 0
        self.collapsed = 0


def speaker_map(speakers, stats: SourceStats) -> dict | None:
    """{original name: "User"/"Bot"}, or None if the row can't map.

    First-appearance order, so `speakers[0]` -- who opens the dialog
    -- is always "User". Mirrors `make_soda_txt.py`.
    """
    if any(speaker is None for speaker in speakers):
        stats.skip_empty_utterance += 1
        return None
    # dict.fromkeys de-duplicates while keeping first-appearance
    # order; a set would lose the order the mapping depends on.
    distinct = list(dict.fromkeys(speakers))
    if len(distinct) != len(_SIDES):
        stats.skip_speakers += 1
        return None
    return dict(zip(distinct, _SIDES))


def dialog_turns(dialogue, speakers, stats: SourceStats) -> list | None:
    """`[{"side": "User", "text": ...}, ...]`, or None if unusable.

    None means the row is dropped, and the reason is already counted.
    The rules are exactly `make_soda_txt.py --names user-bot`'s, so
    the surviving dialogs -- and their order -- match
    `data/soda/soda_train_ub.txt`.
    """
    if dialogue is None or speakers is None:
        stats.skip_mismatch += 1
        return None
    if len(dialogue) != len(speakers):
        stats.skip_mismatch += 1
        return None
    if not dialogue:
        stats.skip_empty_dialog += 1
        return None
    rename = speaker_map(speakers, stats)
    if rename is None:
        return None
    turns = []
    for speaker, utterance in zip(speakers, dialogue):
        if speaker is None or utterance is None:
            stats.skip_empty_utterance += 1
            return None
        text, collapsed = collapse(utterance)
        if not text:
            stats.skip_empty_utterance += 1
            return None
        stats.collapsed += collapsed
        turns.append({"side": rename[speaker], "text": text})
    return turns


def iter_dialogs(path: str, start: int, count: int, stats: SourceStats):
    """Yield `(dialog_idx, turns)` for dialogs `[start, start+count)`.

    Streams the parquet file; `dialog_idx` counts dialogs that pass
    the filter, not rows read.
    """
    if pq is None:
        raise RuntimeError(
            "pyarrow is needed to read the source parquet; run this with "
            "`uv run --with pyarrow python scripts/simplify_corpus.py ...`")
    parquet = pq.ParquetFile(path)
    idx = 0
    for batch in parquet.iter_batches(
            batch_size=_BATCH_SIZE, columns=_COLUMNS):
        dialogues = batch.column("dialogue").to_pylist()
        speaker_lists = batch.column("speakers").to_pylist()
        for dialogue, speakers in zip(dialogues, speaker_lists):
            stats.rows += 1
            turns = dialog_turns(dialogue, speakers, stats)
            if turns is None:
                continue
            if idx >= start + count:
                return
            if idx >= start:
                stats.dialogs += 1
                yield idx, turns
            idx += 1


# ------------------------------------------------------------- prompting


def render_numbered(turns: list[dict]) -> str:
    """The dialog as `<i> <side>: <text>` lines, one per turn."""
    return "\n".join(
        f"{i} {turn['side']}: {turn['text']}"
        for i, turn in enumerate(turns))


def build_messages(turns: list[dict],
                   retry_error: str | None = None) -> list[dict]:
    """The chat-completions messages for one dialog.

    `retry_error` appends the error-correcting suffix used for the
    single retry after a validation failure.
    """
    user = _USER_PROMPT.format(
        n=len(turns), last=len(turns) - 1, dialog=render_numbered(turns))
    if retry_error is not None:
        user += "\n\n" + _RETRY_SUFFIX.format(
            error=retry_error, n=len(turns))
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------- validation


def _text_field(obj: dict, key: str, i: int) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"turn {i}: {key!r} is not a non-empty string")
    return value.strip()


def parse_simplification(text: str, turns: list[dict]) -> list[dict]:
    """Validate the reply against the source dialog; raise ValueError.

    Returns the per-turn records `render` consumes:
    `{"side", "original", "simple", "trivial"}`, in source order.
    Everything the array claims is checked against `turns`, because a
    plausible-looking answer for the *wrong* turns is the failure
    mode that would silently corrupt a corpus: the index set must be
    exactly `0..n-1`, the sides must match, `simple` must be a
    non-empty string everywhere and `trivial` on every Bot turn.
    """
    body = ce.strip_fences(text)
    parsed = None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        match = _ARRAY_RE.search(body)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"not JSON: {e}") from e
    if parsed is None:
        raise ValueError("no JSON array in the reply")
    if not isinstance(parsed, list):
        raise ValueError(
            f"expected a JSON array, got {type(parsed).__name__}")
    if len(parsed) != len(turns):
        raise ValueError(
            f"expected {len(turns)} objects, got {len(parsed)}")

    by_index: dict[int, dict] = {}
    for entry in parsed:
        if not isinstance(entry, dict):
            raise ValueError(
                f"array holds a {type(entry).__name__}, not an object")
        index = entry.get("i")
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"turn index {entry.get('i')!r} is not an int")
        if not 0 <= index < len(turns):
            raise ValueError(f"turn index {index} out of range")
        if index in by_index:
            raise ValueError(f"turn index {index} appears twice")
        by_index[index] = entry

    out = []
    for i, turn in enumerate(turns):
        entry = by_index.get(i)
        if entry is None:
            raise ValueError(f"turn {i} is missing")
        side = entry.get("side")
        if side != turn["side"]:
            raise ValueError(
                f"turn {i}: side {side!r}, expected {turn['side']!r}")
        record = {
            "side": side,
            "original": turn["text"],
            "simple": _text_field(entry, "simple", i),
            "trivial": None,
        }
        if side == _BOT:
            record["trivial"] = _text_field(entry, "trivial", i)
        else:
            trivial = entry.get("trivial")
            # A trivial phrase on a User turn is unused (s3 takes the
            # User side from `simple`), but harmless -- keep it rather
            # than fail an otherwise good dialog.
            if isinstance(trivial, str) and trivial.strip():
                record["trivial"] = trivial.strip()
        out.append(record)
    return out


# ------------------------------------------------------------ requests


_local = threading.local()


def thread_session() -> requests.Session:
    """One `requests.Session` per worker thread.

    A Session is not documented as thread-safe, and the pool has N of
    them in flight at once.
    """
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        _local.session = session
    return session


def process_dialog(session, url: str, participant: dict, dialog_idx: int,
                   turns: list[dict], model: str, timeout) -> dict:
    """One dialog to one log record: the simplification, or the error.

    Two attempts: the first plain, the second carrying the validation
    error back to the model. A record with an `error` key is a
    permanent failure that `render` skips.
    """
    t0 = time.perf_counter()
    error = None
    for attempt in range(2):
        messages = build_messages(turns, error if attempt else None)
        try:
            raw, _, _ = ce.ask(session, url, participant, messages, timeout)
            simplified = parse_simplification(raw, turns)
        except ValueError as e:
            error = str(e)
            logging.warning(
                f"dialog {dialog_idx}: {error}; "
                f"{'retrying' if not attempt else 'giving up'}")
            continue
        return {
            "dialog_idx": dialog_idx,
            "turns": simplified,
            "model": model,
            "prompt_version": _PROMPT_VERSION,
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    return {
        "dialog_idx": dialog_idx,
        "error": error,
        "model": model,
        "prompt_version": _PROMPT_VERSION,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


# --------------------------------------------------------- log handling


def read_log(path: str) -> list[dict]:
    """Every record in a log file, in file order; missing file -> []."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"skipping unreadable record in {path}")
    return records


def existing_indices(path: str) -> set:
    """`dialog_idx`es already logged -- failures included.

    A failure is a decision, not a gap: re-running the same command
    resumes rather than re-litigating dialogs the model already
    refused twice.
    """
    return {r["dialog_idx"] for r in read_log(path) if "dialog_idx" in r}


# ------------------------------------------------------------- summary


_TRIM = " .!?\"'"


def trivial_key(text: str) -> str:
    """Normalized form for counting distinct trivial phrases.

    "Yes.", "yes" and "Yes!" are one phrase for the purpose of asking
    whether the model actually varied its answers.
    """
    return " ".join(text.lower().split()).strip(_TRIM)


def summarize(records: list[dict]) -> dict:
    """Character counts per side and the trivial-phrase distribution."""
    chars: dict = {
        side: {"original": [], "simple": [], "trivial": []}
        for side in _SIDES
    }
    trivials = collections.Counter()
    failed = 0
    for record in records:
        if "error" in record:
            failed += 1
            continue
        for turn in record["turns"]:
            bucket = chars[turn["side"]]
            bucket["original"].append(len(turn["original"]))
            bucket["simple"].append(len(turn["simple"]))
            if turn.get("trivial"):
                bucket["trivial"].append(len(turn["trivial"]))
                if turn["side"] == _BOT:
                    trivials[trivial_key(turn["trivial"])] += 1
    top = trivials.most_common(1)
    total = sum(trivials.values())
    return {
        "dialogs": len(records) - failed,
        "failed": failed,
        "chars": {
            side: {
                field: (sum(values) / len(values) if values else None)
                for field, values in fields.items()
            }
            for side, fields in chars.items()
        },
        "turns": {side: len(fields["original"])
                  for side, fields in chars.items()},
        "trivial_total": total,
        "trivial_distinct": len(trivials),
        "trivial_top": top[0][0] if top else None,
        "trivial_top_share": (top[0][1] / total) if top and total else None,
        "trivial_common": trivials.most_common(10),
    }


def _mean(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def format_summary(summary: dict) -> list[str]:
    """The `generate` summary block, degenerate-output warning included."""
    lines = [
        f"dialogs {summary['dialogs']} ok, {summary['failed']} failed",
        "mean chars per turn:",
        "  side  turns  original  simple  trivial",
    ]
    for side in _SIDES:
        chars = summary["chars"][side]
        lines.append(
            f"  {side:<5} {summary['turns'][side]:>6} "
            f"{_mean(chars['original']):>9} {_mean(chars['simple']):>7} "
            f"{_mean(chars['trivial']):>8}")
    share = summary["trivial_top_share"]
    lines.append(
        f"trivial phrases: {summary['trivial_total']} Bot turns, "
        f"{summary['trivial_distinct']} distinct")
    if share is not None:
        lines.append(
            f"  most common: {summary['trivial_top']!r} "
            f"{100 * share:.1f}%")
        lines.append("  top 10: " + ", ".join(
            f"{phrase!r} x{count}"
            for phrase, count in summary["trivial_common"]))
        if share > _TRIVIAL_TOP_WARN:
            lines.append(
                f"*** WARNING: the top trivial phrase covers "
                f"{100 * share:.1f}% of Bot turns (> "
                f"{100 * _TRIVIAL_TOP_WARN:.0f}%). s2/s3 would teach a "
                f"model one phrase; fix the prompt before training on "
                f"this log. ***")
    return lines


# ---------------------------------------------------------- generation


def _server_conf(args) -> dict:
    """The `server` block `dialog_harness` expects, from our flags.

    `-np` is what makes the concurrency real: without the slots the
    server serializes the requests the pool sends. No bare `-fa` --
    current llama.cpp wants a value for it, and the default is fine.
    """
    conf = {
        "ngl": args.ngl,
        "args": ["-c", str(args.ctx_size), "-np", str(args.parallel)],
    }
    if args.llama_binary:
        conf["binary"] = args.llama_binary
    return conf


def build_participant(model: str, temperature: float,
                      max_tokens: int) -> dict:
    """The llama.cpp seat: sampling params plus the no-thinking rule.

    No `system_prompt` and no `base_url`: the prompt is per-request
    (`build_messages`) and the endpoint is per-run.
    """
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": _NO_THINKING,
    }


def run_generate(args) -> int:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = existing_indices(args.out)
    stats = SourceStats()
    jobs = [(idx, turns)
            for idx, turns in iter_dialogs(
                args.source, args.start, args.n_dialogs, stats)
            if idx not in done]
    skipped = stats.dialogs - len(jobs)
    print(f"{stats.rows} rows read, {stats.dialogs} dialogs in "
          f"[{args.start}, {args.start + args.n_dialogs}), "
          f"{skipped} already in {args.out}, {len(jobs)} to do")
    if not jobs:
        print(format_summary_block(read_log(args.out)))
        return 0

    # `--model` keeps its default only when no `--url` was given: an
    # argparse default survives the mutually-exclusive group.
    model_path = None if args.url else (args.model or _DEFAULT_MODEL)
    model = (os.path.basename(model_path) if model_path
             else args.model_name)
    participant = build_participant(
        model, args.temperature, args.max_tokens)
    timeout = (dh._CONNECT_TIMEOUT, args.request_timeout)
    session = requests.Session()
    server = None
    written = failed = 0
    # Three marks, because the number that projects to a full corpus
    # is the middle one: loading a 12B quant and killing the server
    # again are fixed costs a 20k-dialog run pays once.
    t_start = time.time()
    t_ready = t_done = None
    try:
        if model_path:
            server = ce.launch_llama_server(
                model_path, args.out, _server_conf(args))
            dh.wait_until_ready(server, session, args.startup_timeout)
            url = server.base_url
            logging.info(f"llama-server on port {server.port} is ready "
                         f"({args.parallel} slots)")
        else:
            url = args.url
        t_ready = time.time()

        def work(idx: int, turns: list[dict]) -> dict:
            # `thread_session()` must run *in* the worker: called at
            # submit time it would hand every worker the main
            # thread's session.
            return process_dialog(
                thread_session(), url, participant, idx, turns, model,
                timeout)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel_requests or args.parallel)
        try:
            futures = {
                executor.submit(work, idx, turns): idx
                for idx, turns in jobs
            }
            with open(args.out, "a", encoding="utf-8", newline="\n") as f:
                for i, future in enumerate(
                        concurrent.futures.as_completed(futures)):
                    record = future.result()
                    dh.write_record(f, record)
                    written += 1
                    failed += "error" in record
                    rate = (time.time() - t_ready) / written
                    print(f"[{i + 1}/{len(jobs)}] dialog "
                          f"{record['dialog_idx']} "
                          f"{'FAILED' if 'error' in record else 'ok'} "
                          f"{record['elapsed_s']:.1f}s "
                          f"({rate:.2f}s/dialog avg)")
            t_done = time.time()
        except KeyboardInterrupt:
            print(f"\ninterrupted; {written} record(s) in {args.out}")
        finally:
            t_done = t_done or time.time()
            executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if server is not None:
            server.stop()

    load = (t_ready or t_start) - t_start
    generating = t_done - (t_ready or t_start)
    print(f"\nwrote {written} record(s) ({failed} failed) to {args.out} "
          f"in {t_done - t_start:.1f}s "
          f"({load:.1f}s model load + {generating:.1f}s generating, "
          f"{generating / max(written, 1):.2f}s per dialog at "
          f"--parallel {args.parallel})")
    print(format_summary_block(read_log(args.out)))
    return 0


def format_summary_block(records: list[dict]) -> str:
    """The whole-log summary, headed so it stands out in the console."""
    return "\n".join(
        [f"\n--- summary over {len(records)} logged dialog(s) ---"]
        + format_summary(summarize(records)))


# --------------------------------------------- yes/no question detection


# Auxiliaries that can open an English polar question. "May", "shall"
# and "must" are deliberately absent -- rare in this corpus and almost
# always stilted requests, so they would cost precision on the cue and
# buy almost no coverage.
_AUX_VERBS = frozenset("""
do does did don't doesn't didn't
is are am was were isn't aren't wasn't weren't
have has had haven't hasn't hadn't
can could cannot can't couldn't
will would won't wouldn't
should shouldn't
""".split())

# Discourse markers that may sit in front of the auxiliary with no
# comma at all ("So did she say yes?"). One of these is skipped, and
# then the comma rule below still applies, so "Hey Marcus, are you..."
# works too.
_LEAD_IN_WORDS = frozenset(
    "so and but well oh ok okay hey hi hmm then now yeah um".split())

# At most this many words may precede the auxiliary as a lead-in
# closed by a comma ("So, do you...", "Hey Marcus, are you...").
_LEAD_IN_MAX_WORDS = 2

# Sentence end: a terminator, any closing quotes/brackets, whitespace.
_SENTENCE_END = re.compile(r"[.!?]['\"’”)\]]*\s+")
# A word, apostrophes included so "don't" survives as one token.
_WORD = re.compile(r"[A-Za-z'’]+")
# Trailing characters tolerated after the final "?".
_QUESTION_TAIL = " \t\"'’”)]}"


def final_question(text: str) -> str | None:
    """The last sentence of `text`, if the turn ends in a question.

    A simplified User turn runs to two sentences and puts the question
    last ("That is sad. Can you come later?"), so the cue has to be
    looked for there rather than at the start of the turn.
    """
    stripped = text.strip().rstrip(_QUESTION_TAIL)
    if not stripped.endswith("?"):
        return None
    matches = list(_SENTENCE_END.finditer(stripped))
    start = matches[-1].end() if matches else 0
    return stripped[start:]


def is_yes_no_question(text: str) -> bool:
    """Does `text` end in an auxiliary-initial (polar) question?

    Purely structural, and deliberately so: this is the cue a
    ~8k-weight model could plausibly learn to detect. It accepts
    requests and offers dressed as questions ("Can you help me?", "Do
    you want some?") along with true polar questions, which is on
    purpose -- a model cannot tell them apart structurally either, and
    "Yes." answers all three. What the reply *meant* is the
    classifier's business, not this function's.
    """
    question = final_question(text)
    if question is None:
        return False
    head = question
    first = _WORD.search(head)
    if first is not None and first.group(0).lower() in _LEAD_IN_WORDS:
        head = head[first.end():]
    comma = head.find(",")
    if comma != -1:
        lead_words = _WORD.findall(head[:comma])
        if 0 < len(lead_words) <= _LEAD_IN_MAX_WORDS:
            head = head[comma + 1:]
    match = _WORD.search(head)
    if match is None:
        return False
    return match.group(0).lower().replace("’", "'") in _AUX_VERBS


# ------------------------------------------------- polar classification


YES, NO, OTHER = "yes", "no", "other"
POLAR_LABELS = (YES, NO, OTHER)

_POLAR_ALIASES = {
    "yes": YES, "yeah": YES, "yep": YES, "affirmative": YES, "true": YES,
    "no": NO, "nope": NO, "negative": NO, "false": NO,
    "other": OTHER, "neither": OTHER, "unclear": OTHER, "none": OTHER,
    "null": OTHER, "na": OTHER, "unknown": OTHER,
}

_POLAR_SYSTEM = """\
You label short conversation replies. You are given the last thing a \
"User" said -- a yes/no question, a request, or an offer -- and the \
reply "Bot" gave. Decide what the reply MEANS as an answer, and \
answer with exactly one word:

yes -- affirmative in effect: it confirms, agrees, admits, accepts \
the offer, or grants the request. A reply that says yes and then adds \
detail, or that simply gets on with what was asked, is still yes.
no -- negative in effect: it denies, refuses, declines, disagrees, or \
says it cannot or will not.
other -- neither: the reply dodges, answers some other question, says \
it does not know, or has nothing to do with what the User said.

Judge the meaning, not the words: "Of course, what's up?" is yes, \
"Sure, let me take a look." is yes, "I wish I could, but I'm busy." \
is no, "Why do you ask?" is other.

Reply with one word -- yes, no, or other -- and nothing else."""

_LABEL_WORD = re.compile(r"[a-z]+")


def polar_messages(question: str, reply: str) -> list[dict]:
    """The classification request for one (question, reply) pair.

    The system message is a constant, so every request shares a
    byte-identical prefix and `cache_prompt` has something to reuse.
    """
    return [
        {"role": "system", "content": _POLAR_SYSTEM},
        {"role": "user", "content": f"User: {question}\nBot: {reply}"},
    ]


def parse_polar(text: str) -> tuple[str, bool]:
    """`(label, loose)` from a classifier reply; raise ValueError.

    `loose` marks a reply that needed more than its first word to
    read, so the tally can keep those apart from clean one-word
    answers.
    """
    body = ce.strip_fences(text).strip().lower()
    words = _LABEL_WORD.findall(body)
    if not words:
        raise ValueError(f"no word in the reply: {text!r}")
    if words[0] in _POLAR_ALIASES:
        return _POLAR_ALIASES[words[0]], False
    for word in words[1:]:
        if word in _POLAR_ALIASES:
            return _POLAR_ALIASES[word], True
    raise ValueError(f"no label in the reply: {text!r}")


def polar_jobs(records: list[dict], variant: str = "s3") -> list[tuple]:
    """Every Bot turn that answers a detected yes/no question.

    `(dialog_idx, turn_idx, question, reply)`, where the question is
    the text the corpus actually shows the model (the `variant`
    selection for the User side) and the reply is the ORIGINAL Bot
    utterance -- the only field still carrying enough meaning to be
    affirmative or negative.
    """
    jobs = []
    for record in records:
        if "error" in record or not record.get("turns"):
            continue
        turns = record["turns"]
        for i, turn in enumerate(turns):
            if turn["side"] != _BOT or i == 0:
                continue
            previous = turns[i - 1]
            if previous["side"] != _USER:
                continue
            question, _ = select_text(previous, variant)
            if not is_yes_no_question(question):
                continue
            reply = turn.get("original") or turn.get("simple") or ""
            jobs.append((record["dialog_idx"], i, question, reply))
    return jobs


def classify_polar(session, url: str, participant: dict, dialog_idx: int,
                   turn_idx: int, question: str, reply: str,
                   timeout) -> dict:
    """One (question, reply) pair to one polar-log record.

    A failure is logged as `other` rather than dropped: the renderer
    then leaves that Bot turn its trivial phrase, which is exactly the
    s3 behaviour, and a re-run does not have to redo the turn.
    """
    try:
        raw, _, _ = ce.ask(
            session, url, participant, polar_messages(question, reply),
            timeout)
    except Exception as e:  # after dialog_harness's own HTTP retries
        logging.warning(f"dialog {dialog_idx} turn {turn_idx}: {e}")
        return {"dialog_idx": dialog_idx, "turn": turn_idx,
                "label": OTHER, "error": str(e)}
    try:
        label, loose = parse_polar(raw)
    except ValueError as e:
        logging.warning(f"dialog {dialog_idx} turn {turn_idx}: {e}")
        return {"dialog_idx": dialog_idx, "turn": turn_idx,
                "label": OTHER, "error": str(e), "raw": raw[:200]}
    record = {"dialog_idx": dialog_idx, "turn": turn_idx, "label": label}
    if loose:
        record["loose"] = True
        record["raw"] = raw[:200]
    return record


def read_polar_records(path: str) -> list[dict]:
    """Every polar-log record carrying the keys the renderer needs."""
    return [r for r in read_log(path) if "dialog_idx" in r and "turn" in r]


def polar_map(records: list[dict]) -> dict:
    """`{dialog_idx: {turn_idx: label}}` from polar-log records."""
    out: dict = {}
    for record in records:
        out.setdefault(record["dialog_idx"], {})[record["turn"]] = (
            record.get("label", OTHER))
    return out


def polar_keys(path: str) -> set:
    """`(dialog_idx, turn)` pairs already classified."""
    return {(r["dialog_idx"], r["turn"]) for r in read_polar_records(path)}


def summarize_polar(records: list[dict]) -> dict:
    counts = collections.Counter(r.get("label", OTHER) for r in records)
    return {
        "classified": len(records),
        YES: counts[YES],
        NO: counts[NO],
        OTHER: counts[OTHER],
        "errors": sum(1 for r in records if r.get("error")),
        "loose": sum(1 for r in records if r.get("loose")),
    }


def format_polar_block(records: list[dict]) -> str:
    """The `classify` summary, headed so it stands out in the console."""
    summary = summarize_polar(records)
    total = max(summary["classified"], 1)
    lines = [f"\n--- polar classification over {summary['classified']} "
             f"turn(s) ---"]
    for label in POLAR_LABELS:
        lines.append(f"  {label:<6}{summary[label]:>8} "
                     f"({100 * summary[label] / total:.1f}%)")
    lines.append(f"  request/parse failures: {summary['errors']}, "
                 f"loose parses: {summary['loose']}")
    return "\n".join(lines)


def polar_participant(model: str, temperature: float,
                      max_tokens: int) -> dict:
    """The classifier seat: no thinking, and a cacheable prefix."""
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": dict(_NO_THINKING, cache_prompt=True),
    }


def run_classify(args) -> int:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    records = read_log(args.log)
    if args.n_dialogs is not None:
        records = records[:args.n_dialogs]
    jobs = polar_jobs(records, args.variant)
    done = polar_keys(args.out)
    todo = [job for job in jobs if (job[0], job[1]) not in done]
    print(f"{len(records)} log record(s): {len(jobs)} Bot turn(s) answer a "
          f"yes/no question, {len(jobs) - len(todo)} already in "
          f"{args.out}, {len(todo)} to do")
    if not todo:
        print(format_polar_block(read_polar_records(args.out)))
        return 0

    # `--model` keeps its default only when no `--url` was given: an
    # argparse default survives the mutually-exclusive group.
    model_path = None if args.url else (args.model or _DEFAULT_POLAR_MODEL)
    model = (os.path.basename(model_path) if model_path
             else args.model_name)
    participant = polar_participant(
        model, args.temperature, args.max_tokens)
    timeout = (dh._CONNECT_TIMEOUT, args.request_timeout)
    session = requests.Session()
    server = None
    written = 0
    t_start = time.time()
    t_ready = t_done = None
    try:
        if model_path:
            server = ce.launch_llama_server(
                model_path, args.out, _server_conf(args))
            dh.wait_until_ready(server, session, args.startup_timeout)
            url = server.base_url
            logging.info(f"llama-server on port {server.port} is ready "
                         f"({args.parallel} slots)")
        else:
            url = args.url
        t_ready = time.time()

        def work(job) -> dict:
            # `thread_session()` must run *in* the worker: called at
            # submit time it would hand every worker the main thread's
            # session.
            return classify_polar(
                thread_session(), url, participant, *job, timeout)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.parallel_requests or args.parallel)
        try:
            futures = [executor.submit(work, job) for job in todo]
            with open(args.out, "a", encoding="utf-8", newline="\n") as f:
                for future in concurrent.futures.as_completed(futures):
                    dh.write_record(f, future.result())
                    written += 1
                    if written % 500 == 0 or written == len(todo):
                        rate = written / max(time.time() - t_ready, 1e-6)
                        print(f"[{written}/{len(todo)}] {rate:.1f} turns/s")
            t_done = time.time()
        except KeyboardInterrupt:
            print(f"\ninterrupted; {written} record(s) in {args.out}")
        finally:
            t_done = t_done or time.time()
            executor.shutdown(wait=False, cancel_futures=True)
    finally:
        if server is not None:
            server.stop()
    print(f"\nwrote {written} record(s) to {args.out} in "
          f"{t_done - t_start:.1f}s")
    print(format_polar_block(read_polar_records(args.out)))
    return 0


# ------------------------------------------------------------ rendering


# Which field each side's utterance comes from, per corpus variant.
VARIANTS = {
    # s0: the natural text of the same dialogs -- the equal-indices
    # control for every simplified variant.
    "s0": {_USER: "original", _BOT: "original"},
    "s1": {_USER: "original", _BOT: "simple"},
    "s2": {_USER: "original", _BOT: "trivial"},
    "s3": {_USER: "simple", _BOT: "trivial"},
    # s4: s3, except that a Bot turn whose original reply *meant* yes
    # or no to an auxiliary-initial User question becomes the bare
    # word. Everything else is the s3 trivial phrase, so s4 - s3 is
    # exactly the polar answers.
    "s4": {_USER: "simple", _BOT: "trivial"},
}

# Variants that additionally consult the polar log.
POLAR_VARIANTS = frozenset({"s4"})

# The two answers s4 substitutes in. Bare and punctuated like the
# trivial phrases around them, and deliberately not varied ("Yes, I
# do."): the point is a single, maximally learnable cue -> answer map.
POLAR_TEXT = {YES: "Yes.", NO: "No."}

# What to fall back on when the chosen field is missing or empty. A
# validated log never needs this; an older or hand-edited one might,
# and a slightly-less-simplified turn beats a hole in the corpus.
_FALLBACK = {
    "original": ("original",),
    "simple": ("simple", "original"),
    "trivial": ("trivial", "simple", "original"),
}


def select_text(turn: dict, variant: str) -> tuple[str, str]:
    """`(text, field used)` for one turn under one variant."""
    wanted = VARIANTS[variant][turn["side"]]
    for field in _FALLBACK[wanted]:
        value = turn.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    raise ValueError(f"no usable text for a {turn['side']} turn: {turn!r}")


def render_dialog(turns: list[dict], variant: str, stats: dict,
                  polar: dict | None = None) -> str:
    """One dialog as `Name: utterance` lines separated by blank lines.

    `polar` is `{turn index: "yes"/"no"/"other"}` for this dialog; only
    the polar variants pass it, and only `yes`/`no` override the text.
    """
    lines = []
    for i, turn in enumerate(turns):
        label = (polar or {}).get(i)
        override = POLAR_TEXT.get(label) if turn["side"] == _BOT else None
        if override is not None:
            text, field = override, VARIANTS[variant][turn["side"]]
            stats["polar"] += 1
            stats[f"polar_{label}"] += 1
        else:
            text, field = select_text(turn, variant)
        text, collapsed = collapse(text)
        stats["collapsed"] += collapsed
        stats["fallbacks"] += field != VARIANTS[variant][turn["side"]]
        stats["turns"] += 1
        lines.append(f"{turn['side']}: {text}")
    return "\n\n".join(lines)


def render_corpus(records: list[dict], variant: str,
                  polar: dict | None = None) -> tuple[str, dict]:
    """The whole corpus text for one variant, plus its tallies.

    Failed dialogs are skipped; the rest are emitted in `dialog_idx`
    order (the log is in completion order, which the thread pool
    shuffles), two blank lines between dialogs, one trailing newline
    -- the format `make_soda_txt.py` writes.

    `polar` is the `{dialog_idx: {turn index: label}}` map of
    `polar_map`, consulted only by the variants in `POLAR_VARIANTS`.
    """
    stats = {"dialogs": 0, "turns": 0, "failed": 0, "collapsed": 0,
             "fallbacks": 0, "polar": 0, f"polar_{YES}": 0,
             f"polar_{NO}": 0}
    usable = []
    for record in records:
        if "error" in record or not record.get("turns"):
            stats["failed"] += 1
            continue
        usable.append(record)
    usable.sort(key=lambda r: r["dialog_idx"])
    chunks = []
    for record in usable:
        by_turn = (polar or {}).get(record["dialog_idx"]) \
            if variant in POLAR_VARIANTS else None
        chunks.append(
            render_dialog(record["turns"], variant, stats, by_turn))
        stats["dialogs"] += 1
    text = "\n\n\n".join(chunks)
    if text:
        text += "\n"
    return text, stats


def default_corpus_path(variant: str) -> str:
    return os.path.join(_CORPUS_DIR, f"soda_train_ub_{variant}.txt")


def run_render(args) -> int:
    records = read_log(args.log)
    if not records:
        raise ValueError(f"no records in {args.log}")
    if args.n_dialogs is not None:
        records = records[:args.n_dialogs]
    polar = None
    if args.variant in POLAR_VARIANTS:
        polar_records = read_polar_records(args.polar_log)
        if not polar_records:
            raise ValueError(
                f"variant {args.variant} needs a polar log; "
                f"{args.polar_log} has no usable records (run "
                f"`simplify_corpus.py classify` first)")
        polar = polar_map(polar_records)
    out_path = args.out or default_corpus_path(args.variant)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    text, stats = render_corpus(records, args.variant, polar)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{args.log} -> {out_path} ({args.variant}: "
          f"User={VARIANTS[args.variant][_USER]}, "
          f"Bot={VARIANTS[args.variant][_BOT]})")
    print(f"  dialogs {stats['dialogs']}, turns {stats['turns']}, "
          f"bytes {len(text.encode('utf-8'))}")
    print(f"  skipped (failed in the log): {stats['failed']}")
    print(f"  newline collapses: {stats['collapsed']}, "
          f"field fallbacks: {stats['fallbacks']}")
    if args.variant in POLAR_VARIANTS:
        turns = max(stats["turns"], 1)
        print(f"  polar answers: {stats['polar']} "
              f"({100 * stats['polar'] / turns:.1f}% of all turns) -- "
              f"{stats[f'polar_{YES}']} \"{POLAR_TEXT[YES]}\", "
              f"{stats[f'polar_{NO}']} \"{POLAR_TEXT[NO]}\"")
    return 0


# ------------------------------------------------------------------ cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM-simplified variants of the User/Bot SODA corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate", help="ask an LLM to simplify SODA dialogs")
    gen.add_argument(
        "--source", default=_DEFAULT_SOURCE,
        help=f"SODA parquet split (default: {_DEFAULT_SOURCE})")
    gen.add_argument(
        "--n-dialogs", type=int, default=1000,
        help="how many dialogs to simplify (default: 1000)")
    gen.add_argument(
        "--start", type=int, default=0,
        help="first dialog index, counting filtered dialogs -- the same "
             "index as in soda_train_ub.txt (default: 0)")
    endpoint = gen.add_mutually_exclusive_group()
    endpoint.add_argument(
        "--model", default=None,
        help=f"GGUF to serve with llama-server (default, unless --url is "
             f"given: {_DEFAULT_MODEL})")
    endpoint.add_argument(
        "--url", default=None,
        help="an already-running chat-completions endpoint instead")
    gen.add_argument(
        "--model-name", default="external",
        help="model label recorded with each dialog when --url is used")
    gen.add_argument(
        "--parallel", type=int, default=_DEFAULT_PARALLEL,
        help=f"llama-server decode slots (-np) and, unless "
             f"--parallel-requests says otherwise, in-flight requests "
             f"(default: {_DEFAULT_PARALLEL})")
    gen.add_argument(
        "--parallel-requests", type=int, default=None,
        help="in-flight requests, if different from --parallel (useful "
             "with --url, whose slot count we do not set)")
    gen.add_argument(
        "--temperature", type=float, default=_DEFAULT_TEMPERATURE,
        help=f"sampling temperature (default: {_DEFAULT_TEMPERATURE})")
    gen.add_argument(
        "--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS,
        help=f"cap on one reply (default: {_DEFAULT_MAX_TOKENS})")
    gen.add_argument(
        "--out", default=_DEFAULT_LOG,
        help=f"output JSONL (default: {_DEFAULT_LOG})")
    gen.add_argument(
        "--llama-binary", default=None,
        help="llama-server executable (default: $LLAMA_SERVER, then PATH, "
             "then the known local install)")
    gen.add_argument(
        "--ngl", type=int, default=99,
        help="GPU layers for the launched llama-server (default: 99)")
    gen.add_argument(
        "--ctx-size", type=int, default=_DEFAULT_CTX,
        help=f"llama-server context, split across the slots "
             f"(default: {_DEFAULT_CTX})")
    gen.add_argument(
        "--startup-timeout", type=float, default=_DEFAULT_STARTUP_TIMEOUT,
        help=f"seconds to wait for /health (default: "
             f"{_DEFAULT_STARTUP_TIMEOUT})")
    gen.add_argument(
        "--request-timeout", type=float, default=_DEFAULT_REQUEST_TIMEOUT,
        help=f"read timeout per request in seconds (default: "
             f"{_DEFAULT_REQUEST_TIMEOUT})")
    gen.set_defaults(func=run_generate)

    cls = subparsers.add_parser(
        "classify",
        help="label the Bot answers to yes/no questions yes/no/other")
    cls.add_argument(
        "--log", default=_DEFAULT_LOG,
        help=f"a `generate` output JSONL (default: {_DEFAULT_LOG})")
    cls.add_argument(
        "--n-dialogs", type=int, default=None,
        help="classify only the first N records of the log, so a corpus "
             "variant covers exactly the dialogs an earlier render did "
             "(default: all)")
    cls.add_argument(
        "--variant", default="s3", choices=sorted(VARIANTS),
        help="which variant's User text the question is read from "
             "(default: s3)")
    endpoint = cls.add_mutually_exclusive_group()
    endpoint.add_argument(
        "--model", default=None,
        help=f"GGUF to serve with llama-server (default, unless --url is "
             f"given: {_DEFAULT_POLAR_MODEL})")
    endpoint.add_argument(
        "--url", default=None,
        help="an already-running chat-completions endpoint instead")
    cls.add_argument(
        "--model-name", default="external",
        help="model label recorded with each turn when --url is used")
    cls.add_argument(
        "--parallel", type=int, default=_DEFAULT_POLAR_PARALLEL,
        help=f"llama-server decode slots (-np) and, unless "
             f"--parallel-requests says otherwise, in-flight requests "
             f"(default: {_DEFAULT_POLAR_PARALLEL})")
    cls.add_argument(
        "--parallel-requests", type=int, default=None,
        help="in-flight requests, if different from --parallel")
    cls.add_argument(
        "--temperature", type=float, default=0.0,
        help="sampling temperature (default: 0, near-deterministic)")
    cls.add_argument(
        "--max-tokens", type=int, default=_DEFAULT_POLAR_MAX_TOKENS,
        help=f"cap on one reply (default: {_DEFAULT_POLAR_MAX_TOKENS}; the "
             "prompt asks for one word)")
    cls.add_argument(
        "--out", default=_DEFAULT_POLAR_LOG,
        help=f"output JSONL (default: {_DEFAULT_POLAR_LOG})")
    cls.add_argument(
        "--llama-binary", default=None,
        help="llama-server executable (default: $LLAMA_SERVER, then PATH, "
             "then the known local install)")
    cls.add_argument(
        "--ngl", type=int, default=99,
        help="GPU layers for the launched llama-server (default: 99)")
    cls.add_argument(
        "--ctx-size", type=int, default=_DEFAULT_POLAR_CTX,
        help=f"llama-server context, split across the slots "
             f"(default: {_DEFAULT_POLAR_CTX})")
    cls.add_argument(
        "--startup-timeout", type=float, default=_DEFAULT_STARTUP_TIMEOUT,
        help=f"seconds to wait for /health (default: "
             f"{_DEFAULT_STARTUP_TIMEOUT})")
    cls.add_argument(
        "--request-timeout", type=float, default=_DEFAULT_REQUEST_TIMEOUT,
        help=f"read timeout per request in seconds (default: "
             f"{_DEFAULT_REQUEST_TIMEOUT})")
    cls.set_defaults(func=run_classify)

    ren = subparsers.add_parser(
        "render", help="write one corpus variant out of a generate log")
    ren.add_argument(
        "--log", default=_DEFAULT_LOG,
        help=f"a `generate` output JSONL (default: {_DEFAULT_LOG})")
    ren.add_argument(
        "--variant", required=True, choices=sorted(VARIANTS),
        help="s1: User original + Bot simple; s2: User original + Bot "
             "trivial; s3: User simple + Bot trivial; s4: s3 with polar "
             "answers")
    ren.add_argument(
        "--n-dialogs", type=int, default=None,
        help="render only the first N records of the log (default: all); "
             "how a later variant covers exactly the dialogs an earlier "
             "one did, the log having grown in between")
    ren.add_argument(
        "--polar-log", default=_DEFAULT_POLAR_LOG,
        help=f"a `classify` output JSONL, required by the polar variants "
             f"({', '.join(sorted(POLAR_VARIANTS))}); default: "
             f"{_DEFAULT_POLAR_LOG}")
    ren.add_argument(
        "--out", default=None,
        help="output text file (default: "
             "data/soda/soda_train_ub_<variant>.txt)")
    ren.set_defaults(func=run_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
