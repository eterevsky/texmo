"""Scripted-examiner eval: generate student dialogs, then grade them.

Milestone 2 of the tiny chatbot program (`docs/roadmap.md`). The eval
measures *each student answer*, not whole dialogs:

    generate  an examiner model (3-9B, local) plays the "User" side of
              a seed dialog from `data/eval/seeds.jsonl` while the
              student (a texmo model, seated through
              `texmo.py chat-server`) plays "Bot"; 10 turns, 5 each.
    judge     a DIFFERENT model grades every student answer on three
              criteria -- (a) internally consistent and grammatically
              correct, (b) consistent with the preceding turn, (c) also
              substantive, not a generic deflection -- and writes a
              report against the working targets a>=90%, b>=90%,
              c>=50%.
    phrases   distil a dialogs file into the student's N most common
              answers with their counts, which is the input for the
              null model below.

    uv run python scripts/chat_eval.py generate --n-seeds 100 \\
        --student models/hb32-8k-1ub.json \\
        --student-temperature 0.3,0.5,0.7 \\
        --examiner C:/Users/oleg/models/gemma-4-12b-it-Q6_K.gguf \\
        --anchor-good --anchor-garbage \\
        --garbage-student scratch/untrained.json

    uv run python scripts/chat_eval.py judge \\
        --dialogs data/eval/dialogs-hb32-8k-1ub-20260826-120000.jsonl \\
        --judge C:/Users/oleg/models/Qwen3-8B-Q6_K.gguf

    uv run python scripts/chat_eval.py phrases \\
        --dialogs data/eval/eval100-hb32-8k-s3.jsonl \\
        --top 20 --out data/eval/phrases-s3-top20.json
    uv run python scripts/chat_eval.py generate --n-seeds 100 \\
        --student-phrases data/eval/phrases-s3-top20.json \\
        --examiner C:/Users/oleg/models/gemma-4-12b-it-Q6_K.gguf

## Temperature sweep

Which sampling temperature reads best is not known a priori, so
`--student-temperature` takes a comma-separated *list* and `generate`
runs seeds x temperatures with the servers kept up for the whole
sweep -- loading the examiner is the expensive part, so it is paid
once. Every dialog record carries its `student_temperature`, one
dialogs file holds the whole sweep, and the report gives each
temperature its own row so "which T is best" is read straight off
report.md. The targets apply per temperature.

The anchors calibrate the *judge*, not the student, so they run once
at a single representative temperature (`--anchor-temperature`,
default: the first of the list).

Both subcommands are resumable: records are appended one line at a
time and flushed, and a rerun skips work already present in the output
file (`(anchor, seed_idx, student_temperature)` for dialogs, plus the
answer position for grades). Ctrl-C therefore costs at most the
in-flight item.

## Generation

The examiner's system prompt carries the seed's *whole* script with
speaker labels -- deliberately, per the design note: without it the
seeds contribute only openers and the examiner's own favorite topics
take over. It is told to follow the topic while adapting to what the
student actually says, and to keep every utterance SODA-length (1-2
short sentences); verbosity would push the student's conditioning out
of its training distribution and measure OOD robustness instead of
coherence.

Turn 1 is the seed's opener **verbatim** -- not generated -- so every
dialog starts from the seed distribution exactly.

Two message-assembly details that are not obvious:

- The examiner speaks first, so from its own seat the dialog reads
  assistant, user, assistant, ... A leading assistant message makes
  Gemma-family chat templates raise ("roles must alternate"), so one
  short kickoff `user` message (`_KICKOFF`) is prepended. It is the
  only text in the examiner's context that is not system prompt or
  dialog.
- A student may return an empty utterance (the model emits a turn
  boundary immediately). The empty string is recorded verbatim -- an
  empty answer is a *failing* answer, not a crash -- but message
  assembly substitutes `_EMPTY_PLACEHOLDER` so endpoints that reject
  empty content still run.

## Anchors

`--anchor-good` / `--anchor-garbage` add calibration dialogs on the
first three seeds (`_ANCHOR_SEEDS`), recorded with `anchor` set:

    good     the examiner model itself seated as the student, through
             the same llama-server, with a "be an ordinary chat
             partner" system prompt. Expected ~100/100/high-c.
    garbage  an untrained texmo model (`--garbage-student <manifest>`,
             e.g. `texmo.py train --steps 2 -o scratch/untrained.json`)
             through its own chat-server. Expected ~0/0/0.

The report scores them separately. A judge that does not put good far
above garbage on all three criteria is broken, and the run's numbers
mean nothing.

## The null model (phrase bot)

The anchors bracket the judge; the phrase bot brackets the *student*
from below. A model with a handful of stock answers scores on (b) and
(c) by luck alone -- sprayed at random, "I am good." lands on a fair
share of turns -- so a raw b/c says nothing until the luck floor is
known.

`--student-phrases <json>` seats exactly that floor:

    {"phrases": [{"text": ..., "weight": ...}, ...]}

One independent weighted draw per turn (`random.choices`), the dialog
so far never read. Build the file from a real run with the `phrases`
subcommand and the bot emits that model's own phrases at that model's
own relative frequencies, differing from it in one respect only: it
does not condition on the question. Whatever b/c the model has *above*
the bot is the part conditioning bought; (a) should come out ~100% by
construction, since every phrase is a sentence the judge already
passed.

The draws come from one `random.Random(--phrase-seed)` stream for the
whole run, so the same seed reproduces the dialogs. No server is
launched and no sampling temperature applies: the sweep collapses to a
single pass and `student_temperature` is recorded as null rather than
as a number nothing used.

## The other reference student: ELIZA

`--student-eliza` seats Weizenbaum's DOCTOR (1966) -- the vendored
pattern matcher in `scripts/eliza/`, zero weights, no training data.
It brackets the eval from a third side. The phrase bot ignores the
question entirely and is the *luck* floor on (b); ELIZA reads the
question, reflects the user's own words back with the pronouns
flipped, and asks about them, which is the maximum (b) a system with
no world model can buy. It cannot say anything of its own, so it is
also a floor on (c). "Responsive but empty" is a shape the eval should
be able to name, and a student that beats the phrase bot on (b)
without beating ELIZA has learned reflection rather than content.

Like the phrase bot it takes no server and no sampling temperature
(`student_temperature` is null). One session per dialog: the matcher
is stateful -- each decomposition rule cycles through its reassembly
rules in order, and unmatched turns push onto a memory stack that a
later turn pops -- so `run_dialog` resets the student per dialog and
`--eliza-seed` plus the seed index reproduce any single transcript.

## Judging: three context-scoped passes

Each criterion is asked in its own request, seeing exactly the context
that criterion is defined on and nothing more:

    A  the student utterance ALONE, framed as a short reply in a
       casual chat          -> {"comment", "a"}
    B  the preceding User turn and the reply, that pair alone
                            -> {"comment", "b"}
    C  the whole dialog through the reply
                            -> {"comment", "c", "user_problem"}

The split is a fix, not a refactor. With one prompt carrying the whole
dialog, context leaked into (a), which is defined on the reply alone:
the null phrase bot -- whose every utterance is a well-formed sentence
some real model produced -- scored a = 91.2%, failing 44 of 500 with
comments like "inconsistent with the conversation". A judge that
cannot see the conversation cannot make that mistake, so the phrase
bot's pass-A rate becomes a *calibration* number: it should read
~100%, and anything below is judge noise rather than student
behaviour.

Output must be strict JSON; a parse failure gets one retry with an
error-correcting suffix and is then recorded as `judge_error`.
Markdown fences and `<think>` blocks are stripped first (Qwen3 habits).

Load-bearing details in the prompts, each of them a fix for a failure
the anchors caught with Qwen3-8B (2026-08-26) -- do not tidy them away
without re-running the anchors:

- **`comment` comes first in every requested JSON.** Keys are
  generated in order, so describing the reply before voting makes the
  comment a one-line chain of thought. Without it the judge went
  straight to `"a": true`.
- **Pass A names word salad explicitly.** With only "is it
  grammatically correct?", the garbage anchor -- literal random bytes
  -- passed (a) on 14 of 15 answers. Spelling out that unreadable
  text, invented words and fluent-looking word salad are all `false`
  took it to 0 of 15, while the good anchor stayed at 100%.
- **Pass A also says a dull-but-clean sentence passes**, and that a
  bare fragment ("Why?", "Me too.") is normal in chat. The clause
  above overshot once and the counter-example restored it. The old
  "do not let a bad conversation drag a down" plea is gone: pass A
  cannot see the conversation at all.
- **The defect key is called `user_problem` in the prompt** (the
  record still calls it `examiner_defect`, and the parser takes either
  name). Named after the model under test, the field collected remarks
  about the *student* on 73% of answers; named after the side it is
  about, and told that an incoherent Bot is never a User problem, it
  goes quiet on clean dialogs and still fires on a User that repeats
  itself.

The record keeps `c_raw` -- what pass C voted -- and reports
`c = c_raw and b`: an answer cannot be substantive if it does not fit
the turn, and pass C is no longer asked about fit. `c_raw` without `b`
is now a disagreement between two independent passes rather than a
self-contradiction inside one answer, so it is counted as
`c_without_b`: a judge-noise diagnostic in the report, not a repaired
grade.

## Judging: caching and parallelism

Every request carries `"cache_prompt": true`, and each pass's system
prompt is byte-identical across requests -- the utterance, the pair or
the dialog goes in the *user* message -- so llama-server reuses the
shared prefix KV per slot. Measured (b10472, Qwen3-8B Q6_K): prompt
eval ~15 ms for the ~30 new tokens of a ~700-token prompt, i.e. the
prefix is not recomputed.

The passes run one after another over a chunk of answers -- all A,
then all B, then all C -- so a slot's cached prefix stays put for a
whole pass, and the server can be *sized per pass*. Two shapes, since
the prompt sizes are an order of magnitude apart:

    A, B  one server, `--parallel-ab` slots of `--ctx-ab` each
          (default 16 x 1024): a ~250-token cached prefix plus a line
          or two, and a reply capped at 192 tokens
    C     `--parallel-c` slots of `--ctx-size` each (default 4 x 4096):
          the whole dialog, `--judge-max-tokens` for the reply

Both budgets are 16k tokens of KV. Note that an explicit `-np N`
*divides* `-c` among the slots on this build, so the script multiplies
it back up: `--ctx-ab` and `--ctx-size` are per request. The judge
reloads in seconds, so relaunching between the two phases is cheaper
than running the short passes at pass C's context.

Measured on this box (RTX 5090 shared with the search server, Qwen3-8B
Q6_K, 500 phrase-bot answers), pass A: 17.5 answers/s at 4 slots,
20.6 at 8, 24.2 at 16 -- about +18% per doubling, no knee yet at 16.
Decode is the bottleneck, not the queue, so more slots keep paying a
little; 16 is where the KV budget stops being free.

The chunk (`--chunk`, default 2000 answers) bounds the loss from a
Ctrl-C: only a complete three-pass grade is ever written, so an
interrupt costs the chunk in flight and nothing else. Each chunk pays
the two server starts.

`--n-seeds` grades only the first N distinct seeds of the dialogs file
(the anchors are always kept), which is how a 15k-answer file is
sampled without regenerating anything.

An n-gram repetition metric rides along without an LLM: `rep3` is the
fraction of repeated word 3-grams (`1 - unique/total`), and
`lrs_ratio` the length of the longest substring occurring twice
divided by the answer length. Both are 0 for a clean short answer and
approach 1 for a stuck one.

## Server lifecycle

Everything reuses `scripts/dialog_harness.py`: free-port assignment,
`llama-server` binary resolution (`--llama-binary`, `$LLAMA_SERVER`,
PATH, the known local install), /health polling, per-server logs next
to the output file, and `kill_tree` teardown (a wrapper script's
grandchild would otherwise keep the GPU). texmo students are launched
the same way, as `uv run python texmo.py chat-server` subprocesses.
Every launched server is stopped on the way out -- normal finish,
exception, Ctrl-C alike. `--student-url` / `--examiner-url` /
`--judge-url` point at servers somebody else runs, which are never
touched.

Every llama.cpp request carries
`{"chat_template_kwargs": {"enable_thinking": false}}`: 2026 instruct
models reason by default and return empty `content` otherwise.
"""
import argparse
import collections
import concurrent.futures
import datetime
import json
import logging
import os
import random
import re
import sys
import threading
import time

import dialog_harness as dh
import eliza
import requests

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from texmo.pjson import save_json

_DEFAULT_SEEDS = os.path.join("data", "eval", "seeds.jsonl")
_DEFAULT_OUT_DIR = os.path.join("data", "eval")

# 10 utterances, 5 per side, examiner first (the forced opener).
_N_TURNS = 10
# Calibration anchors run on the first N seeds.
_ANCHOR_SEEDS = 3

_EXAMINER = "examiner"
_STUDENT = "student"

# House rule for llama.cpp seats: without it 2026 instruct models burn
# the budget in <think> and return empty content.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
# The judge seat adds prompt caching: every request of a pass shares a
# byte-identical system prefix, so the slot keeps its KV.
_JUDGE_EXTRA = dict(_NO_THINKING, cache_prompt=True)

# Prepended to the examiner's messages so the roles alternate from a
# user turn -- see the module docstring.
_KICKOFF = "Start the conversation."
# Stands in for an empty utterance in message assembly only.
_EMPTY_PLACEHOLDER = "..."

# Working targets (docs/roadmap.md, 2026-08-26), in percent.
_TARGETS = {"a": 90.0, "b": 90.0, "c": 50.0}
_CRITERIA = ("a", "b", "c")
# One judge request per criterion, in the order they are asked.
_PASSES = ("a", "b", "c")

_USER_LABEL = "User"
_BOT_LABEL = "Bot"

_DEFAULT_CTX = 4096
_DEFAULT_STARTUP_TIMEOUT = 600
_DEFAULT_EXAMINER_MAX_TOKENS = 96
_DEFAULT_JUDGE_MAX_TOKENS = 400
# Generously above a SODA utterance (~1-3 bytes per hexbpe token):
# the student normally stops at a turn boundary, and a reply that
# reaches the cap is a run-on the model never closed -- a real defect,
# which the judge should see as a run-on rather than as a cut-off word.
_DEFAULT_STUDENT_MAX_TOKENS = 256

# Null model: how many of the student's own answers the `phrases`
# subcommand keeps, and the RNG seed the bot draws with.
_DEFAULT_TOP_N = 20
_DEFAULT_PHRASE_SEED = 0
# ELIZA: the base seed a dialog's session stream is derived from.
_DEFAULT_ELIZA_SEED = 0

# Judge concurrency, per phase: server slots == in-flight requests.
# Passes A and B are short prompts and share a server with many small
# slots; pass C carries a whole dialog and gets few big ones. Both
# budgets are slots x per-slot context = 16k tokens of KV.
_DEFAULT_PARALLEL_AB = 16
_DEFAULT_CTX_AB = 1024
_DEFAULT_PARALLEL_C = 4
# A pass-A/B reply is one short sentence plus a one-key JSON object.
# The cap is not just politeness: llama-server checks prompt +
# n_predict against the slot's context, so a fat cap shrinks the
# prompt that fits in a small slot.
_JUDGE_SHORT_MAX_TOKENS = 192
# How many answers go through all three passes before anything is
# written. A Ctrl-C costs the chunk in flight.
_DEFAULT_CHUNK = 2000

_EXAMINER_PROMPT = """\
You are playing the "{user}" side of a short, casual conversation with \
a much weaker chatbot called "{bot}".

Here is a reference dialog. Follow its topic, but adapt naturally to \
what {bot} actually says -- {bot} is a very small model and will often \
be confused, repetitive or off-topic, and you should react like an \
ordinary person would.

{script}

Rules:
- Write only your next utterance as {user}. No speaker label, no \
quotation marks, no stage directions, no commentary.
- Keep it short: 1-2 short sentences, like the reference dialog.
- Stay in character as an ordinary person. Never mention the reference \
dialog, this conversation being a test, or that you are a model."""

_GOOD_ANCHOR_PROMPT = """\
You are "{bot}", an ordinary, friendly person having a short casual \
conversation.

Rules:
- Write only your next utterance. No speaker label, no quotation \
marks, no stage directions, no commentary.
- Keep it short: 1-2 short sentences.
- Never mention that you are a model or that this is a test."""

_PASS_A_PROMPT = """\
You are grading ONE short reply written by a very small, weak chatbot \
in a casual chat. You will not see the conversation it came from, and \
you do not need it: this question is about the reply itself.

a) Is the reply internally consistent and grammatically correct? Is \
it readable English, and does it avoid contradicting itself? Answer \
false whenever it is not readable English at all -- random \
characters, word fragments, invented words, a garbled loop -- or when \
it breaks off mid-thought, repeats itself, or states two incompatible \
things. A weak model often produces fluent-looking word salad: that \
is false, not true.

This is casual chat, so replies are short. A bare fragment such as \
"Why?", "Me too." or "At the park." is normal speech and is TRUE. A \
short, plain, dull sentence such as "I really like it." is readable \
and self-consistent, so it is TRUE as well. You cannot see what was \
said before, so never answer false because the reply is vague, dull, \
generic or an odd thing to say -- other questions cover that.

Reply with STRICT JSON and nothing else -- no markdown fences, no \
explanation before or after. Write `comment` FIRST and let it decide \
the verdict, not the other way round:

{{"comment": <one short sentence describing the reply>, \
"a": <true or false>}}"""

_PASS_B_PROMPT = """\
You are grading one reply from a very small, weak chatbot called \
"{bot}" in a casual conversation with a person called "{user}". You \
will see exactly two lines: what {user} said, and the {bot} reply to \
it. The rest of the conversation is deliberately withheld and is not \
needed.

b) Is the {bot} reply consistent with the preceding {user} turn? Does \
it make sense as a response to what {user} just said? Answer true \
when an ordinary person could plausibly have said it at that point in \
a chat -- a short, vague or dull response can still fit the turn. \
Answer false when it answers some other question, ignores what {user} \
said, or does not fit the turn at all.

Grammar is not your question here: a clumsy reply that fits the turn \
is still true.

Reply with STRICT JSON and nothing else -- no markdown fences, no \
explanation before or after. Write `comment` FIRST and let it decide \
the verdict, not the other way round:

{{"comment": <one short sentence on how the reply relates to what \
{user} said>, "b": <true or false>}}"""

_PASS_C_PROMPT = """\
You are grading one reply from a very small, weak chatbot called \
"{bot}" that is having a casual conversation with a person called \
"{user}". You will see the conversation so far; the LAST line is the \
{bot} reply you must grade. Grade only that line.

c) Is the reply substantive -- a contentful, on-topic answer rather \
than a generic deflection? For example, to "What do you like to do on \
weekends?": "I don't know." is NOT substantive, while "I usually go \
hiking with my brother." IS substantive. Grammar and fit with the \
preceding turn are graded elsewhere and are not your question here.

Separately, `user_problem` is quality control on the {user} side and \
has nothing to do with grading {bot}. {user} is played by another \
model: does {user} itself misbehave? Say so in one short phrase when \
{user} repeats one of its own earlier turns nearly word for word, \
ignores what {bot} just said, contradicts its own earlier turn, \
breaks character, or writes a speech instead of a line of chat -- for \
example "User asked the same question three times in a row". \
Otherwise use JSON null (not the string "null"), which is the usual \
answer. {bot} being incoherent or off-topic is never a {user} \
problem, and neither is {user} simply being clear and on-topic.

Reply with STRICT JSON and nothing else -- no markdown fences, no \
explanation before or after. Write `comment` FIRST and let it decide \
the verdict, not the other way round:

{{"comment": <one short sentence describing the {bot} reply>, \
"c": <true or false>, "user_problem": <a short phrase, or null>}}"""

_RETRY_SUFFIX = """\
Your previous answer could not be parsed as JSON ({error}). Reply \
again with the JSON object only: {keys}. No fences, no prose."""


# ---------------------------------------------------------------- seeds


def load_seeds(path: str, n_seeds: int) -> list[dict]:
    """The first `n_seeds` seed dialogs, in file order.

    The 10/100 subsets of the design are prefixes of this file, so
    there is nothing else to slice.
    """
    seeds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))
            if len(seeds) >= n_seeds:
                break
    if not seeds:
        raise ValueError(f"no seeds read from {path}")
    return seeds


def render_script(seed: dict) -> str:
    """The seed dialog with speaker labels, one utterance per line.

    Labels come from `speakers`, never from turn parity: 2.2% of SODA
    two-speaker dialogs have a side taking two turns in a row.
    """
    speakers = seed["speakers"]
    return "\n".join(
        f"{speakers[i]}: {text}" for i, text in enumerate(seed["script"]))


def examiner_system_prompt(seed: dict) -> str:
    """The examiner's seat prompt for one seed."""
    return _EXAMINER_PROMPT.format(
        user=_USER_LABEL, bot=_BOT_LABEL, script=render_script(seed))


# ------------------------------------------------------ message assembly


def _content(text: str) -> str:
    return text if text.strip() else _EMPTY_PLACEHOLDER


def examiner_messages(system_prompt: str, turns: list[dict]) -> list[dict]:
    """The dialog as the examiner sees it, in chat-completions form.

    Its own utterances are `assistant`, the student's are `user`. The
    `_KICKOFF` user message keeps the roles alternating from a user
    turn, which Gemma-family templates require.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _KICKOFF},
    ]
    for turn in turns:
        role = "assistant" if turn["side"] == _EXAMINER else "user"
        messages.append({"role": role, "content": _content(turn["text"])})
    return messages


def student_messages(turns: list[dict],
                     system_prompt: str | None = None) -> list[dict]:
    """The dialog as the student sees it, in chat-completions form.

    Examiner utterances are `user`, the student's own are `assistant`,
    so the list always ends with the user turn to answer -- which is
    what `texmo.chat_server` requires. `system_prompt` is for the
    good anchor (a real instruct model in the student seat); texmo
    students get none and would drop it anyway.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    for turn in turns:
        role = "user" if turn["side"] == _EXAMINER else "assistant"
        messages.append({"role": role, "content": _content(turn["text"])})
    return messages


def render_dialog(turns: list[dict], upto: int) -> str:
    """Turns `0..upto` inclusive as labeled lines, for the judge."""
    lines = []
    for turn in turns[:upto + 1]:
        label = _USER_LABEL if turn["side"] == _EXAMINER else _BOT_LABEL
        lines.append(f"{label}: {turn['text']}")
    return "\n".join(lines)


def preceding_user_text(turns: list[dict], upto: int) -> str:
    """What the examiner said last before the answer at `upto`.

    Searched backwards rather than taken as `turns[upto - 1]`: turns
    alternate today, but a student that answers twice in a row must
    still be graded against the question it was actually answering.
    """
    for turn in reversed(turns[:upto]):
        if turn["side"] == _EXAMINER:
            return turn["text"]
    return ""


def pass_a_user(turns: list[dict], upto: int) -> str:
    """Pass A sees the utterance and nothing else."""
    return f"Reply to grade:\n\n{turns[upto]['text']}"


def pass_b_user(turns: list[dict], upto: int) -> str:
    """Pass B sees the preceding turn and the reply, that pair alone."""
    return (f"The two lines (grade the {_BOT_LABEL} line):\n\n"
            f"{_USER_LABEL}: {preceding_user_text(turns, upto)}\n"
            f"{_BOT_LABEL}: {turns[upto]['text']}")


def pass_c_user(turns: list[dict], upto: int) -> str:
    """Pass C sees the whole dialog through the reply."""
    return (f"Conversation so far (grade the last {_BOT_LABEL} line):\n\n"
            f"{render_dialog(turns, upto)}")


# The three context-scoped judge requests. `system` is formatted once,
# at import: every request of a pass must send the *same bytes* or the
# server's prompt cache has nothing to reuse.
_PASS_SPECS = {
    "a": {
        "system": _PASS_A_PROMPT.format(user=_USER_LABEL, bot=_BOT_LABEL),
        "user": pass_a_user,
        "keys": ("a",),
        "key_spec": '"comment" (string), "a" (true/false)',
        "max_tokens": _JUDGE_SHORT_MAX_TOKENS,
    },
    "b": {
        "system": _PASS_B_PROMPT.format(user=_USER_LABEL, bot=_BOT_LABEL),
        "user": pass_b_user,
        "keys": ("b",),
        "key_spec": '"comment" (string), "b" (true/false)',
        "max_tokens": _JUDGE_SHORT_MAX_TOKENS,
    },
    "c": {
        "system": _PASS_C_PROMPT.format(user=_USER_LABEL, bot=_BOT_LABEL),
        "user": pass_c_user,
        "keys": ("c",),
        "key_spec": '"comment" (string), "c" (true/false), "user_problem" '
                    "(string or null)",
        # None: pass C keeps whatever --judge-max-tokens says.
        "max_tokens": None,
    },
}

# The two server shapes a judging run needs. Passes A and B are both
# "system prefix + a couple of lines", so they share one server; only
# pass C needs room for a whole dialog.
_PHASES = (("ab", ("a", "b")), ("c", ("c",)))


def judge_messages(pass_name: str, turns: list[dict], upto: int,
                   retry_error: str | None = None) -> list[dict]:
    """One grading request for one pass: its prompt, then its context.

    `retry_error` appends the error-correcting suffix used for the
    single retry after a parse failure. It goes in the *user* message,
    like the context: the system prompt stays byte-identical across
    every request of the pass, which is what the prompt cache keys on.
    """
    spec = _PASS_SPECS[pass_name]
    user = spec["user"](turns, upto)
    if retry_error is not None:
        user += "\n\n" + _RETRY_SUFFIX.format(
            error=retry_error, keys=spec["key_spec"])
    return [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": user},
    ]


# -------------------------------------------------- repetition metrics


_WORD_RE = re.compile(r"[a-z0-9']+")


def repeated_ngram_fraction(text: str, n: int = 3) -> float:
    """Fraction of word n-grams that are not the first of their kind.

    `1 - unique/total`: 0.0 for text with no repeated n-gram, ->1.0
    for a model stuck in a loop. Texts shorter than `n` words score
    0.0 -- too short to repeat.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def longest_repeated_substring(text: str) -> int:
    """Length of the longest substring occurring at least twice.

    Sorted suffixes, longest common prefix of adjacent pairs. O(n^2)
    in the worst case, which is nothing at answer lengths.
    """
    if len(text) < 2:
        return 0
    suffixes = sorted(text[i:] for i in range(len(text)))
    best = 0
    for a, b in zip(suffixes, suffixes[1:]):
        limit = min(len(a), len(b))
        k = 0
        while k < limit and a[k] == b[k]:
            k += 1
        best = max(best, k)
    return best


def repetition_stats(text: str) -> dict:
    """The LLM-free fourth column for one answer."""
    chars = len(text)
    lrs = longest_repeated_substring(text)
    return {
        "chars": chars,
        "rep3": round(repeated_ngram_fraction(text, 3), 4),
        "lrs": lrs,
        "lrs_ratio": round(lrs / chars, 4) if chars else 0.0,
    }


# ------------------------------------------------- judge output parsing


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def strip_fences(text: str) -> str:
    """Unwrap ```json ... ``` and drop <think> blocks.

    Both are habits of the 2026 instruct models used as judges; the
    thinking block should not appear at all with
    `enable_thinking: false`, but a leaked one must not eat the JSON.
    """
    text = _THINK_RE.sub("", text).strip()
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "y"):
            return True
        if lowered in ("false", "no", "n"):
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"not a boolean: {value!r}")


_NO_DEFECT = ("null", "none", "n/a", "na", "no defect", "no defects",
              "nothing", "-")


def _defect(value) -> str | None:
    """Normalize `examiner_defect` to a real note or None.

    Judges write the *string* "null" about as often as JSON `null`,
    and an untouched one would show up in the report as a defect on
    every answer.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text or text.rstrip(".").lower() in _NO_DEFECT:
        return None
    return text


def parse_judge_output(text: str, keys: tuple = _CRITERIA) -> dict:
    """Strict-JSON grade out of one pass's raw reply.

    `keys` are the boolean verdicts that pass must return -- ("a",)
    for pass A and so on. Raises ValueError on anything that is not an
    object carrying them. Tolerates fences, a leaked thinking block,
    and prose around the object -- all failure modes seen from local
    judges, none of them a reason to throw away a good grade.
    """
    body = strip_fences(text)
    obj = None
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        match = _OBJECT_RE.search(body)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"not JSON: {e}") from e
    if obj is None:
        raise ValueError("no JSON object in the reply")
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")

    grade = {}
    for key in keys:
        if key not in obj:
            raise ValueError(f"missing key {key!r}")
        grade[key] = _as_bool(obj[key])
    # `user_problem` is what the prompt asks for -- naming the field
    # after the side it is about measurably stops the judge from
    # filling it with remarks about the student. `examiner_defect` is
    # what the record calls it, and is still accepted as an alias.
    defect = obj.get("user_problem")
    if defect is None:
        defect = obj.get("examiner_defect")
    grade["examiner_defect"] = _defect(defect)
    comment = obj.get("comment")
    grade["comment"] = "" if comment is None else str(comment)
    return grade


def is_c_without_b(grade: dict) -> bool:
    """Pass C said substantive where pass B said the reply does not fit.

    The headline `c` is `c_raw and b`, so this combination never
    survives into a grade; counting it is how the two passes'
    disagreement rate stays visible. `b` unknown (a judge error) is
    not a disagreement.
    """
    return bool(grade.get("c_raw")) and grade.get("b") is False


# --------------------------------------------------------- resumability


def record_key(record: dict) -> tuple:
    """Identity of a dialog record: anchor, seed, student temperature.

    All three are needed: the anchors reuse the first seed indexes, and
    the temperature sweep reuses every seed once per temperature.
    """
    return (record.get("anchor"), record["seed_idx"],
            record.get("student_temperature"))


def grade_key(record: dict) -> tuple:
    """Identity of a grade record: its dialog plus the answer index."""
    return record_key(record) + (record["position"],)


def existing_keys(path: str, key_fn) -> set:
    """Keys already in an output file; missing file -> empty set.

    A truncated last line (killed mid-write, which the flush-per-record
    discipline makes unlikely but not impossible) is skipped.
    """
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(key_fn(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                logging.warning(f"skipping unreadable record in {path}")
    return keys


def read_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# ------------------------------------------------------------- servers


def _server_conf(args, parallel: int | None = None,
                 ctx: int | None = None) -> dict:
    """The `server` block `dialog_harness` expects, from our flags.

    `parallel` adds `-np N`: without it llama-server queues concurrent
    requests instead of running them. An explicit `-np N` also *splits*
    `-c` across the slots (measured on b10472: `-c 4096 -np 4` gives
    1024 tokens per slot and rejects a 1119-token judge request), so
    `-c` is scaled by the slot count and `ctx` keeps meaning what it
    says: the context one request may use, prompt plus reply.
    """
    per_slot = args.ctx_size if ctx is None else ctx
    total = per_slot * parallel if parallel else per_slot
    conf = {"ngl": args.ngl, "args": ["-c", str(total)]}
    if parallel:
        conf["args"] += ["-np", str(parallel)]
    if args.llama_binary:
        conf["binary"] = args.llama_binary
    return conf


def phase_server_conf(args, phase: str) -> tuple[dict, int]:
    """The server shape and slot count for one judging phase.

    Two shapes, because the two prompt sizes are an order of magnitude
    apart: A and B are a cached system prefix plus a line or two, so
    they want many small slots, while C carries the whole dialog and
    wants a few big ones. The 8B judge reloads in seconds, so a second
    server start is cheaper than running the short passes at the wide
    context they do not need.
    """
    if phase == "ab":
        parallel, ctx = args.parallel_ab, args.ctx_ab
    else:
        parallel, ctx = args.parallel_c, args.ctx_size
    return _server_conf(args, parallel, ctx), parallel


def launch_llama_server(model_path: str, out_path: str,
                        server_conf: dict) -> dh.Server:
    """One `llama-server` on a free port, logging next to `out_path`."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    port = dh.free_port()
    log_path = dh.server_log_path(out_path, model_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    cmd = dh.server_command(
        dh.resolve_binary(server_conf), model_path, port, server_conf)
    log_file.write(f"\n=== {_now()} {' '.join(cmd)} ===\n")
    log_file.flush()
    logging.info(f"llama-server for {os.path.basename(model_path)} on port "
                 f"{port} (log {log_path})")
    return dh.Server(model_path, port, dh.spawn_server(cmd, log_file),
                     log_path, log_file)


def chat_server_command(manifest: str, port: int, temperature: float,
                        max_reply_tokens: int) -> list[str]:
    """The `texmo.py chat-server` command line for a texmo student."""
    return [
        sys.executable, "texmo.py", "chat-server",
        "-m", manifest,
        "--port", str(port),
        "-T", str(temperature),
        "--max-reply-tokens", str(max_reply_tokens),
    ]


def launch_chat_server(manifest: str, out_path: str, temperature: float,
                       max_reply_tokens: int) -> dh.Server:
    """A texmo model seated at /v1/chat/completions on a free port.

    Same shape as a llama-server seat (`dh.Server`), so readiness
    polling and `kill_tree` teardown are shared.
    """
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"model manifest not found: {manifest}")
    port = dh.free_port()
    log_path = dh.server_log_path(out_path, manifest)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    cmd = chat_server_command(manifest, port, temperature, max_reply_tokens)
    log_file.write(f"\n=== {_now()} {' '.join(cmd)} ===\n")
    log_file.flush()
    logging.info(f"chat-server for {manifest} on port {port} "
                 f"(log {log_path})")
    return dh.Server(manifest, port, dh.spawn_server(cmd, log_file),
                     log_path, log_file)


def _stop_all(servers: list):
    for server in servers:
        try:
            server.stop()
        except Exception as e:
            logging.warning(f"failed to stop server for "
                            f"{server.model_path}: {e}")


# ------------------------------------------------------------ requests


def _usage_tokens(response: dict) -> int:
    return int((response.get("usage") or {}).get("completion_tokens") or 0)


def ask(session, base_url: str, participant: dict, messages: list[dict],
        timeout) -> tuple[str, int, float]:
    """One utterance: (text, completion tokens, seconds)."""
    body = dh.build_request(participant, messages)
    t0 = time.perf_counter()
    response = dh.post_completion(
        session, dh.completions_url(base_url), body, timeout)
    elapsed = time.perf_counter() - t0
    return dh.extract_text(response), _usage_tokens(response), elapsed


# ------------------------------------------------------------- students


# A student seat is `.reply(session, turns, timeout) -> (text, tokens,
# seconds)` plus `.reset(seed_idx)`, called once per dialog. Only a
# stateful seat (ELIZA) does anything in `reset`; the others declare
# it so `run_dialog` can call it unconditionally.


class UrlStudent:
    """A student behind a chat-completions endpoint -- the usual seat.

    Holds the URL, the participant record and the optional system
    prompt, so `run_dialog` has to know only "given the turns so far,
    what does the student say next". That is what lets the reference
    students below take the same chair without a server behind them.
    """

    def __init__(self, base_url: str, participant: dict,
                 system_prompt: str | None = None):
        self.base_url = base_url
        self.participant = participant
        self.system_prompt = system_prompt

    def reset(self, seed_idx: int | None = None):
        """Nothing to reset: the whole dialog goes in every request."""

    def reply(self, session, turns: list[dict],
              timeout) -> tuple[str, int, float]:
        return ask(session, self.base_url, self.participant,
                   student_messages(turns, self.system_prompt), timeout)


class PhraseStudent:
    """The null model: one weighted random draw, the context ignored.

    `turns` is accepted and deliberately unused -- not conditioning on
    it is the whole point of the seat. One `random.Random` serves the
    entire run, so `--phrase-seed` reproduces it end to end.
    """

    def __init__(self, phrases: list[str], weights: list[float],
                 participant: dict, seed: int = _DEFAULT_PHRASE_SEED):
        if not phrases:
            raise ValueError("the phrase bot needs at least one phrase")
        self.phrases = phrases
        self.weights = weights
        self.participant = participant
        self._rng = random.Random(seed)

    def reset(self, seed_idx: int | None = None):
        """Deliberately not reseeded per dialog: one stream serves the
        whole run, which is what `--phrase-seed` reproduces."""

    def reply(self, session, turns: list[dict],
              timeout) -> tuple[str, int, float]:
        text = self._rng.choices(self.phrases, self.weights)[0]
        return text, 0, 0.0


# " ?" and " ." come out of the port joining its output words with
# spaces; "??" out of a reassembly rule appending its own mark to an
# inserted phrase that already ended in one. Both are artifacts of the
# port's plumbing, not of the 1966 script.
_SPACES_RE = re.compile(r"\s+")
_DETOKEN_RE = re.compile(r"\s+([,.;:!?])")
_PUNCT_RUN_RE = re.compile(r"[,.;:!?]{2,}")


def tidy_eliza(text: str) -> str:
    """ELIZA's word list as a typed sentence.

    Collapses runs of spaces, pulls punctuation back onto the
    preceding word, and keeps only the last mark of a punctuation run.
    Presentation only -- no word is added, removed or reordered -- so
    that pass A grades the sentence rather than the port's spacing.
    """
    text = _DETOKEN_RE.sub(r"\1", _SPACES_RE.sub(" ", text))
    return _PUNCT_RUN_RE.sub(lambda m: m.group(0)[-1], text).strip()


class ElizaStudent:
    """Weizenbaum's DOCTOR in the student seat: responsive but empty.

    Reads only the last thing the examiner said -- which is all the
    1966 program ever read -- and answers from the vendored script. It
    is stateful in two ways that make the session, not the utterance,
    the unit: a decomposition rule cycles through its reassembly rules
    in order, and a turn that matches a `$` rule is pushed onto a
    memory stack for a later turn with no keyword to pop. So `reset`
    builds a *new* session, keyed on the dialog's seed index rather
    than on a call counter, and a resumed run reproduces the transcript
    of every dialog it did not already have.

    An utterance that is exactly a quit word ("bye") makes `respond`
    return None; the sign-off is the faithful answer there, and the
    dialog continues, because the eval always runs its 10 turns.
    """

    def __init__(self, participant: dict, seed: int = _DEFAULT_ELIZA_SEED):
        self.participant = participant
        self.seed = seed
        self._bot = None
        self.reset()

    def reset(self, seed_idx: int | None = None):
        self._bot = eliza.load_doctor(
            random.Random(self.seed + (seed_idx or 0)))

    def reply(self, session, turns: list[dict],
              timeout) -> tuple[str, int, float]:
        t0 = time.perf_counter()
        said = preceding_user_text(turns, len(turns))
        text = self._bot.respond(said)
        if text is None:
            text = self._bot.final()
        return tidy_eliza(text), 0, time.perf_counter() - t0


def eliza_participant(name: str, seed: int) -> dict:
    """The DOCTOR seat, recorded like every other participant.

    No model file and no sampling params: what reproduces this seat is
    the vendored script plus the seed, so that is what is written down.
    """
    return {
        "name": name,
        "kind": "eliza",
        "model": "eliza",
        "script": os.path.basename(eliza.DOCTOR_SCRIPT),
        "source": "github.com/wadetb/eliza@6055a7d (MIT)",
        "eliza_seed": seed,
    }


def load_phrase_file(path: str) -> tuple[list[str], list[float]]:
    """`{"phrases": [{"text", "weight"}, ...]}` -> texts and weights.

    The weights are the raw counts `phrases` writes; normalization
    happens at draw time (`random.choices` does it), so a hand-edited
    file may use any positive scale it likes.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("phrases") if isinstance(data, dict) else None
    if not entries:
        raise ValueError(f"no phrases in {path}")
    phrases, weights = [], []
    for entry in entries:
        weight = float(entry.get("weight", 1))
        if weight <= 0:
            raise ValueError(f"phrase weight must be > 0: {entry!r}")
        phrases.append(str(entry["text"]))
        weights.append(weight)
    return phrases, weights


def phrase_participant(name: str, path: str, seed: int,
                       n_phrases: int) -> dict:
    """The null-model seat, recorded like every other participant.

    No model file and no sampling params, because neither exists: what
    reproduces this seat is the phrase file plus the seed, so that is
    what the record carries.
    """
    return {
        "name": name,
        "kind": "phrases",
        "model": os.path.basename(path),
        "phrases_path": path,
        "phrase_seed": seed,
        "n_phrases": n_phrases,
    }


# ---------------------------------------------------------- generation


def run_dialog(session, seed: dict, examiner: dict, examiner_url: str,
               student, n_turns: int, timeout) -> list[dict]:
    """One eval dialog: forced opener, then alternating turns.

    `student` is any object with `.reply(session, turns, timeout)` and
    `.reset(seed_idx)` -- `UrlStudent`, `PhraseStudent` or
    `ElizaStudent`. The reset opens the dialog: a stateful seat must
    not carry a rule cursor or a memory stack over from the previous
    seed.
    """
    student.reset(seed["idx"])
    system_prompt = examiner["system_prompt"]
    turns = [{
        "side": _EXAMINER,
        "text": seed["opener"],
        "forced": True,
        "tokens": 0,
        "elapsed_s": 0.0,
    }]
    while len(turns) < n_turns:
        if turns[-1]["side"] == _EXAMINER:
            text, tokens, elapsed = student.reply(session, turns, timeout)
            side = _STUDENT
        else:
            text, tokens, elapsed = ask(
                session, examiner_url, examiner,
                examiner_messages(system_prompt, turns), timeout)
            side = _EXAMINER
        turns.append({
            "side": side,
            "text": text,
            "forced": False,
            "tokens": tokens,
            "elapsed_s": round(elapsed, 3),
        })
    return turns


def default_dialogs_path(student: str) -> str:
    stem = os.path.splitext(os.path.basename(student))[0]
    return os.path.join(
        _DEFAULT_OUT_DIR, f"dialogs-{stem}-{_stamp()}.jsonl")


def llama_participant(name: str, model_path: str | None, base_url: str | None,
                      system_prompt: str, temperature: float,
                      max_tokens: int, extra: dict = _NO_THINKING) -> dict:
    """A llama.cpp seat, recorded harness-style (provenance verbatim)."""
    participant = {
        "name": name,
        "model": os.path.basename(model_path) if model_path else name,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": extra,
    }
    if model_path:
        participant["model_path"] = model_path
    else:
        participant["base_url"] = base_url
    return participant


def texmo_participant(name: str, manifest: str | None, base_url: str | None,
                      temperature: float, max_tokens: int) -> dict:
    """A texmo seat. No system prompt: these models have no system
    channel (`texmo.chat_server` drops one)."""
    participant = {
        "name": name,
        "model": os.path.basename(manifest) if manifest else name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if manifest:
        participant["manifest"] = manifest
    else:
        participant["base_url"] = base_url
    return participant


def parse_temperatures(text: str) -> list[float]:
    """`--student-temperature` -> the sweep values, in order.

    A single value is a one-element sweep, so the sweep path is the
    only path. Duplicates are dropped (they would collide on the
    resumability key and the second would be skipped anyway).
    """
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0:
            raise ValueError(f"temperature must be >= 0, got {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("--student-temperature needs at least one value")
    return values


def plan_jobs(seeds: list[dict], temperatures: list[float],
              anchor_good: bool, anchor_garbage: bool,
              anchor_temperature: float) -> list[tuple]:
    """(anchor, seed, temperature) jobs: the sweep, then the anchors.

    Seeds vary fastest within a temperature, so an interrupted run
    leaves whole temperatures finished rather than every one partial.
    The anchors calibrate the judge, not the student, so they run once
    at `anchor_temperature` instead of across the sweep.
    """
    jobs = [(None, seed, t) for t in temperatures for seed in seeds]
    if anchor_good:
        jobs += [("good", seed, anchor_temperature)
                 for seed in seeds[:_ANCHOR_SEEDS]]
    if anchor_garbage:
        jobs += [("garbage", seed, anchor_temperature)
                 for seed in seeds[:_ANCHOR_SEEDS]]
    return jobs


def run_generate(args) -> int:
    seeds = load_seeds(args.seeds_file, args.n_seeds)
    temperatures = parse_temperatures(args.student_temperature)
    anchor_temperature = (args.anchor_temperature
                          if args.anchor_temperature is not None
                          else temperatures[0])
    phrases = weights = None
    if args.student_phrases:
        phrases, weights = load_phrase_file(args.student_phrases)
    if args.student_phrases or args.student_eliza:
        # Neither reference student has a sampling temperature, so the
        # sweep is one pass and the field is recorded as null rather
        # than as a number nothing used. The anchors keep their own.
        temperatures = [None]
    out_path = args.out or default_dialogs_path(
        args.student or args.student_phrases
        or ("eliza" if args.student_eliza else "student"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    jobs = plan_jobs(seeds, temperatures, args.anchor_good,
                     args.anchor_garbage, anchor_temperature)
    done = existing_keys(out_path, record_key)
    timeout = (dh._CONNECT_TIMEOUT, args.request_timeout)
    session = requests.Session()

    if args.anchor_garbage and not args.garbage_student:
        raise ValueError(
            "--anchor-garbage needs --garbage-student <manifest> "
            "(e.g. `texmo.py train --steps 2 -o scratch/untrained.json`)")
    if args.anchor_good and not (args.examiner or args.examiner_url):
        raise ValueError("--anchor-good needs the examiner model")

    servers: list[dh.Server] = []
    try:
        if args.examiner:
            examiner_server = launch_llama_server(
                args.examiner, out_path, _server_conf(args))
            servers.append(examiner_server)
            examiner_url = examiner_server.base_url
        else:
            examiner_url = args.examiner_url
        # One server per model for the whole sweep: temperature is a
        # per-request field, so the expensive loads are paid once. The
        # phrase bot needs no server at all.
        student_url = args.student_url
        if args.student:
            student_server = launch_chat_server(
                args.student, out_path, temperatures[0],
                args.max_reply_tokens)
            servers.append(student_server)
            student_url = student_server.base_url
        garbage_url = None
        if args.anchor_garbage:
            garbage_server = launch_chat_server(
                args.garbage_student, out_path, anchor_temperature,
                args.max_reply_tokens)
            servers.append(garbage_server)
            garbage_url = garbage_server.base_url

        for server in servers:
            dh.wait_until_ready(server, session, args.startup_timeout)
            logging.info(f"server on port {server.port} is ready")

        # Built once, outside the loop: the bot draws from a single
        # seeded stream for the whole run.
        phrase_student = None
        if phrases is not None:
            phrase_student = PhraseStudent(
                phrases, weights,
                phrase_participant(_STUDENT, args.student_phrases,
                                   args.phrase_seed, len(phrases)),
                args.phrase_seed)
        # Also built once, but re-seeded per dialog by `run_dialog`.
        eliza_student = None
        if args.student_eliza:
            eliza_student = ElizaStudent(
                eliza_participant(_STUDENT, args.eliza_seed),
                args.eliza_seed)

        written = skipped = 0
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            for i, (anchor, seed, temperature) in enumerate(jobs):
                if (anchor, seed["idx"], temperature) in done:
                    skipped += 1
                    continue
                examiner = llama_participant(
                    _EXAMINER, args.examiner, args.examiner_url,
                    examiner_system_prompt(seed), args.examiner_temperature,
                    args.examiner_max_tokens)
                if anchor == "good":
                    seat = llama_participant(
                        "good-student", args.examiner, args.examiner_url,
                        _GOOD_ANCHOR_PROMPT.format(bot=_BOT_LABEL),
                        temperature, args.examiner_max_tokens)
                    student = UrlStudent(
                        examiner_url, seat, seat["system_prompt"])
                elif anchor == "garbage":
                    seat = texmo_participant(
                        "garbage-student", args.garbage_student, None,
                        temperature, args.max_reply_tokens)
                    student = UrlStudent(garbage_url, seat)
                elif phrase_student is not None:
                    student = phrase_student
                    seat = phrase_student.participant
                elif eliza_student is not None:
                    student = eliza_student
                    seat = eliza_student.participant
                else:
                    seat = texmo_participant(
                        _STUDENT, args.student, args.student_url,
                        temperature, args.max_reply_tokens)
                    student = UrlStudent(student_url, seat)
                started_at = _now()
                t0 = time.time()
                try:
                    turns = run_dialog(
                        session, seed, examiner, examiner_url, student,
                        args.n_turns, timeout)
                except KeyboardInterrupt:
                    print(f"\ninterrupted during job {i + 1}; "
                          f"{written} dialog(s) written to {out_path}")
                    break
                record = {
                    "seed_idx": seed["idx"],
                    "anchor": anchor,
                    "student_temperature": temperature,
                    "turns": turns,
                    "participants": {_EXAMINER: examiner, _STUDENT: seat},
                    "started_at": started_at,
                    "ended_at": _now(),
                }
                dh.write_record(f, record)
                written += 1
                label = anchor or "main"
                print(f"[{i + 1}/{len(jobs)}] {label} seed={seed['idx']} "
                      f"T={temperature} turns={len(turns)} "
                      f"{time.time() - t0:.1f}s")
        print(f"wrote {written} dialog(s) ({skipped} already present) "
              f"to {out_path}")
    finally:
        _stop_all(servers)
    return 0


# -------------------------------------------------------- phrase table


def count_answers(dialogs: list[dict]) -> collections.Counter:
    """Exact (stripped) student answers over the non-anchor dialogs.

    The anchors are a different student -- the examiner model, or an
    untrained one -- so their answers would poison a distillation of
    the model under test.
    """
    counts = collections.Counter()
    for record in dialogs:
        if record.get("anchor") is not None:
            continue
        for turn in record["turns"]:
            if turn["side"] == _STUDENT:
                counts[turn["text"].strip()] += 1
    return counts


def phrase_table(counts: collections.Counter, top_n: int,
                 dialogs_path: str) -> dict:
    """The top-N phrases plus the provenance of the distillation.

    `weight` is the raw count: the bot normalizes at draw time, so the
    file doubles as a readable frequency table. `coverage` is the share
    of all answers the kept phrases account for -- the part of the
    student's behaviour the null model actually reproduces.
    """
    total = sum(counts.values())
    top = counts.most_common(top_n)
    return {
        "source_dialogs": dialogs_path,
        "generated": _now(),
        "n_answers": total,
        "n_distinct": len(counts),
        "top_n": top_n,
        "coverage": round(sum(c for _, c in top) / total, 4) if total else 0.0,
        "phrases": [{"text": text, "weight": count} for text, count in top],
    }


def format_phrase_table(table: dict) -> str:
    """The same table for the terminal, one line per phrase."""
    total = table["n_answers"] or 1
    lines = [f"{'#':>3}  {'count':>5}  {'share':>6}  text"]
    for i, entry in enumerate(table["phrases"], 1):
        lines.append(
            f"{i:>3}  {entry['weight']:>5}  "
            f"{100.0 * entry['weight'] / total:>5.1f}%  {entry['text']!r}")
    lines.append(
        f"\n{table['n_distinct']} distinct answers over "
        f"{table['n_answers']}; the top {len(table['phrases'])} cover "
        f"{100.0 * table['coverage']:.1f}%.")
    return "\n".join(lines)


def run_phrases(args) -> int:
    dialogs = read_records(args.dialogs)
    if not dialogs:
        raise ValueError(f"no dialogs in {args.dialogs}")
    counts = count_answers(dialogs)
    if not counts:
        raise ValueError(f"no student answers in {args.dialogs}")
    table = phrase_table(counts, args.top, args.dialogs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        save_json(table, f)
    print(format_phrase_table(table))
    print(f"\nwrote {len(table['phrases'])} phrase(s) to {args.out}")
    return 0


# ------------------------------------------------------------- judging


def student_positions(turns: list[dict]) -> list[tuple[int, int]]:
    """(position, turn index) for every student answer, position 1-based."""
    out = []
    for i, turn in enumerate(turns):
        if turn["side"] == _STUDENT:
            out.append((len(out) + 1, i))
    return out


def grade_pass(session, base_url: str, judge: dict, pass_name: str,
               turns: list[dict], upto: int, timeout) -> dict:
    """One pass over one answer; one retry, then `judge_error`.

    The returned dict carries the pass's verdict key(s), its `comment`,
    the raw reply and the seconds spent (both attempts, if there were
    two).
    """
    spec = _PASS_SPECS[pass_name]
    error = None
    raw = ""
    spent = 0.0
    for attempt in range(2):
        messages = judge_messages(
            pass_name, turns, upto, error if attempt else None)
        raw, _, elapsed = ask(session, base_url, judge, messages, timeout)
        spent += elapsed
        try:
            grade = parse_judge_output(raw, spec["keys"])
        except ValueError as e:
            error = str(e)
            logging.warning(f"judge pass {pass_name} parse failure "
                            f"({error}); "
                            f"{'retrying' if not attempt else 'giving up'}")
            continue
        grade["judge_error"] = None
        grade["raw"] = raw
        grade["elapsed_s"] = round(spent, 3)
        return grade
    grade = {key: None for key in spec["keys"]}
    grade.update({"examiner_defect": None, "comment": "",
                  "judge_error": error, "raw": raw,
                  "elapsed_s": round(spent, 3)})
    return grade


def merge_grades(results: dict) -> dict:
    """The three passes' results as one grade record.

    `c` is the headline substantive rate and is `c_raw and b`: pass C
    is no longer asked whether the reply fits the turn, so a reply that
    pass B rejected cannot be substantive whatever pass C thought.
    Either verdict missing (a judge error) leaves `c` unknown.
    """
    a, b, c = (results[p] for p in _PASSES)
    c_raw = c.get("c")
    grade = {
        "a": a.get("a"),
        "b": b.get("b"),
        "c_raw": c_raw,
        "c": None if c_raw is None or b.get("b") is None
             else bool(c_raw and b["b"]),
        "examiner_defect": c.get("examiner_defect"),
        "comment_a": a.get("comment", ""),
        "comment_b": b.get("comment", ""),
        "comment_c": c.get("comment", ""),
        "raw": {p: results[p].get("raw", "") for p in _PASSES},
        "elapsed_s": {p: results[p].get("elapsed_s", 0.0) for p in _PASSES},
    }
    errors = {p: results[p].get("judge_error") for p in _PASSES
              if results[p].get("judge_error")}
    grade["judge_error"] = (
        None if not errors
        else "; ".join(f"{p}: {e}" for p, e in sorted(errors.items())))
    grade["c_without_b"] = is_c_without_b(grade)
    return grade


def run_pass(pass_name: str, tasks: list[tuple], base_url: str, judge: dict,
             timeout, parallel: int, session_factory=requests.Session,
             progress=None) -> dict:
    """One pass over many answers, `parallel` requests in flight.

    `tasks` are `(key, turns, upto)`; the result is `{key: grade}`, so
    the aggregation does not depend on completion order. Each worker
    thread keeps its own session (`requests.Session` is not designed
    to be shared), and `session_factory` is what the tests stub out.

    A Ctrl-C cancels whatever has not started; the pool still waits out
    the requests already in flight, which is one per slot.
    """
    local = threading.local()

    def work(task):
        key, turns, upto = task
        session = getattr(local, "session", None)
        if session is None:
            session = local.session = session_factory()
        return key, grade_pass(
            session, base_url, judge, pass_name, turns, upto, timeout)

    results = {}
    if parallel <= 1:
        for task in tasks:
            key, grade = work(task)
            results[key] = grade
            if progress is not None:
                progress(len(results), len(tasks))
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        try:
            for future in concurrent.futures.as_completed(futures):
                key, grade = future.result()
                results[key] = grade
                if progress is not None:
                    progress(len(results), len(tasks))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def default_grades_path(dialogs: str) -> str:
    stem, _ = os.path.splitext(dialogs)
    return stem + "-grades.jsonl"


def default_report_path(dialogs: str) -> str:
    stem, _ = os.path.splitext(dialogs)
    return stem + "-report.md"


def limit_seeds(dialogs: list[dict], n_seeds: int | None) -> list[dict]:
    """The dialogs of the first `n_seeds` distinct seeds, anchors kept.

    Sampling a big dialogs file down without regenerating it. Seed
    identity comes from the file itself (`seed_idx` in order of first
    appearance), not from an index range, so it works on any file; the
    anchors are only a handful of dialogs and calibrate the judge, so
    they always ride along.
    """
    if n_seeds is None:
        return dialogs
    keep = []
    for record in dialogs:
        if record.get("anchor") is None and record["seed_idx"] not in keep:
            keep.append(record["seed_idx"])
            if len(keep) >= n_seeds:
                break
    wanted = set(keep)
    return [r for r in dialogs
            if r.get("anchor") is not None or r["seed_idx"] in wanted]


def plan_answers(dialogs: list[dict], done: set) -> list[tuple]:
    """Every ungraded student answer as `(key, record, position, index)`."""
    answers = []
    for record in dialogs:
        turns = record["turns"]
        for position, index in student_positions(turns):
            key = record_key(record) + (position,)
            if key in done:
                continue
            answers.append((key, record, position, index))
    return answers


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def judge_seats(args) -> dict:
    """One recorded seat per pass -- they differ only in the reply cap.

    The system prompt is per-request and per-pass (`judge_messages`),
    so the seat carries only the sampling params, the no-thinking rule
    and `cache_prompt`.
    """
    seats = {}
    for pass_name in _PASSES:
        cap = _PASS_SPECS[pass_name]["max_tokens"] or args.judge_max_tokens
        seat = llama_participant(
            "judge", args.judge, args.judge_url, "",
            args.judge_temperature, cap, _JUDGE_EXTRA)
        seat.pop("system_prompt")
        seats[pass_name] = seat
    return seats


def run_phase(args, phase: str, passes: tuple, tasks: list[tuple],
              seats: dict, session, out_path: str, timeout,
              label: str) -> tuple[dict, dict]:
    """One phase's passes against a server sized for them.

    Returns `({pass: {key: result}}, {pass: wall seconds})`. The server
    is started here and stopped here -- including on a Ctrl-C, which is
    the whole reason for the `finally`. With `--judge-url` nothing is
    launched: someone else's endpoint serves every phase, its slot
    count is unknown, so concurrency falls back to the narrower
    `--parallel-c`.
    """
    conf, parallel = phase_server_conf(args, phase)
    results, wall = {}, {}
    server = None
    try:
        if args.judge:
            server = launch_llama_server(args.judge, out_path, conf)
            dh.wait_until_ready(server, session, args.startup_timeout)
            logging.info(f"server on port {server.port} is ready "
                         f"({' '.join(conf['args'])})")
            url = server.base_url
        else:
            url = args.judge_url
            parallel = min(parallel, args.parallel_c)
        for pass_name in passes:
            t0 = time.perf_counter()
            results[pass_name] = run_pass(
                pass_name, tasks, url, seats[pass_name], timeout, parallel)
            wall[pass_name] = time.perf_counter() - t0
            rate = len(tasks) / wall[pass_name] if wall[pass_name] else 0.0
            print(f"{label} pass {pass_name}: {len(tasks)} answers in "
                  f"{wall[pass_name]:.1f}s ({rate:.1f} answers/s, "
                  f"{parallel} slots)")
    finally:
        if server is not None:
            _stop_all([server])
    return results, wall


def run_judge(args) -> int:
    dialogs = limit_seeds(read_records(args.dialogs), args.n_seeds)
    if not dialogs:
        raise ValueError(f"no dialogs in {args.dialogs}")
    out_path = args.out or default_grades_path(args.dialogs)
    report_path = args.report or default_report_path(args.dialogs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = existing_keys(out_path, grade_key)
    timeout = (dh._CONNECT_TIMEOUT, args.request_timeout)
    session = requests.Session()
    seats = judge_seats(args)
    todo = plan_answers(dialogs, done)
    skipped = sum(len(student_positions(r["turns"])) for r in dialogs) \
        - len(todo)
    wall = {p: 0.0 for p in _PASSES}
    written = 0

    try:
        chunks = _chunks(todo, args.chunk)
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            for chunk_no, chunk in enumerate(chunks, 1):
                tasks = [(key, record["turns"], index)
                         for key, record, _, index in chunk]
                results = {}
                for phase, passes in _PHASES:
                    part, spent = run_phase(
                        args, phase, passes, tasks, seats, session, out_path,
                        timeout, f"[chunk {chunk_no}/{len(chunks)}]")
                    results.update(part)
                    for pass_name, seconds in spent.items():
                        wall[pass_name] += seconds
                for key, record, position, index in chunk:
                    answer = record["turns"][index]["text"]
                    grade = merge_grades(
                        {p: results[p][key] for p in _PASSES})
                    grade.update({
                        "seed_idx": record["seed_idx"],
                        "anchor": record.get("anchor"),
                        "student_temperature":
                            record.get("student_temperature"),
                        "position": position,
                        "answer": answer,
                        "repetition": repetition_stats(answer),
                        "judge_model": seats["c"]["model"],
                    })
                    dh.write_record(f, grade)
                    written += 1
    except KeyboardInterrupt:
        print(f"\ninterrupted; {written} grade(s) in {out_path}")

    grades = read_records(out_path)
    report = build_report(args.dialogs, out_path, grades,
                          student_kind(dialogs), wall)
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print(f"wrote {written} grade(s) ({skipped} already present) to "
          f"{out_path}\nreport: {report_path}")
    return 0


# ------------------------------------------------------------ reporting


def _rate(grades: list[dict], key: str) -> float | None:
    valid = [g for g in grades if g.get(key) is not None]
    if not valid:
        return None
    return 100.0 * sum(1 for g in valid if g[key]) / len(valid)


def _fmt(value, unit="%") -> str:
    return "n/a" if value is None else f"{value:.1f}{unit}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pass_latency(grades: list[dict], pass_name: str) -> float | None:
    """Mean seconds one pass spent per answer, as the client saw it."""
    values = [g["elapsed_s"][pass_name] for g in grades
              if isinstance(g.get("elapsed_s"), dict)
              and pass_name in g["elapsed_s"]]
    return _mean(values)


def summarize(grades: list[dict]) -> dict:
    """Rates and counts for one group of grades.

    Rates skip a missing verdict per criterion rather than dropping the
    whole answer: the three passes fail independently, so a pass-A
    parse failure must not cost the b and c that did come back.
    """
    valid = [g for g in grades if g.get("judge_error") is None]
    rates = {k: _rate(grades, k) for k in _CRITERIA}
    deflection = None
    if rates["b"] is not None and rates["c"] is not None:
        deflection = rates["b"] - rates["c"]
    return {
        "n": len(grades),
        "n_valid": len(valid),
        "rates": rates,
        "c_raw": _rate(grades, "c_raw"),
        "deflection": deflection,
        "errors": sum(1 for g in grades if g.get("judge_error") is not None),
        "c_without_b": sum(1 for g in grades if g.get("c_without_b")),
        "latency": {p: _pass_latency(grades, p) for p in _PASSES},
        "rep3": _mean([g["repetition"]["rep3"] for g in grades
                       if "repetition" in g]),
        "lrs_ratio": _mean([g["repetition"]["lrs_ratio"] for g in grades
                            if "repetition" in g]),
        "chars": _mean([g["repetition"]["chars"] for g in grades
                        if "repetition" in g]),
    }


def verdict(summary: dict) -> str:
    """PASS only if every criterion clears its target."""
    rates = summary["rates"]
    if any(rates[k] is None for k in _CRITERIA):
        return "n/a"
    return ("PASS" if all(rates[k] >= _TARGETS[k] for k in _CRITERIA)
            else "FAIL")


def _temperatures(grades: list[dict]) -> list:
    """The temperatures present, sorted; `None` (older files) last."""
    values = {g.get("student_temperature") for g in grades}
    known = sorted(v for v in values if v is not None)
    return known + ([None] if None in values else [])


def _summary_row(label, summary: dict) -> str:
    rates = summary["rates"]
    return (f"| {label} | {summary['n']} | {_fmt(rates['a'])} | "
            f"{_fmt(rates['b'])} | {_fmt(rates['c'])} | "
            f"{_fmt(summary['deflection'])} | "
            f"{_fmt(summary['rep3'], '')} | {verdict(summary)} |")


_SUMMARY_HEADER = [
    "| T | n | a | b | c | b-c | rep3 | verdict |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]


def student_kind(dialogs: list[dict]) -> str | None:
    """The student seat's `kind`, from the first non-anchor dialog.

    Only the null model sets one (`phrases`), and that is exactly what
    the report needs to know: for that student, pass A is a judge
    calibration number rather than a measurement of the student.
    """
    for record in dialogs:
        if record.get("anchor") is None:
            seat = (record.get("participants") or {}).get(_STUDENT) or {}
            return seat.get("kind")
    return None


def _calibration_lines(summary: dict) -> list[str]:
    """The phrase bot's pass-A rate, called what it is."""
    return [
        "## Judge calibration (pass A on the null phrase bot)",
        "",
        "Every answer here is a sentence a real model produced and this "
        "judge already passed, drawn at random. Pass A sees the reply "
        "alone, so its rate measures the *judge*, not the student: it "
        "should read ~100%, and the gap below 100% is judge noise that "
        "lands on every other file's `a` as well.",
        "",
        f"**pass A: {_fmt(summary['rates']['a'])}** over {summary['n']} "
        f"answers.",
        "",
    ]


def _timing_lines(summary: dict, wall: dict | None) -> list[str]:
    """Per-pass cost: request latency from the grades, wall from the run."""
    lines = [
        "## Judge cost, per pass",
        "",
        "`latency` is the mean seconds one request took (retries "
        "included); `wall` is this run's elapsed time for the pass, "
        "which is what `--parallel-ab` (a, b) and `--parallel-c` (c) "
        "shrink. Every request sends `cache_prompt: true` and a "
        "byte-identical system prompt, so the server keeps the shared "
        "prefix in the slot's KV and only the utterance, the pair or "
        "the dialog is evaluated.",
        "",
        "| pass | mean latency | wall this run |",
        "| --- | --- | --- |",
    ]
    for pass_name in _PASSES:
        elapsed = (wall or {}).get(pass_name)
        lines.append(
            f"| {pass_name} | {_fmt(summary['latency'][pass_name], 's')} | "
            f"{'n/a' if not elapsed else f'{elapsed:.1f}s'} |")
    lines.append("")
    return lines


def build_report(dialogs_path: str, grades_path: str, grades: list[dict],
                 kind: str | None = None, wall: dict | None = None) -> str:
    """The markdown report: rates per temperature, positions, anchors."""
    main = [g for g in grades if g.get("anchor") is None]
    overall = summarize(main)
    lines = [
        "# Scripted-examiner eval report",
        "",
        f"- dialogs: `{dialogs_path}`",
        f"- grades: `{grades_path}`",
        f"- judge: `{main[0]['judge_model'] if main else 'n/a'}`",
        f"- generated: {_now()}",
        "",
    ]
    if kind == "phrases" and main:
        lines += _calibration_lines(overall)
    lines += [
        "## Main set, by student temperature",
        "",
        f"{overall['n']} student answers "
        f"({overall['n_valid']} with a valid grade). Targets: "
        + ", ".join(f"{k} >= {_TARGETS[k]:.0f}%" for k in _CRITERIA)
        + "; `verdict` is PASS only when all three clear. Each criterion "
        "is judged in its own request, on its own context: a sees the "
        "reply alone, b the reply and the turn before it, c the whole "
        "dialog. The reported `c` is c_raw AND b.",
        "",
    ] + list(_SUMMARY_HEADER)
    for temperature in _temperatures(main):
        group = [g for g in main
                 if g.get("student_temperature") == temperature]
        lines.append(_summary_row(temperature, summarize(group)))
    lines += [
        "",
        f"- judge errors: {overall['errors']}",
        f"- pass disagreements (c_raw without b): "
        f"{overall['c_without_b']}",
        "",
    ] + _timing_lines(overall, wall) + [
        "## Repetition (LLM-free)",
        "",
        "`rep3` is the fraction of repeated word 3-grams "
        "(`1 - unique/total`); `lrs_ratio` is the longest substring "
        "occurring twice, over the answer length. Both are 0 for a "
        "clean answer and approach 1 for a stuck one.",
        "",
        "| T | rep3 | lrs_ratio | mean chars |",
        "| --- | --- | --- | --- |",
    ]
    for temperature in _temperatures(main):
        s = summarize([g for g in main
                       if g.get("student_temperature") == temperature])
        lines.append(
            f"| {temperature} | {_fmt(s['rep3'], '')} | "
            f"{_fmt(s['lrs_ratio'], '')} | {_fmt(s['chars'], '')} |")
    lines += [
        "",
        "## By turn position",
        "",
        "| T | position | n | a | b | c | rep3 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for temperature in _temperatures(main):
        group = [g for g in main
                 if g.get("student_temperature") == temperature]
        for position in sorted({g["position"] for g in group}):
            s = summarize([g for g in group if g["position"] == position])
            lines.append(
                f"| {temperature} | {position} | {s['n']} | "
                f"{_fmt(s['rates']['a'])} | {_fmt(s['rates']['b'])} | "
                f"{_fmt(s['rates']['c'])} | {_fmt(s['rep3'], '')} |")

    anchors = sorted({g["anchor"] for g in grades
                      if g.get("anchor") is not None})
    lines += ["", "## Calibration anchors", ""]
    if not anchors:
        lines.append("None in this run.")
    else:
        lines += [
            "Run once at a single temperature: they calibrate the "
            "judge, not the student. Expected ordering: good far above "
            "garbage on all three. If it does not hold, the judge is "
            "broken and the main numbers mean nothing.",
            "",
            "| anchor | T | n | a | b | c | rep3 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for anchor in anchors:
            group = [g for g in grades if g.get("anchor") == anchor]
            s = summarize(group)
            temps = ",".join(str(t) for t in _temperatures(group))
            lines.append(
                f"| {anchor} | {temps} | {s['n']} | "
                f"{_fmt(s['rates']['a'])} | {_fmt(s['rates']['b'])} | "
                f"{_fmt(s['rates']['c'])} | {_fmt(s['rep3'], '')} |")

    defects = [g for g in grades if g.get("examiner_defect")]
    lines += ["", "## Examiner defects", ""]
    if not defects:
        lines.append("None reported.")
    else:
        lines.append(f"{len(defects)} of {len(grades)} answers "
                     f"({100.0 * len(defects) / len(grades):.1f}%) came "
                     f"with a note on the User side:")
        lines.append("")
        for g in defects:
            anchor = g.get("anchor") or "main"
            lines.append(f"- {anchor} seed {g['seed_idx']} pos "
                         f"{g['position']}: {g['examiner_defect']}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ cli


def _add_server_args(parser):
    parser.add_argument(
        "--llama-binary", default=None,
        help="llama-server executable (default: $LLAMA_SERVER, then PATH, "
             "then the known local install)")
    parser.add_argument(
        "--ngl", type=int, default=99,
        help="GPU layers for launched llama-servers (default: 99)")
    parser.add_argument(
        "--ctx-size", type=int, default=_DEFAULT_CTX,
        help=f"llama-server context size (default: {_DEFAULT_CTX})")
    parser.add_argument(
        "--startup-timeout", type=float, default=_DEFAULT_STARTUP_TIMEOUT,
        help="seconds to wait for a launched server to answer /health "
             f"(default: {_DEFAULT_STARTUP_TIMEOUT}; a cold 12B Q6 load "
             "is minutes)")
    parser.add_argument(
        "--request-timeout", type=float, default=300.0,
        help="read timeout per request in seconds (default: 300)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scripted-examiner eval for texmo chat models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "generate", help="run examiner/student dialogs from seeds")
    gen.add_argument(
        "--seeds-file", default=_DEFAULT_SEEDS,
        help=f"seed dialogs JSONL (default: {_DEFAULT_SEEDS})")
    gen.add_argument(
        "--n-seeds", type=int, default=10,
        help="how many seeds to run, from the top of the file (default: 10)")
    student = gen.add_mutually_exclusive_group(required=True)
    student.add_argument(
        "--student", default=None,
        help="texmo model manifest; a chat-server is launched for it")
    student.add_argument(
        "--student-url", default=None,
        help="an already-running chat-completions endpoint for the student")
    student.add_argument(
        "--student-phrases", default=None,
        help="null model: a `phrases` JSON file whose entries are drawn at "
             "random, one per turn, ignoring the conversation entirely")
    student.add_argument(
        "--student-eliza", action="store_true",
        help="reference model: ELIZA (Weizenbaum's DOCTOR script, vendored "
             "in scripts/eliza/), one session per dialog -- zero weights, "
             "reflects the examiner's own words back")
    gen.add_argument(
        "--phrase-seed", type=int, default=_DEFAULT_PHRASE_SEED,
        help=f"RNG seed for --student-phrases (default: "
             f"{_DEFAULT_PHRASE_SEED})")
    gen.add_argument(
        "--eliza-seed", type=int, default=_DEFAULT_ELIZA_SEED,
        help=f"base RNG seed for --student-eliza; a dialog's session "
             f"draws from seed + its seed index (default: "
             f"{_DEFAULT_ELIZA_SEED})")
    examiner = gen.add_mutually_exclusive_group(required=True)
    examiner.add_argument(
        "--examiner", default=None,
        help="examiner GGUF; a llama-server is launched for it")
    examiner.add_argument(
        "--examiner-url", default=None,
        help="an already-running chat-completions endpoint for the examiner")
    gen.add_argument(
        "--student-temperature", default="0.5",
        help="student sampling temperature, or a comma-separated sweep "
             "(e.g. '0.3,0.5,0.7'): every seed runs once per value, with "
             "the servers kept up (default: 0.5)")
    gen.add_argument(
        "--anchor-temperature", type=float, default=None,
        help="single temperature for the calibration anchors (default: "
             "the first --student-temperature value)")
    gen.add_argument(
        "--examiner-temperature", type=float, default=0.7,
        help="examiner sampling temperature (default: 0.7)")
    gen.add_argument(
        "--max-reply-tokens", type=int, default=_DEFAULT_STUDENT_MAX_TOKENS,
        help=f"cap on a student reply (default: "
             f"{_DEFAULT_STUDENT_MAX_TOKENS})")
    gen.add_argument(
        "--examiner-max-tokens", type=int,
        default=_DEFAULT_EXAMINER_MAX_TOKENS,
        help=f"cap on an examiner utterance (default: "
             f"{_DEFAULT_EXAMINER_MAX_TOKENS}; the prompt asks for 1-2 "
             "short sentences)")
    gen.add_argument(
        "--n-turns", type=int, default=_N_TURNS,
        help=f"utterances per dialog, both sides (default: {_N_TURNS})")
    gen.add_argument(
        "--anchor-good", action="store_true",
        help=f"also run the examiner model as the student on the first "
             f"{_ANCHOR_SEEDS} seeds")
    gen.add_argument(
        "--anchor-garbage", action="store_true",
        help=f"also run --garbage-student on the first {_ANCHOR_SEEDS} seeds")
    gen.add_argument(
        "--garbage-student", default=None,
        help="manifest of an untrained texmo model for the garbage anchor")
    gen.add_argument(
        "--out", default=None,
        help="output JSONL (default: "
             "data/eval/dialogs-<student stem>-<timestamp>.jsonl)")
    _add_server_args(gen)
    gen.set_defaults(func=run_generate)

    jud = subparsers.add_parser(
        "judge", help="grade the student answers of a dialogs file")
    jud.add_argument(
        "--dialogs", required=True, help="a `generate` output JSONL")
    judge = jud.add_mutually_exclusive_group(required=True)
    judge.add_argument(
        "--judge", default=None,
        help="judge GGUF; must be a different model from the examiner")
    judge.add_argument(
        "--judge-url", default=None,
        help="an already-running chat-completions endpoint for the judge")
    jud.add_argument(
        "--judge-temperature", type=float, default=0.0,
        help="judge sampling temperature (default: 0, near-deterministic)")
    jud.add_argument(
        "--judge-max-tokens", type=int, default=_DEFAULT_JUDGE_MAX_TOKENS,
        help=f"cap on a judge reply (default: {_DEFAULT_JUDGE_MAX_TOKENS})")
    jud.add_argument(
        "--parallel-ab", type=int, default=_DEFAULT_PARALLEL_AB,
        help=f"slots (-np) and in-flight requests for passes a and b, "
             f"whose prompts are a cached prefix plus a line or two "
             f"(default: {_DEFAULT_PARALLEL_AB})")
    jud.add_argument(
        "--ctx-ab", type=int, default=_DEFAULT_CTX_AB,
        help=f"per-slot context for passes a and b (default: "
             f"{_DEFAULT_CTX_AB}); the server is started with "
             "--ctx-ab * --parallel-ab")
    jud.add_argument(
        "--parallel-c", type=int, default=_DEFAULT_PARALLEL_C,
        help=f"slots and in-flight requests for pass c, which carries "
             f"a whole dialog at --ctx-size per slot (default: "
             f"{_DEFAULT_PARALLEL_C})")
    jud.add_argument(
        "--chunk", type=int, default=_DEFAULT_CHUNK,
        help=f"answers taken through all three passes before anything is "
             f"written (default: {_DEFAULT_CHUNK}); a Ctrl-C costs at most "
             "one chunk")
    jud.add_argument(
        "--n-seeds", type=int, default=None,
        help="grade only the first N distinct seeds of the dialogs file "
             "(default: all); the anchors are always included")
    jud.add_argument(
        "--out", default=None,
        help="grades JSONL (default: <dialogs stem>-grades.jsonl)")
    jud.add_argument(
        "--report", default=None,
        help="markdown report (default: <dialogs stem>-report.md)")
    _add_server_args(jud)
    jud.set_defaults(func=run_judge)

    phr = subparsers.add_parser(
        "phrases",
        help="distil a dialogs file into the student's most common answers")
    phr.add_argument(
        "--dialogs", required=True, help="a `generate` output JSONL")
    phr.add_argument(
        "--top", type=int, default=_DEFAULT_TOP_N,
        help=f"how many answers to keep (default: {_DEFAULT_TOP_N})")
    phr.add_argument(
        "--out", required=True,
        help="phrase JSON, the input for `generate --student-phrases`")
    phr.set_defaults(func=run_phrases)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
