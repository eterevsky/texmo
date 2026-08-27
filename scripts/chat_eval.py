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

    uv run python scripts/chat_eval.py generate --n-seeds 100 \\
        --student models/hb32-8k-1ub.json \\
        --student-temperature 0.3,0.5,0.7 \\
        --examiner C:/Users/oleg/models/gemma-4-12b-it-Q6_K.gguf \\
        --anchor-good --anchor-garbage \\
        --garbage-student scratch/untrained.json

    uv run python scripts/chat_eval.py judge \\
        --dialogs data/eval/dialogs-hb32-8k-1ub-20260826-120000.jsonl \\
        --judge C:/Users/oleg/models/Qwen3-8B-Q6_K.gguf

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

## Judging

One request per student answer, carrying the dialog up to and
including that answer with `User:` / `Bot:` labels, the three criteria
as separate yes/no questions, and an example pair for "substantive".
Output must be strict JSON; a parse failure gets one retry with an
error-correcting suffix and is then recorded as `judge_error`.
Markdown fences and `<think>` blocks are stripped first (Qwen3 habits).

Four things in `_JUDGE_PROMPT` are load-bearing, each of them a fix
for a failure the anchors caught with Qwen3-8B (2026-08-26) -- do not
tidy them away without re-running the anchors:

- **`comment` comes first in the requested JSON.** Keys are generated
  in order, so describing the reply before voting makes the comment a
  one-line chain of thought. Without it the judge went straight to
  `"a": true`.
- **Criterion (a) names word salad explicitly.** With only "is it
  grammatically correct?", the garbage anchor -- literal random bytes
  -- passed (a) on 14 of 15 answers. Spelling out that unreadable
  text, invented words and fluent-looking word salad are all `false`
  took it to 0 of 15, while the good anchor stayed at 100%.
- **(a) also says a dull-but-clean sentence passes.** The clause
  above overshot: after five incoherent turns the judge failed a plain
  `"I really like it."` on (a) too, letting the conversation drag down
  a criterion that is defined on the reply alone. The counter-example
  restored it (that answer now grades a/b/c all true) without moving
  either anchor.
- **The defect key is called `user_problem` in the prompt** (the
  record still calls it `examiner_defect`, and the parser takes either
  name). Named after the model under test, the field collected remarks
  about the *student* on 73% of answers; named after the side it is
  about, and told that an incoherent Bot is never a User problem, it
  goes quiet on clean dialogs and still fires on a User that repeats
  itself.

Two checks ride along without an LLM:

- `c` without `b` is impossible by construction (c is b plus
  substance). Violations are counted as `judge_inconsistency`; the
  grade is kept as returned and marked.
- an n-gram repetition rate per answer: `rep3` is the fraction of
  repeated word 3-grams (`1 - unique/total`), and `lrs_ratio` the
  length of the longest substring occurring twice divided by the
  answer length. Both are 0 for a clean short answer and approach 1
  for a stuck one.

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
import datetime
import json
import logging
import os
import re
import sys
import time

import dialog_harness as dh
import requests

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

# Prepended to the examiner's messages so the roles alternate from a
# user turn -- see the module docstring.
_KICKOFF = "Start the conversation."
# Stands in for an empty utterance in message assembly only.
_EMPTY_PLACEHOLDER = "..."

# Working targets (docs/roadmap.md, 2026-08-26), in percent.
_TARGETS = {"a": 90.0, "b": 90.0, "c": 50.0}
_CRITERIA = ("a", "b", "c")

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

_JUDGE_PROMPT = """\
You are grading one reply from a very small, weak chatbot called \
"{bot}" that is having a casual conversation with a person called \
"{user}". You will see the conversation so far; the LAST line is the \
{bot} reply you must grade. Grade only that line.

Answer three yes/no questions about it:

a) Is the reply internally consistent and grammatically correct? \
Judge the reply on its own, ignoring the rest of the conversation: is \
it readable English, and does it avoid contradicting itself? Answer \
false whenever it is not readable English at all -- random characters, \
word fragments, invented words, a garbled loop -- or when it breaks \
off mid-thought, repeats itself, or states two incompatible things. A \
weak model often produces fluent-looking word salad: that is false, \
not true. But a short, plain, well-formed sentence such as "I really \
like it." is readable and self-consistent, so a is TRUE even when the \
reply is vague, dull or a poor answer to {user} -- that is what b and \
c are for. Do not let a bad conversation drag a down.

b) Is the reply consistent with the preceding turn? Does it make \
sense as a response to what {user} just said?

c) Is the reply also substantive -- a contentful, on-topic answer \
rather than a generic deflection? For example, to "What do you like \
to do on weekends?": "I don't know." is NOT substantive (it can still \
pass a and b), while "I usually go hiking with my brother." IS \
substantive.

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
the three verdicts, not the other way round:

{{"comment": <one short sentence describing the {bot} reply>, \
"a": <true or false>, "b": <true or false>, "c": <true or false>, \
"user_problem": <a short phrase, or null>}}"""

_RETRY_SUFFIX = """\
Your previous answer could not be parsed as JSON ({error}). Reply \
again with the JSON object only: keys "comment" (string), "a", "b", \
"c" (true/false), "user_problem" (string or null). No fences, no \
prose."""


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


def judge_messages(turns: list[dict], upto: int,
                   retry_error: str | None = None) -> list[dict]:
    """One grading request: the criteria, then the dialog.

    `retry_error` appends the error-correcting suffix used for the
    single retry after a parse failure.
    """
    system = _JUDGE_PROMPT.format(user=_USER_LABEL, bot=_BOT_LABEL)
    user = (f"Conversation so far (grade the last {_BOT_LABEL} line):\n\n"
            f"{render_dialog(turns, upto)}")
    if retry_error is not None:
        user += "\n\n" + _RETRY_SUFFIX.format(error=retry_error)
    return [
        {"role": "system", "content": system},
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


def parse_judge_output(text: str) -> dict:
    """Strict-JSON grade out of the judge's raw reply.

    Raises ValueError on anything that is not an object with boolean
    a/b/c. Tolerates fences, a leaked thinking block, and prose around
    the object -- all failure modes seen from local judges, none of
    them a reason to throw away a good grade.
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
    for key in _CRITERIA:
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


def is_inconsistent(grade: dict) -> bool:
    """`c` without `b` -- impossible by construction, so a judge error.

    Only c subset b holds; a and b are independent axes, so no other
    combination is checkable.
    """
    return bool(grade.get("c")) and not bool(grade.get("b"))


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


def _server_conf(args) -> dict:
    """The `server` block `dialog_harness` expects, from our flags."""
    conf = {"ngl": args.ngl, "args": ["-c", str(args.ctx_size)]}
    if args.llama_binary:
        conf["binary"] = args.llama_binary
    return conf


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


# ---------------------------------------------------------- generation


def run_dialog(session, seed: dict, examiner: dict, examiner_url: str,
               student: dict, student_url: str, student_system: str | None,
               n_turns: int, timeout) -> list[dict]:
    """One eval dialog: forced opener, then alternating turns."""
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
            text, tokens, elapsed = ask(
                session, student_url, student,
                student_messages(turns, student_system), timeout)
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
                      max_tokens: int) -> dict:
    """A llama.cpp seat, recorded harness-style (provenance verbatim)."""
    participant = {
        "name": name,
        "model": os.path.basename(model_path) if model_path else name,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra": _NO_THINKING,
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
    out_path = args.out or default_dialogs_path(args.student or "student")
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
        # per-request field, so the expensive loads are paid once.
        if args.student:
            student_server = launch_chat_server(
                args.student, out_path, temperatures[0],
                args.max_reply_tokens)
            servers.append(student_server)
            student_url = student_server.base_url
        else:
            student_url = args.student_url
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
                    seat_url, seat_system = (
                        examiner_url, seat["system_prompt"])
                elif anchor == "garbage":
                    seat = texmo_participant(
                        "garbage-student", args.garbage_student, None,
                        temperature, args.max_reply_tokens)
                    seat_url, seat_system = garbage_url, None
                else:
                    seat = texmo_participant(
                        _STUDENT, args.student, args.student_url,
                        temperature, args.max_reply_tokens)
                    seat_url, seat_system = student_url, None
                started_at = _now()
                t0 = time.time()
                try:
                    turns = run_dialog(
                        session, seed, examiner, examiner_url, seat, seat_url,
                        seat_system, args.n_turns, timeout)
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


# ------------------------------------------------------------- judging


def student_positions(turns: list[dict]) -> list[tuple[int, int]]:
    """(position, turn index) for every student answer, position 1-based."""
    out = []
    for i, turn in enumerate(turns):
        if turn["side"] == _STUDENT:
            out.append((len(out) + 1, i))
    return out


def grade_answer(session, base_url: str, judge: dict, turns: list[dict],
                 upto: int, timeout) -> dict:
    """One judged answer; one retry, then `judge_error`."""
    error = None
    raw = ""
    for attempt in range(2):
        messages = judge_messages(turns, upto, error if attempt else None)
        raw, _, _ = ask(session, base_url, judge, messages, timeout)
        try:
            grade = parse_judge_output(raw)
        except ValueError as e:
            error = str(e)
            logging.warning(f"judge parse failure ({error}); "
                            f"{'retrying' if not attempt else 'giving up'}")
            continue
        grade["judge_error"] = None
        grade["raw"] = raw
        return grade
    return {
        "a": None, "b": None, "c": None,
        "examiner_defect": None, "comment": "",
        "judge_error": error, "raw": raw,
    }


def default_grades_path(dialogs: str) -> str:
    stem, _ = os.path.splitext(dialogs)
    return stem + "-grades.jsonl"


def default_report_path(dialogs: str) -> str:
    stem, _ = os.path.splitext(dialogs)
    return stem + "-report.md"


def run_judge(args) -> int:
    dialogs = read_records(args.dialogs)
    if not dialogs:
        raise ValueError(f"no dialogs in {args.dialogs}")
    out_path = args.out or default_grades_path(args.dialogs)
    report_path = args.report or default_report_path(args.dialogs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = existing_keys(out_path, grade_key)
    timeout = (dh._CONNECT_TIMEOUT, args.request_timeout)
    session = requests.Session()

    servers: list[dh.Server] = []
    try:
        if args.judge:
            judge_server = launch_llama_server(
                args.judge, out_path, _server_conf(args))
            servers.append(judge_server)
            judge_url = judge_server.base_url
        else:
            judge_url = args.judge_url
        for server in servers:
            dh.wait_until_ready(server, session, args.startup_timeout)
            logging.info(f"server on port {server.port} is ready")

        judge = llama_participant(
            "judge", args.judge, args.judge_url, "",
            args.judge_temperature, args.judge_max_tokens)
        # The system prompt is per-request (`judge_messages`); the seat
        # only carries the sampling params and the no-thinking rule.
        judge.pop("system_prompt")

        written = skipped = 0
        with open(out_path, "a", encoding="utf-8", newline="\n") as f:
            for record in dialogs:
                turns = record["turns"]
                for position, index in student_positions(turns):
                    if record_key(record) + (position,) in done:
                        skipped += 1
                        continue
                    try:
                        grade = grade_answer(
                            session, judge_url, judge, turns, index, timeout)
                    except KeyboardInterrupt:
                        print(f"\ninterrupted; {written} grade(s) in "
                              f"{out_path}")
                        raise
                    answer = turns[index]["text"]
                    grade.update({
                        "seed_idx": record["seed_idx"],
                        "anchor": record.get("anchor"),
                        "student_temperature":
                            record.get("student_temperature"),
                        "position": position,
                        "answer": answer,
                        "repetition": repetition_stats(answer),
                        "judge_inconsistency": is_inconsistent(grade),
                        "judge_model": judge["model"],
                    })
                    dh.write_record(f, grade)
                    written += 1
                    print(f"[{written}] "
                          f"{record.get('anchor') or 'main'} "
                          f"seed={record['seed_idx']} "
                          f"T={record.get('student_temperature')} "
                          f"pos={position} "
                          f"a={grade['a']} b={grade['b']} c={grade['c']}")
    except KeyboardInterrupt:
        pass
    finally:
        _stop_all(servers)

    grades = read_records(out_path)
    report = build_report(args.dialogs, out_path, grades)
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


def summarize(grades: list[dict]) -> dict:
    """Rates and counts for one group of grades."""
    valid = [g for g in grades if g.get("judge_error") is None]
    rates = {k: _rate(valid, k) for k in _CRITERIA}
    deflection = None
    if rates["b"] is not None and rates["c"] is not None:
        deflection = rates["b"] - rates["c"]
    return {
        "n": len(grades),
        "n_valid": len(valid),
        "rates": rates,
        "deflection": deflection,
        "errors": sum(1 for g in grades if g.get("judge_error") is not None),
        "inconsistent": sum(1 for g in grades
                            if g.get("judge_inconsistency")),
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


def build_report(dialogs_path: str, grades_path: str,
                 grades: list[dict]) -> str:
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
        "## Main set, by student temperature",
        "",
        f"{overall['n']} student answers "
        f"({overall['n_valid']} with a valid grade). Targets: "
        + ", ".join(f"{k} >= {_TARGETS[k]:.0f}%" for k in _CRITERIA)
        + "; `verdict` is PASS only when all three clear.",
        "",
    ] + list(_SUMMARY_HEADER)
    for temperature in _temperatures(main):
        group = [g for g in main
                 if g.get("student_temperature") == temperature]
        lines.append(_summary_row(temperature, summarize(group)))
    lines += [
        "",
        f"- judge errors: {overall['errors']}",
        f"- judge inconsistencies (c without b): {overall['inconsistent']}",
        "",
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
        "--out", default=None,
        help="grades JSONL (default: <dialogs stem>-grades.jsonl)")
    jud.add_argument(
        "--report", default=None,
        help="markdown report (default: <dialogs stem>-report.md)")
    _add_server_args(jud)
    jud.set_defaults(func=run_judge)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
