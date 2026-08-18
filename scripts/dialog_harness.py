"""Record synthetic dialogs between two full-size LLMs.

Milestone 1 of the tiny chatbot program (see `docs/roadmap.md`): two
generator models talk to each other, each with its own system prompt,
and every dialog is recorded verbatim for later processing into
training data for a nano-sized texmo model.

The generators run **outside texmo**, on their own runtime: a local
llama.cpp `llama-server`, vLLM, or anything else that speaks the
OpenAI chat-completions API. This script therefore imports no
inference library at all -- only `requests` and the standard library
-- and knows the models solely through
`POST {base_url}/v1/chat/completions` (non-streaming, sequential
turns, no async).

    uv run python scripts/dialog_harness.py scripts/dialog_sample_conf.json \\
        [--out data/dialogs/run1.jsonl] [--n-dialogs 20]

## Two modes, per participant

**Managed** (`model_path`): the harness launches `llama-server` on the
GGUF itself, on an auto-assigned free port, waits for it to load, runs
the dialogs, and always shuts it down again. One command, no server
babysitting. Both seats naming the *same* file share one server --
the usual case, one model talking to itself in two roles -- so the
weights are loaded once.

**External** (`base_url`): an endpoint somebody else runs, which the
harness only talks to and never starts, stops, or otherwise touches.
This is the path for a model served from another machine.

Exactly one of the two per participant; the modes can be mixed across
the two seats. `scripts/dialog_sample_conf.json` is the external
example (two seats on one already-running `llama-server`) and
`scripts/dialog_sample_local_conf.json` the managed one (both seats on
one GGUF, so one shared server); both are a curious child and a
patient teacher at 2000 bytes per dialog. Set the real model path or
URLs before running either.

Note the `extra` both samples carry:
`{"chat_template_kwargs": {"enable_thinking": false}}`. 2026 instruct
models (Qwen3, Gemma 4) reason by default and return their answer in a
separate field, so `content` comes back **empty** and the harness
stops the dialog with `empty_utterance`. Turning thinking off is what
makes them emit plain replies.

## Config schema

A single JSON object:

    participants     exactly two objects, each:
                       name           label for the record
                       base_url       external endpoint, e.g.
                                      "http://127.0.0.1:8080" (a
                                      trailing "/v1" is optional), or
                       model_path     path to a .gguf the harness
                                      serves itself -- exactly one of
                                      the two is required
                       model          model name sent in the request.
                                      llama-server ignores it, so for
                                      `model_path` any label will do;
                                      vLLM-style endpoints want the
                                      served name
                       system_prompt  this participant's seat prompt
                       temperature?   } sent only when present, so
                       top_p?         } the endpoint's own defaults
                       max_tokens?    } apply otherwise
                       extra?         opaque dict merged into the
                                      request body, last -- see below
    opener           {"text": "..."}  fixed first utterance, spoken by
                                      participant 0, or
                     {"from": 0|1}    that participant writes the first
                                      utterance from its system prompt
                                      alone (a system-only request --
                                      llama-server is fine with it,
                                      some endpoints want a user
                                      message, in which case use the
                                      `text` form)
    min_dialog_bytes stop rule: stop once the summed UTF-8 length of
                     all utterances reaches this
    max_turns?       safety cap on the number of utterances
                     (default 64)
    n_dialogs?       how many dialogs to record (default 1;
                     --n-dialogs overrides)
    request_timeout? read timeout in seconds (default 300; the connect
                     timeout is fixed at 10)
    server?          settings for harness-launched servers (ignored
                     when every participant is `base_url`):
                       binary?          llama-server executable
                       ngl?             GPU layers (default 99)
                       args?            list appended verbatim to the
                                        command line, e.g.
                                        ["-c", "8192", "-fa"]
                       startup_timeout? seconds to wait for the model
                                        to load (default 300)

The binary is resolved in this order: `server.binary`, the
`$LLAMA_SERVER` environment variable, `llama-server` on `PATH`, and
finally a hardcoded local install path as a last resort.

Each launched server gets its stdout and stderr appended to
`<out stem>.<model stem>.server.log` next to the output JSONL, so a
crashed or slow load leaves a trail. A server that exits before it is
ready aborts the run with the tail of that log in the error.

Every launched server is stopped on the way out -- normal finish,
readiness failure, exception, Ctrl-C alike -- and stopping kills the
whole process *tree*, so pointing `server.binary` at a wrapper script
does not strand the real server holding the GPU.

`extra` is the reserved hook for **forcing the models' hand** --
constraining generation to a small vocabulary. Anything the endpoint
accepts goes in verbatim: `logit_bias`, llama.cpp's `grammar` /
`json_schema`, `stop`, `seed`, `repeat_penalty`. It is merged into the
request body *after* the named sampling params, so it can also
override them. It is recorded verbatim in every dialog, so a tranche
always says which constraint produced it.

## Output

JSONL under `data/dialogs/` (untracked; created on demand), one dialog
per line, append mode. Each record carries full provenance -- the two
participant objects verbatim, system prompts and sampling params
included -- so tranche processing can slice by anything later:

    {config_hash, started_at, ended_at, stop_reason, total_bytes,
     turns: [{speaker, text}], participants: [...], opener}

`speaker` is the participant index. `stop_reason` is one of:

    bytes            total_bytes reached min_dialog_bytes (the
                     crossing utterance is kept). The normal ending.
    max_turns        the safety cap tripped first.
    empty_utterance  a model returned empty/whitespace-only text; that
                     utterance is not recorded.

Records are written one at a time with a single `write` + `flush`, so
Ctrl-C mid-dialog truncates the file at a record boundary rather than
corrupting it -- the in-flight dialog is simply lost.

Note on JSON style: the repo writes JSON files through `texmo.pjson`,
but JSONL is defined by one record per line, which is exactly what
that pretty-printer breaks. Records use plain `json.dumps` with
`sort_keys=True` (stable field order for diffs); `config_hash` hashes
the same canonical form.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

import requests

# Connect timeout for every request. The read timeout comes from the
# config (`request_timeout`), since a long dialog turn on a big
# quantized model can legitimately take minutes.
_CONNECT_TIMEOUT = 10
_DEFAULT_READ_TIMEOUT = 300

_DEFAULT_MAX_TURNS = 64
_DEFAULT_OUT_DIR = os.path.join("data", "dialogs")

# Self-managed llama.cpp servers. The binary is looked up in this
# order: `server.binary` in the config, $LLAMA_SERVER, `llama-server`
# on PATH, and finally the known local install below -- a
# last-resort convenience default, not a machine the harness assumes.
_FALLBACK_LLAMA_SERVER = "C:/Users/oleg/llama.cpp/llama-server.exe"
_SERVER_HOST = "127.0.0.1"
_DEFAULT_NGL = 99
# Model load off a cold page cache is minutes for a big quant.
_DEFAULT_STARTUP_TIMEOUT = 300
_HEALTH_POLL_INTERVAL = 1.0
_HEALTH_TIMEOUT = (_CONNECT_TIMEOUT, 10)
# Grace between SIGTERM-equivalent and kill (Windows: terminate() is
# TerminateProcess, so the grace mostly matters on POSIX).
_STOP_GRACE = 10.0
_LOG_TAIL_LINES = 20

# Retry the same shape as `texmo/client.py:retry` (exponential
# backoff), but bounded: a data-generation run should give up on a
# wedged endpoint rather than spin forever.
_RETRY_ATTEMPTS = 5
_RETRY_MIN_DELAY = 1.0
_RETRY_EXP = 1.5
_RETRY_MAX_DELAY = 60.0

_REQUIRED_PARTICIPANT_KEYS = ("name", "model", "system_prompt")
_SAMPLING_KEYS = ("temperature", "top_p", "max_tokens")


def load_config(path: str) -> dict:
    """Read and validate an experiment config."""
    with open(path, encoding="utf-8") as f:
        conf = json.load(f)
    validate_config(conf)
    return conf


def validate_config(conf: dict):
    """Raise ValueError on anything `run_dialog` cannot execute."""
    if not isinstance(conf, dict):
        raise ValueError("config must be a JSON object")

    participants = conf.get("participants")
    if not isinstance(participants, list) or len(participants) != 2:
        raise ValueError("config needs exactly two `participants`")
    for i, p in enumerate(participants):
        if not isinstance(p, dict):
            raise ValueError(f"participant {i} must be an object")
        missing = [k for k in _REQUIRED_PARTICIPANT_KEYS if k not in p]
        if missing:
            raise ValueError(
                f"participant {i} is missing {', '.join(missing)}")
        # Either the harness starts the server (`model_path`) or it
        # talks to one somebody else runs (`base_url`) -- never both,
        # since the two answer the same question differently.
        if ("base_url" in p) == ("model_path" in p):
            raise ValueError(
                f"participant {i} needs exactly one of `base_url` or "
                f"`model_path`")
        if "extra" in p and not isinstance(p["extra"], dict):
            raise ValueError(f"participant {i}: `extra` must be an object")

    opener = conf.get("opener")
    if not isinstance(opener, dict):
        raise ValueError("config needs an `opener` object")
    has_text, has_from = "text" in opener, "from" in opener
    if has_text == has_from:
        raise ValueError(
            "`opener` needs exactly one of `text` or `from`")
    if has_from and opener["from"] not in (0, 1):
        raise ValueError("`opener.from` must be 0 or 1")
    if has_text and not str(opener["text"]).strip():
        raise ValueError("`opener.text` must not be empty")

    min_bytes = conf.get("min_dialog_bytes")
    if not isinstance(min_bytes, int) or min_bytes <= 0:
        raise ValueError("`min_dialog_bytes` must be a positive integer")

    max_turns = conf.get("max_turns", _DEFAULT_MAX_TURNS)
    if not isinstance(max_turns, int) or max_turns <= 0:
        raise ValueError("`max_turns` must be a positive integer")

    server_conf = conf.get("server", {})
    if not isinstance(server_conf, dict):
        raise ValueError("`server` must be an object")
    if not isinstance(server_conf.get("args", []), list):
        raise ValueError("`server.args` must be a list")


def canonical_json(conf: dict) -> str:
    """The canonical form both `config_hash` and records are built on."""
    return json.dumps(
        conf, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(conf: dict) -> str:
    """Short sha256 over the canonicalized config.

    Stable across key order and formatting, so re-running the same
    experiment appends records that group by hash.
    """
    digest = hashlib.sha256(canonical_json(conf).encode("utf-8"))
    return digest.hexdigest()[:12]


def seat_messages(participant: dict, turns: list[dict], seat: int
                  ) -> list[dict]:
    """The dialog as `seat` sees it, in chat-completions form.

    Every participant sees its own system prompt first, its own past
    utterances as `assistant`, and the other's as `user`. Since turns
    strictly alternate, roles alternate too.
    """
    messages = [{"role": "system", "content": participant["system_prompt"]}]
    for turn in turns:
        role = "assistant" if turn["speaker"] == seat else "user"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def build_request(participant: dict, messages: list[dict]) -> dict:
    """The chat-completions request body for one utterance."""
    body = {
        "model": participant["model"],
        "messages": messages,
        "stream": False,
    }
    for key in _SAMPLING_KEYS:
        if key in participant:
            body[key] = participant[key]
    # Merged last: `extra` is allowed to override the named params,
    # and is where vocabulary constraints (logit_bias, grammar) live.
    body.update(participant.get("extra", {}))
    return body


def completions_url(base_url: str) -> str:
    """`base_url` -> the chat-completions endpoint.

    Tolerates a `base_url` that already ends in `/v1`, since that is
    how OpenAI-compatible endpoints are usually quoted.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def post_completion(session, url: str, body: dict, timeout) -> dict:
    """POST with bounded exponential backoff; returns the parsed JSON."""
    delay = _RETRY_MIN_DELAY
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            response = session.post(url, json=body, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == _RETRY_ATTEMPTS - 1:
                raise
            logging.warning(f"Request to {url} failed: {e}")
            logging.warning(f"Retrying in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * _RETRY_EXP, _RETRY_MAX_DELAY)


def extract_text(response: dict) -> str:
    """The utterance out of a chat-completions response.

    Surrounding whitespace is stripped: models routinely emit a
    leading newline, and these strings become training text.
    """
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    return (content or "").strip()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def run_dialog(conf: dict, session, chash: str,
               base_urls: list[str] | None = None) -> dict:
    """Run one full dialog and return its record.

    Turns alternate strictly. After every utterance -- the fixed
    opener included -- the byte total decides whether to stop, so the
    utterance that crosses `min_dialog_bytes` is kept. `bytes` wins
    over `max_turns` when both would trip on the same utterance.

    `base_urls` gives the endpoint per seat, which is how a
    harness-launched server (whose port is only known at runtime)
    reaches its participant; without it each participant's own
    `base_url` is used.
    """
    participants = conf["participants"]
    if base_urls is None:
        base_urls = [p["base_url"] for p in participants]
    min_bytes = conf["min_dialog_bytes"]
    max_turns = conf.get("max_turns", _DEFAULT_MAX_TURNS)
    timeout = (_CONNECT_TIMEOUT,
               conf.get("request_timeout", _DEFAULT_READ_TIMEOUT))
    opener = conf["opener"]

    started_at = _now()
    turns: list[dict] = []
    total_bytes = 0
    stop_reason = None

    if "text" in opener:
        text = str(opener["text"]).strip()
        turns.append({"speaker": 0, "text": text})
        total_bytes += len(text.encode("utf-8"))
        speaker = 1
        if total_bytes >= min_bytes:
            stop_reason = "bytes"
        elif len(turns) >= max_turns:
            stop_reason = "max_turns"
    else:
        speaker = opener["from"]

    while stop_reason is None:
        participant = participants[speaker]
        messages = seat_messages(participant, turns, speaker)
        response = post_completion(
            session, completions_url(base_urls[speaker]),
            build_request(participant, messages), timeout)
        text = extract_text(response)
        if not text:
            # Not recorded: an empty assistant turn is noise in the
            # training data, and the reason is kept in the record.
            stop_reason = "empty_utterance"
            break
        turns.append({"speaker": speaker, "text": text})
        total_bytes += len(text.encode("utf-8"))
        if total_bytes >= min_bytes:
            stop_reason = "bytes"
        elif len(turns) >= max_turns:
            stop_reason = "max_turns"
        speaker = 1 - speaker

    return {
        "config_hash": chash,
        "started_at": started_at,
        "ended_at": _now(),
        "stop_reason": stop_reason,
        "total_bytes": total_bytes,
        "turns": turns,
        # Verbatim, system prompts and sampling params included.
        "participants": participants,
        "opener": opener,
    }


def write_record(f, record: dict):
    """Append one record as a single line, flushed immediately."""
    f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    f.flush()


def default_out_path(conf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(conf_path))[0]
    return os.path.join(_DEFAULT_OUT_DIR, stem + ".jsonl")


def model_key(model_path: str) -> str:
    """Dedupe key for a model file.

    Absolute and case-normalized, so `models/m.gguf` and
    `./MODELS/M.GGUF` name one server on Windows.
    """
    return os.path.normcase(os.path.abspath(model_path))


def free_port() -> int:
    """Ask the OS for an unused port and release it immediately.

    The usual bind-to-0 trick. There is a small race between the close
    here and llama-server binding, which is acceptable for a handful
    of servers started once per run.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_SERVER_HOST, 0))
        return sock.getsockname()[1]


def resolve_binary(server_conf: dict) -> str:
    """Locate the `llama-server` binary.

    In order: `server.binary` from the config, `$LLAMA_SERVER`,
    `llama-server` on PATH, then `_FALLBACK_LLAMA_SERVER`.
    """
    if server_conf.get("binary"):
        return server_conf["binary"]
    if os.environ.get("LLAMA_SERVER"):
        return os.environ["LLAMA_SERVER"]
    return shutil.which("llama-server") or _FALLBACK_LLAMA_SERVER


def server_command(binary: str, model_path: str, port: int,
                   server_conf: dict) -> list[str]:
    """The llama-server command line. `server.args` is appended raw."""
    cmd = [
        binary,
        "-m", model_path,
        "--host", _SERVER_HOST,
        "--port", str(port),
        "-ngl", str(server_conf.get("ngl", _DEFAULT_NGL)),
    ]
    return cmd + [str(a) for a in server_conf.get("args", [])]


def server_log_path(out_path: str, model_path: str) -> str:
    """`<out stem>.<model stem>.server.log`, next to the output file."""
    out_stem = os.path.splitext(os.path.basename(out_path))[0]
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(
        os.path.dirname(out_path) or ".",
        f"{out_stem}.{model_stem}.server.log")


def spawn_server(cmd: list[str], log_file) -> subprocess.Popen:
    """Start one llama-server with both streams into `log_file`.

    On POSIX each server gets its own session, so `kill_tree` can take
    down the whole group at the end.
    """
    kwargs = {} if os.name == "nt" else {"start_new_session": True}
    return subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)


def kill_tree(proc, grace: float = _STOP_GRACE):
    """Stop a server *and* anything it wrapped.

    `Popen.terminate()` reaches only the direct child. When
    `server.binary` is a shim -- a .cmd/.bat wrapper, a launcher
    script, `conda run` -- the real llama-server is a grandchild that
    survives its parent and keeps the GPU. So kill the tree while the
    parent is still alive to name it: `taskkill /T` on Windows, the
    process group on POSIX. Measured, not assumed: a .cmd shim left
    the port bound after the run until this was added.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            logging.warning(f"llama-server pid {proc.pid} would not die")


class Server:
    """One llama-server process the harness started (and must stop)."""

    def __init__(self, model_path: str, port: int, proc, log_path: str,
                 log_file):
        self.model_path = model_path
        self.port = port
        self.proc = proc
        self.log_path = log_path
        self.log_file = log_file

    @property
    def base_url(self) -> str:
        return f"http://{_SERVER_HOST}:{self.port}"

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def log_tail(self, lines: int = _LOG_TAIL_LINES) -> str:
        """The last log lines, for the abort message."""
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return "".join(f.readlines()[-lines:])
        except OSError:
            return ""

    def stop(self, grace: float = _STOP_GRACE):
        """Take the server down for good. Idempotent."""
        if self.proc.poll() is None:
            kill_tree(self.proc, grace)
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()


def stop_servers(servers: dict):
    """Stop every launched server; one failure must not skip the rest."""
    for server in servers.values():
        try:
            server.stop()
        except Exception as e:
            logging.warning(
                f"Failed to stop llama-server for {server.model_path}: {e}")


def launch_servers(conf: dict, out_path: str) -> dict:
    """Start one llama-server per *unique* model file.

    Participants with a `base_url` are somebody else's servers and are
    left alone. Two participants naming the same file share a single
    server -- the common case, two seats of one model. If any launch
    fails, whatever already started is stopped before the error
    propagates.
    """
    server_conf = conf.get("server", {})
    binary = resolve_binary(server_conf)
    servers: dict[str, Server] = {}
    try:
        for p in conf["participants"]:
            if "model_path" not in p:
                continue
            key = model_key(p["model_path"])
            if key in servers:
                continue
            if not os.path.exists(p["model_path"]):
                raise FileNotFoundError(
                    f"model file not found: {p['model_path']}")
            port = free_port()
            log_path = server_log_path(out_path, p["model_path"])
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8",
                            errors="replace")
            cmd = server_command(binary, p["model_path"], port, server_conf)
            log_file.write(f"\n=== {_now()} {' '.join(cmd)} ===\n")
            log_file.flush()
            servers[key] = Server(
                p["model_path"], port, spawn_server(cmd, log_file),
                log_path, log_file)
            logging.info(
                f"llama-server for {os.path.basename(p['model_path'])} "
                f"on port {port} (log {log_path})")
    except BaseException:
        stop_servers(servers)
        raise
    return servers


def wait_until_ready(server: Server, session,
                     timeout: float = _DEFAULT_STARTUP_TIMEOUT,
                     poll: float = _HEALTH_POLL_INTERVAL):
    """Block until the server answers /health, or give up.

    llama-server answers 503 while the weights load and 200 once it can
    serve. A process that exits first aborts the run immediately rather
    than waiting out the timeout, with its last log lines attached.
    """
    deadline = time.time() + timeout
    url = server.base_url + "/health"
    while True:
        if not server.is_running():
            raise RuntimeError(
                f"llama-server for {server.model_path} exited with code "
                f"{server.proc.returncode} before it was ready.\n"
                f"Last lines of {server.log_path}:\n{server.log_tail()}")
        try:
            if session.get(url, timeout=_HEALTH_TIMEOUT).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass  # not listening yet
        if time.time() >= deadline:
            raise TimeoutError(
                f"llama-server for {server.model_path} was not ready after "
                f"{timeout:.0f}s (log {server.log_path})")
        time.sleep(poll)


def seat_base_urls(conf: dict, servers: dict) -> list[str]:
    """The endpoint each seat talks to, launched or external."""
    urls = []
    for p in conf["participants"]:
        if "model_path" in p:
            urls.append(servers[model_key(p["model_path"])].base_url)
        else:
            urls.append(p["base_url"])
    return urls


def run(conf: dict, out_path: str, n_dialogs: int, session=None,
        base_urls: list[str] | None = None) -> int:
    """Record `n_dialogs` dialogs into `out_path`; returns how many."""
    if session is None:
        session = requests.Session()
    chash = config_hash(conf)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = 0
    # newline="\n": a data file read by other tools should not pick up
    # CRLF on Windows.
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for i in range(n_dialogs):
            start = time.time()
            try:
                record = run_dialog(conf, session, chash, base_urls)
            except KeyboardInterrupt:
                print(f"\ninterrupted during dialog {i + 1}; "
                      f"{written} complete dialog(s) in {out_path}")
                break
            write_record(f, record)
            written += 1
            print(f"[{i + 1}/{n_dialogs}] turns={len(record['turns'])} "
                  f"bytes={record['total_bytes']} "
                  f"stop={record['stop_reason']} "
                  f"{time.time() - start:.1f}s")
    return written


def run_experiment(conf: dict, out_path: str, n_dialogs: int,
                   session=None) -> int:
    """Config to dialogs: start the servers, run, always tear down.

    The `finally` is the whole point -- normal exit, a readiness
    failure, an exception mid-dialog and Ctrl-C all take the same path
    out, and no llama-server is left holding the GPU. Servers named by
    `base_url` were never ours and are not touched.
    """
    if session is None:
        session = requests.Session()
    server_conf = conf.get("server", {})
    startup_timeout = server_conf.get(
        "startup_timeout", _DEFAULT_STARTUP_TIMEOUT)
    servers = {}
    try:
        servers = launch_servers(conf, out_path)
        for server in servers.values():
            wait_until_ready(server, session, startup_timeout)
            logging.info(f"llama-server on port {server.port} is ready")
        return run(conf, out_path, n_dialogs, session=session,
                   base_urls=seat_base_urls(conf, servers))
    finally:
        stop_servers(servers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record synthetic dialogs between two LLM endpoints.")
    parser.add_argument("config", help="experiment config JSON")
    parser.add_argument(
        "--out", default=None,
        help="output JSONL (default data/dialogs/<config stem>.jsonl)")
    parser.add_argument(
        "--n-dialogs", type=int, default=None,
        help="how many dialogs to record (default from config, or 1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conf = load_config(args.config)
    out_path = args.out or default_out_path(args.config)
    n_dialogs = (args.n_dialogs if args.n_dialogs is not None
                 else conf.get("n_dialogs", 1))

    print(f"config {args.config} (hash {config_hash(conf)}) -> {out_path}")
    written = run_experiment(conf, out_path, n_dialogs)
    print(f"wrote {written} dialog(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
