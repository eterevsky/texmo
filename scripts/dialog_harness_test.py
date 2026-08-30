"""Tests for `scripts/dialog_harness.py`.

The harness lives in `scripts/` (it is deliberately not texmo-specific
-- it talks to external OpenAI-compatible endpoints and imports
nothing from texmo), but its test lives here: `pyproject.toml` sets
`testpaths = ["texmo"]`, so a test under `scripts/` would never run in
the repo's default `uv run pytest`. The import shim below is the
mirror image of the one `scripts/*.py` use to reach texmo.

No live endpoint is needed: every test drives the harness with a fake
`requests` session that records the request bodies and returns canned
chat-completion payloads.
"""
import datetime
import json
import os
import socket
import subprocess
import sys

import pytest
import requests

# Appended, not prepended: scripts/ holds modules whose names collide
# with real packages (e.g. coverage.py), and this path stays on
# sys.path for the rest of the pytest session.
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"))
import dialog_harness
import dialog_text


class _FakeResponse:

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for `requests.Session`, recording every call.

    `replies` are utterance texts handed out in order; once exhausted
    the last one repeats, so a max_turns test needs only one entry.
    `errors` optionally maps a call index to an exception to raise.
    """

    def __init__(self, replies, errors=None, healthy_after=0):
        self.replies = list(replies)
        self.errors = errors or {}
        self.healthy_after = healthy_after
        self.calls = []
        self.health_calls = []

    def post(self, url, json=None, timeout=None):
        index = len(self.calls)
        self.calls.append({"url": url, "body": json, "timeout": timeout})
        if index in self.errors:
            raise self.errors[index]
        text = self.replies[min(index, len(self.replies) - 1)]
        return _FakeResponse(
            {"choices": [{"message": {"role": "assistant",
                                      "content": text}}]})

    def get(self, url, timeout=None):
        """Readiness probe: 503 while 'loading', then 200."""
        self.health_calls.append(url)
        response = _FakeResponse({})
        if len(self.health_calls) <= self.healthy_after:
            response.status_code = 503
        return response

    def bodies(self):
        return [c["body"] for c in self.calls]

    def ports(self):
        return [c["url"].split(":")[2].split("/")[0] for c in self.calls]


class _FakePopen:
    """A llama-server that never actually runs.

    `exit_code=None` means "still alive"; set it to simulate a server
    that died during load.
    """

    pid = 424242  # never dereferenced: kill_tree is stubbed in tests

    def __init__(self, cmd, log_file, exit_code=None):
        self.cmd = cmd
        self.log_file = log_file
        self.returncode = exit_code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _Spawner:
    """Stands in for `spawn_server`, recording every launch."""

    def __init__(self, exit_code=None):
        self.spawned = []
        self.exit_code = exit_code

    def __call__(self, cmd, log_file):
        proc = _FakePopen(cmd, log_file, self.exit_code)
        self.spawned.append(proc)
        return proc


def _fake_kill_tree(proc, grace=0):
    """Stand-in for `kill_tree`: no real taskkill against a fake pid."""
    proc.terminate()


def _install_fake_servers(monkeypatch, exit_code=None, first_port=9001):
    """Patch the seams: spawning, port assignment, and killing."""
    spawner = _Spawner(exit_code)
    ports = iter(range(first_port, first_port + 10))
    monkeypatch.setattr(dialog_harness, "spawn_server", spawner)
    monkeypatch.setattr(dialog_harness, "free_port", lambda: next(ports))
    monkeypatch.setattr(dialog_harness, "kill_tree", _fake_kill_tree)
    monkeypatch.setattr(dialog_harness.time, "sleep", lambda _s: None)
    return spawner


def _gguf(tmp_path, name):
    """A stand-in model file, so the existence check passes."""
    path = tmp_path / name
    path.write_bytes(b"GGUF")
    return str(path)


def _local_conf(model_paths, **overrides):
    """`_make_conf`, but managed-mode participants."""
    conf = _make_conf(**overrides)
    for participant, model_path in zip(conf["participants"], model_paths):
        del participant["base_url"]
        participant["model_path"] = model_path
    return conf


def _make_conf(**overrides):
    conf = {
        "participants": [
            {
                "name": "child",
                "base_url": "http://127.0.0.1:8080",
                "model": "model-a",
                "system_prompt": "You are a curious child.",
                "temperature": 1.0,
            },
            {
                "name": "teacher",
                "base_url": "http://127.0.0.1:8081",
                "model": "model-b",
                "system_prompt": "You are a patient teacher.",
                "max_tokens": 160,
                "extra": {"logit_bias": {"5": -100}},
            },
        ],
        "opener": {"text": "Hi!"},
        "min_dialog_bytes": 20,
        "max_turns": 8,
    }
    conf.update(overrides)
    return conf


def _run(conf, replies, errors=None):
    session = _FakeSession(replies, errors)
    record = dialog_harness.run_dialog(
        conf, session, dialog_harness.config_hash(conf))
    return record, session


# --- seat mapping ---------------------------------------------------

def test_seat_messages_maps_own_to_assistant():
    conf = _make_conf()
    turns = [{"speaker": 0, "text": "a"}, {"speaker": 1, "text": "b"}]

    seen_by_0 = dialog_harness.seat_messages(
        conf["participants"][0], turns, 0)
    assert seen_by_0 == [
        {"role": "system", "content": "You are a curious child."},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    seen_by_1 = dialog_harness.seat_messages(
        conf["participants"][1], turns, 1)
    assert seen_by_1 == [
        {"role": "system", "content": "You are a patient teacher."},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_each_seat_sees_the_dialog_from_its_own_side():
    # Long enough to need three generated utterances before the stop.
    conf = _make_conf(min_dialog_bytes=30)
    record, session = _run(conf, ["reply-one", "reply-two", "reply-three"])
    bodies = session.bodies()
    assert len(bodies) >= 3

    # Turn 1: the teacher (seat 1) answers the fixed opener.
    assert bodies[0]["model"] == "model-b"
    assert bodies[0]["messages"] == [
        {"role": "system", "content": "You are a patient teacher."},
        {"role": "user", "content": "Hi!"},
    ]

    # Turn 2: the child (seat 0) sees its own opener as `assistant`.
    assert bodies[1]["model"] == "model-a"
    assert bodies[1]["messages"] == [
        {"role": "system", "content": "You are a curious child."},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "reply-one"},
    ]

    # Turn 3: back to the teacher, whose own past turn is `assistant`.
    assert bodies[2]["messages"] == [
        {"role": "system", "content": "You are a patient teacher."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "reply-one"},
        {"role": "user", "content": "reply-two"},
    ]

    # Every request: system prompt first and only there, then strictly
    # alternating roles ending with the other participant as `user`.
    for body in bodies:
        roles = [m["role"] for m in body["messages"]]
        assert roles[0] == "system"
        tail = roles[1:]
        assert all(r != "system" for r in tail)
        assert tail[-1] == "user"
        assert all(a != b for a, b in zip(tail, tail[1:]))
    assert record["turns"][0] == {"speaker": 0, "text": "Hi!"}


def test_requests_go_to_each_participants_own_endpoint():
    conf = _make_conf(min_dialog_bytes=30)
    _, session = _run(conf, ["reply-one", "reply-two", "reply-three"])
    assert session.calls[0]["url"] == (
        "http://127.0.0.1:8081/v1/chat/completions")
    assert session.calls[1]["url"] == (
        "http://127.0.0.1:8080/v1/chat/completions")


# --- stop rules -----------------------------------------------------

def test_bytes_stop_keeps_the_crossing_utterance():
    # "Hi!" is 3 bytes; each reply is 10 -> crosses 20 on the second.
    conf = _make_conf(min_dialog_bytes=20)
    record, session = _run(conf, ["0123456789"])

    assert record["stop_reason"] == "bytes"
    assert record["total_bytes"] == 23
    assert record["total_bytes"] >= conf["min_dialog_bytes"]
    # The utterance that crossed the threshold is kept ...
    assert record["turns"][-1] == {"speaker": 0, "text": "0123456789"}
    assert len(record["turns"]) == 3
    # ... and nothing was generated after it.
    assert len(session.calls) == 2


def test_total_bytes_counts_utf8_not_characters():
    # Four 3-byte characters per reply.
    conf = _make_conf(opener={"text": "hi"}, min_dialog_bytes=12)
    record, _ = _run(conf, ["日本語だ"])
    assert record["stop_reason"] == "bytes"
    assert record["total_bytes"] == 2 + 12


def test_max_turns_stop():
    conf = _make_conf(min_dialog_bytes=10 ** 6, max_turns=5)
    record, session = _run(conf, ["short"])
    assert record["stop_reason"] == "max_turns"
    assert len(record["turns"]) == 5
    # The fixed opener is one of the five turns, so four were generated.
    assert len(session.calls) == 4


def test_bytes_wins_when_both_rules_trip_together():
    # The 3rd turn both crosses 20 bytes and hits max_turns=3.
    conf = _make_conf(min_dialog_bytes=20, max_turns=3)
    record, _ = _run(conf, ["0123456789"])
    assert len(record["turns"]) == 3
    assert record["stop_reason"] == "bytes"


def test_empty_utterance_stop():
    conf = _make_conf(min_dialog_bytes=10 ** 6)
    record, session = _run(conf, ["a real reply", "   \n  "])
    assert record["stop_reason"] == "empty_utterance"
    # The blank utterance is not recorded; the good ones are.
    assert record["turns"] == [
        {"speaker": 0, "text": "Hi!"},
        {"speaker": 1, "text": "a real reply"},
    ]
    assert record["total_bytes"] == 3 + len("a real reply")
    assert len(session.calls) == 2


def test_fixed_opener_alone_can_satisfy_the_byte_rule():
    conf = _make_conf(opener={"text": "x" * 50}, min_dialog_bytes=20)
    record, session = _run(conf, ["never used"])
    assert record["stop_reason"] == "bytes"
    assert len(record["turns"]) == 1
    assert session.calls == []


# --- opener forms ---------------------------------------------------

def test_opener_text_is_attributed_to_participant_zero():
    conf = _make_conf(min_dialog_bytes=10 ** 6, max_turns=2)
    record, _ = _run(conf, ["answer"])
    assert record["turns"][0] == {"speaker": 0, "text": "Hi!"}
    assert record["turns"][1]["speaker"] == 1
    assert record["opener"] == {"text": "Hi!"}


def test_opener_from_lets_that_participant_speak_first():
    conf = _make_conf(opener={"from": 1}, min_dialog_bytes=10 ** 6,
                      max_turns=2)
    record, session = _run(conf, ["teacher opens", "child answers"])

    # First request is participant 1, with the system prompt alone.
    assert session.bodies()[0]["model"] == "model-b"
    assert session.bodies()[0]["messages"] == [
        {"role": "system", "content": "You are a patient teacher."},
    ]
    assert record["turns"] == [
        {"speaker": 1, "text": "teacher opens"},
        {"speaker": 0, "text": "child answers"},
    ]


def test_opener_from_zero_starts_with_participant_zero():
    conf = _make_conf(opener={"from": 0}, min_dialog_bytes=10 ** 6,
                      max_turns=1)
    record, session = _run(conf, ["child opens"])
    assert session.bodies()[0]["model"] == "model-a"
    assert len(session.bodies()[0]["messages"]) == 1
    assert record["turns"] == [{"speaker": 0, "text": "child opens"}]


# --- request body ---------------------------------------------------

def test_sampling_params_are_sent_only_when_present():
    conf = _make_conf()
    child = dialog_harness.build_request(conf["participants"][0], [])
    assert child["temperature"] == 1.0
    assert "max_tokens" not in child
    assert "top_p" not in child
    assert child["stream"] is False


def test_extra_is_merged_last_and_can_override():
    participant = dict(_make_conf()["participants"][1])
    participant["temperature"] = 0.5
    participant["extra"] = {"temperature": 0.1, "grammar": "root ::= x"}
    body = dialog_harness.build_request(participant, [])
    assert body["temperature"] == 0.1
    assert body["grammar"] == "root ::= x"
    # Only what `extra` holds now -- it is copied in, not accumulated.
    assert "logit_bias" not in body


def test_completions_url_forms():
    assert dialog_harness.completions_url("http://h:8080") == (
        "http://h:8080/v1/chat/completions")
    assert dialog_harness.completions_url("http://h:8080/") == (
        "http://h:8080/v1/chat/completions")
    assert dialog_harness.completions_url("http://h:8080/v1") == (
        "http://h:8080/v1/chat/completions")
    assert dialog_harness.completions_url("http://h:8080/v1/") == (
        "http://h:8080/v1/chat/completions")


def test_extract_text_handles_missing_and_null_content():
    assert dialog_harness.extract_text({}) == ""
    assert dialog_harness.extract_text({"choices": []}) == ""
    assert dialog_harness.extract_text(
        {"choices": [{"message": {"content": None}}]}) == ""
    assert dialog_harness.extract_text(
        {"choices": [{"message": {"content": "  hi \n"}}]}) == "hi"


def test_retry_then_success(monkeypatch):
    slept = []
    monkeypatch.setattr(dialog_harness.time, "sleep", slept.append)
    conf = _make_conf(min_dialog_bytes=10 ** 6, max_turns=2)
    errors = {0: requests.exceptions.ConnectionError("refused")}
    record, session = _run(conf, ["recovered"], errors=errors)
    assert slept  # backed off once
    assert len(session.calls) == 2  # failed call, then the retry
    assert record["turns"][1]["text"] == "recovered"


# --- record shape and output ----------------------------------------

def test_record_shape_and_verbatim_participants():
    conf = _make_conf()
    record, _ = _run(conf, ["0123456789"])

    assert set(record) == {
        "config_hash", "started_at", "ended_at", "stop_reason",
        "total_bytes", "turns", "participants", "opener"}
    assert record["config_hash"] == dialog_harness.config_hash(conf)
    # Verbatim: system prompts, sampling params and the `extra` hook.
    assert record["participants"] == conf["participants"]
    assert record["participants"][1]["extra"] == {"logit_bias": {"5": -100}}
    assert record["participants"][0]["system_prompt"] == (
        "You are a curious child.")
    assert record["opener"] == conf["opener"]
    for stamp in (record["started_at"], record["ended_at"]):
        assert datetime.datetime.fromisoformat(stamp).tzinfo is not None
    for turn in record["turns"]:
        assert set(turn) == {"speaker", "text"}
        assert turn["speaker"] in (0, 1)


def test_config_hash_is_stable_and_content_sensitive():
    conf = _make_conf()
    reordered = {k: conf[k] for k in reversed(list(conf))}
    assert dialog_harness.config_hash(reordered) == (
        dialog_harness.config_hash(conf))

    changed = _make_conf()
    changed["participants"][0]["system_prompt"] = "different prompt"
    assert dialog_harness.config_hash(changed) != (
        dialog_harness.config_hash(conf))


def test_run_appends_one_line_per_dialog(tmp_path):
    conf = _make_conf()
    out = str(tmp_path / "dialogs" / "run.jsonl")

    assert dialog_harness.run(
        conf, out, 3, session=_FakeSession(["0123456789"])) == 3
    with open(out, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 3

    # Appends rather than truncating on a second run.
    assert dialog_harness.run(
        conf, out, 2, session=_FakeSession(["0123456789"])) == 2
    with open(out, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 5

    for line in lines:
        record = json.loads(line)
        assert record["config_hash"] == dialog_harness.config_hash(conf)
        assert record["stop_reason"] == "bytes"
        assert record["turns"][0]["text"] == "Hi!"


def test_records_survive_a_keyboard_interrupt_mid_dialog(tmp_path):
    conf = _make_conf()
    out = str(tmp_path / "run.jsonl")
    session = _FakeSession(["0123456789"], errors={2: KeyboardInterrupt()})

    # Dialog 1 completes (one call), dialog 2 is interrupted.
    assert dialog_harness.run(conf, out, 4, session=session) == 1
    with open(out, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 1
    json.loads(lines[0])  # a complete, parseable record


def test_default_out_path_follows_the_config_name():
    assert dialog_harness.default_out_path(
        os.path.join("scripts", "dialog_sample_conf.json")) == (
        os.path.join("data", "dialogs", "dialog_sample_conf.jsonl"))


# --- config validation ----------------------------------------------

def test_sample_config_is_valid():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "dialog_sample_conf.json")
    conf = dialog_harness.load_config(path)
    assert len(conf["participants"]) == 2
    assert conf["min_dialog_bytes"] == 2000


def test_validate_config_rejects_bad_configs():
    with pytest.raises(ValueError):
        dialog_harness.validate_config(_make_conf(participants=[]))
    with pytest.raises(ValueError):
        dialog_harness.validate_config(
            _make_conf(opener={"text": "hi", "from": 0}))
    with pytest.raises(ValueError):
        dialog_harness.validate_config(_make_conf(opener={}))
    with pytest.raises(ValueError):
        dialog_harness.validate_config(_make_conf(opener={"from": 2}))
    with pytest.raises(ValueError):
        dialog_harness.validate_config(_make_conf(min_dialog_bytes=0))
    with pytest.raises(ValueError):
        dialog_harness.validate_config(_make_conf(max_turns=-1))

    missing_key = _make_conf()
    del missing_key["participants"][0]["model"]
    with pytest.raises(ValueError):
        dialog_harness.validate_config(missing_key)


# --- managed servers: validation ------------------------------------

def test_participant_needs_exactly_one_endpoint_source():
    both = _make_conf()
    both["participants"][0]["model_path"] = "models/m.gguf"
    with pytest.raises(ValueError, match="exactly one"):
        dialog_harness.validate_config(both)

    neither = _make_conf()
    del neither["participants"][1]["base_url"]
    with pytest.raises(ValueError, match="exactly one"):
        dialog_harness.validate_config(neither)

    # model_path alone is valid, and so is mixing the two modes.
    dialog_harness.validate_config(
        _local_conf(["models/a.gguf", "models/b.gguf"]))
    mixed = _local_conf(["models/a.gguf", "models/b.gguf"])
    del mixed["participants"][1]["model_path"]
    mixed["participants"][1]["base_url"] = "http://127.0.0.1:9999"
    dialog_harness.validate_config(mixed)


def test_server_block_is_validated():
    with pytest.raises(ValueError, match="`server`"):
        dialog_harness.validate_config(_make_conf(server="llama-server"))
    with pytest.raises(ValueError, match="server.args"):
        dialog_harness.validate_config(_make_conf(server={"args": "-fa"}))
    dialog_harness.validate_config(
        _make_conf(server={"ngl": 33, "args": ["-c", "8192"]}))


def test_local_sample_config_is_valid():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "dialog_sample_local_conf.json")
    conf = dialog_harness.load_config(path)
    assert len(conf["participants"]) == 2
    # Both seats name one file: the shared-server case by construction.
    assert (conf["participants"][0]["model_path"]
            == conf["participants"][1]["model_path"])
    # No user-specific absolute path in a committed sample.
    assert not os.path.isabs(conf["participants"][0]["model_path"])
    for participant in conf["participants"]:
        assert participant["extra"] == {
            "chat_template_kwargs": {"enable_thinking": False}}


# --- managed servers: launching -------------------------------------

def test_same_model_file_in_both_seats_shares_one_server(
        monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "shared.gguf")
    session = _FakeSession(["0123456789"])

    assert dialog_harness.run_experiment(
        _local_conf([model, model]), str(tmp_path / "out.jsonl"), 1,
        session) == 1

    # One file -> one process -> both seats on the same port.
    assert len(spawner.spawned) == 1
    assert session.ports() == ["9001", "9001"]


def test_two_model_files_launch_two_servers(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    conf = _local_conf(
        [_gguf(tmp_path, "a.gguf"), _gguf(tmp_path, "b.gguf")])
    session = _FakeSession(["0123456789"])

    dialog_harness.run_experiment(
        conf, str(tmp_path / "out.jsonl"), 1, session)

    assert len(spawner.spawned) == 2
    assert "a.gguf" in " ".join(spawner.spawned[0].cmd)
    assert "b.gguf" in " ".join(spawner.spawned[1].cmd)
    # Seat 0 got the first port, seat 1 the second. The fixed opener is
    # seat 0's, so seat 1 (port 9002) is asked first.
    assert session.ports() == ["9002", "9001"]


def test_mixed_modes_launch_only_the_managed_seat(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    conf = _local_conf([_gguf(tmp_path, "a.gguf"), "replaced-below"])
    del conf["participants"][1]["model_path"]
    conf["participants"][1]["base_url"] = "http://127.0.0.1:9999"
    session = _FakeSession(["0123456789"])

    dialog_harness.run_experiment(
        conf, str(tmp_path / "out.jsonl"), 1, session)

    assert len(spawner.spawned) == 1
    assert session.ports() == ["9999", "9001"]


def test_base_url_participants_launch_nothing(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    session = _FakeSession(["0123456789"])

    assert dialog_harness.run_experiment(
        _make_conf(), str(tmp_path / "out.jsonl"), 1, session) == 1

    # Nothing started and nothing probed: not our servers.
    assert spawner.spawned == []
    assert session.health_calls == []
    assert session.ports() == ["8081", "8080"]


def test_server_command_shape():
    cmd = dialog_harness.server_command(
        "llama-server", "models/m.gguf", 9001, {})
    assert cmd[0] == "llama-server"
    assert cmd[cmd.index("-m") + 1] == "models/m.gguf"
    assert cmd[cmd.index("--port") + 1] == "9001"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("-ngl") + 1] == "99"

    tuned = dialog_harness.server_command(
        "llama-server", "models/m.gguf", 9001,
        {"ngl": 33, "args": ["-c", "8192", "-fa"]})
    assert tuned[tuned.index("-ngl") + 1] == "33"
    # `args` is appended verbatim, at the end.
    assert tuned[-3:] == ["-c", "8192", "-fa"]


def test_binary_resolution_order(monkeypatch):
    monkeypatch.setattr(dialog_harness.shutil, "which", lambda _n: None)
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    # Nothing configured, nothing on PATH -> the documented fallback.
    assert dialog_harness.resolve_binary({}) == (
        dialog_harness._FALLBACK_LLAMA_SERVER)

    monkeypatch.setattr(
        dialog_harness.shutil, "which", lambda _n: "/usr/bin/llama-server")
    assert dialog_harness.resolve_binary({}) == "/usr/bin/llama-server"

    monkeypatch.setenv("LLAMA_SERVER", "/env/llama-server")
    assert dialog_harness.resolve_binary({}) == "/env/llama-server"

    # An explicit config entry wins over everything.
    assert dialog_harness.resolve_binary(
        {"binary": "/conf/llama-server"}) == "/conf/llama-server"


def test_model_key_dedupes_equivalent_paths(tmp_path):
    model = _gguf(tmp_path, "m.gguf")
    indirect = os.path.join(str(tmp_path), ".", "m.gguf")
    assert dialog_harness.model_key(model) == (
        dialog_harness.model_key(indirect))
    assert dialog_harness.model_key(model) != (
        dialog_harness.model_key(_gguf(tmp_path, "other.gguf")))


def test_server_log_path_naming():
    assert dialog_harness.server_log_path(
        os.path.join("data", "dialogs", "run1.jsonl"),
        os.path.join("models", "Qwen3-8B-Q6_K.gguf")) == os.path.join(
            "data", "dialogs", "run1.Qwen3-8B-Q6_K.server.log")


def test_free_port_returns_a_bindable_port():
    port = dialog_harness.free_port()
    assert 1024 < port < 65536
    # Released again, so the server can take it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_missing_model_file_aborts_before_launching(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    conf = _local_conf([str(tmp_path / "nope.gguf"), "replaced-below"])
    del conf["participants"][1]["model_path"]
    conf["participants"][1]["base_url"] = "http://127.0.0.1:9999"

    with pytest.raises(FileNotFoundError, match="nope.gguf"):
        dialog_harness.run_experiment(
            conf, str(tmp_path / "out.jsonl"), 1, _FakeSession(["hi"]))
    assert spawner.spawned == []


def test_failed_launch_stops_servers_already_started(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    # The first model exists, the second does not: the first must not
    # be left running.
    conf = _local_conf(
        [_gguf(tmp_path, "a.gguf"), str(tmp_path / "missing.gguf")])

    with pytest.raises(FileNotFoundError):
        dialog_harness.run_experiment(
            conf, str(tmp_path / "out.jsonl"), 1, _FakeSession(["hi"]))
    assert len(spawner.spawned) == 1
    assert spawner.spawned[0].terminated


# --- managed servers: readiness and teardown ------------------------

def test_waits_for_health_before_the_first_request(monkeypatch, tmp_path):
    _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "slow.gguf")
    # Two 503s (still loading) before the server answers 200.
    session = _FakeSession(["0123456789"], healthy_after=2)

    dialog_harness.run_experiment(
        _local_conf([model, model]), str(tmp_path / "out.jsonl"), 1,
        session)

    assert len(session.health_calls) == 3
    assert session.health_calls[0] == "http://127.0.0.1:9001/health"


def test_readiness_timeout_aborts_and_stops_the_server(
        monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "hung.gguf")
    conf = _local_conf([model, model], server={"startup_timeout": 0})
    session = _FakeSession(["never"], healthy_after=10 ** 6)

    with pytest.raises(TimeoutError, match="not ready"):
        dialog_harness.run_experiment(
            conf, str(tmp_path / "out.jsonl"), 1, session)

    assert spawner.spawned[0].terminated
    assert session.calls == []  # no dialog was attempted


def test_server_that_dies_during_load_aborts_with_the_log_tail(
        monkeypatch, tmp_path):
    _install_fake_servers(monkeypatch, exit_code=1)
    model = _gguf(tmp_path, "broken.gguf")

    with pytest.raises(RuntimeError) as excinfo:
        dialog_harness.run_experiment(
            _local_conf([model, model]), str(tmp_path / "out.jsonl"), 1,
            _FakeSession(["never"]))

    message = str(excinfo.value)
    assert "exited with code 1" in message
    assert "server.log" in message
    # The command line is logged, so the tail names the failing model.
    assert "broken.gguf" in message


def test_servers_stop_after_a_normal_run(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "m.gguf")

    dialog_harness.run_experiment(
        _local_conf([model, model]), str(tmp_path / "out.jsonl"), 1,
        _FakeSession(["0123456789"]))

    assert spawner.spawned[0].terminated
    assert spawner.spawned[0].log_file.closed


def test_servers_stop_on_keyboard_interrupt_mid_dialog(
        monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "m.gguf")
    out = str(tmp_path / "out.jsonl")
    # Dialog 1 takes two calls; dialog 2 is interrupted on call 3.
    session = _FakeSession(["0123456789"], errors={2: KeyboardInterrupt()})

    assert dialog_harness.run_experiment(
        _local_conf([model, model]), out, 4, session) == 1

    assert spawner.spawned[0].terminated
    with open(out, encoding="utf-8") as f:
        assert len(f.read().splitlines()) == 1


def test_servers_stop_when_a_dialog_raises(monkeypatch, tmp_path):
    spawner = _install_fake_servers(monkeypatch)
    model = _gguf(tmp_path, "m.gguf")
    # Every retry fails, so the error escapes run_dialog.
    session = _FakeSession(
        ["hi"],
        errors={i: requests.exceptions.HTTPError("500")
                for i in range(dialog_harness._RETRY_ATTEMPTS)})

    with pytest.raises(requests.exceptions.RequestException):
        dialog_harness.run_experiment(
            _local_conf([model, model]), str(tmp_path / "out.jsonl"), 1,
            session)
    assert spawner.spawned[0].terminated


def _stub_platform_kill(monkeypatch):
    """Neutralize the real process-killing calls, recording them."""
    taskkills = []
    groups = []
    monkeypatch.setattr(
        dialog_harness.subprocess, "run",
        lambda cmd, **kw: taskkills.append(cmd))
    if hasattr(os, "killpg"):
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(os, "killpg", lambda pgid, sig: groups.append(pgid))
    return taskkills, groups


def test_kill_tree_targets_the_whole_process_tree(monkeypatch):
    # A wrapper shim would leave the real server as a surviving
    # grandchild, so the kill must address the tree, not the child.
    taskkills, groups = _stub_platform_kill(monkeypatch)
    proc = _FakePopen(["llama-server"], None, exit_code=0)

    dialog_harness.kill_tree(proc, grace=0)

    if os.name == "nt":
        assert taskkills == [
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)]]
    else:
        assert groups == [proc.pid]


def test_kill_tree_escalates_when_the_process_will_not_die(monkeypatch):
    _stub_platform_kill(monkeypatch)

    class _Stubborn(_FakePopen):

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("llama-server", timeout)
            return self.returncode

    proc = _Stubborn(["llama-server"], None)
    dialog_harness.kill_tree(proc, grace=0)
    assert proc.killed


def test_stop_servers_continues_after_one_failure(monkeypatch):

    def kill(proc, grace=0):
        if proc.cmd == ["angry"]:
            raise OSError("access denied")
        proc.terminate()

    monkeypatch.setattr(dialog_harness, "kill_tree", kill)
    good = _FakePopen(["llama-server"], None)
    dialog_harness.stop_servers({
        "a": dialog_harness.Server(
            "a.gguf", 9001, _FakePopen(["angry"], None), "a.log", None),
        "b": dialog_harness.Server("b.gguf", 9002, good, "b.log", None),
    })
    assert good.terminated


def test_run_dialog_honours_a_base_urls_override():
    # Three turns: the opener, then one utterance from each seat.
    conf = _make_conf(min_dialog_bytes=10 ** 6, max_turns=3)
    session = _FakeSession(["reply"])
    dialog_harness.run_dialog(
        conf, session, "hash", ["http://a:1/v1", "http://b:2"])
    assert session.calls[0]["url"] == "http://b:2/v1/chat/completions"
    assert session.calls[1]["url"] == "http://a:1/v1/chat/completions"


# --- dialog_text: rendering JSONL as readable text ------------------

def _dialog_record(names, turns):
    return {"participants": [{"name": n} for n in names], "turns": turns}


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


_TWO_DIALOGS = [
    _dialog_record(
        ["child", "teacher"],
        [{"speaker": 0, "text": "Hi! Can I ask you a question?"},
         {"speaker": 1, "text": "Of course! What would you like to know?"}]),
    _dialog_record(
        ["child", "teacher"],
        [{"speaker": 0, "text": "Why is the sky blue?"}]),
]

_TWO_DIALOGS_TEXT = (
    "====\n"
    "child\n"
    "Hi! Can I ask you a question?\n"
    "\n"
    "teacher\n"
    "Of course! What would you like to know?\n"
    "\n"
    "====\n"
    "child\n"
    "Why is the sky blue?\n"
    "\n"
    "====\n"
)


def test_render_matches_the_requested_layout():
    # A rule before each dialog, a closing rule at the end, and every
    # turn as name / text / blank line.
    assert dialog_text.render(_TWO_DIALOGS) == _TWO_DIALOGS_TEXT


def test_render_of_nothing_is_a_single_rule():
    assert dialog_text.render([]) == "====\n"


def test_speaker_names_come_from_each_records_own_participants():
    records = [
        _dialog_record(["child", "teacher"], [{"speaker": 1, "text": "a"}]),
        _dialog_record(["ann", "bob"], [{"speaker": 1, "text": "b"}]),
    ]
    text = dialog_text.render(records)
    assert "teacher\na\n" in text
    assert "bob\nb\n" in text


def test_speaker_name_falls_back_to_bot_index():
    turns = [{"speaker": 0, "text": "x"}, {"speaker": 1, "text": "y"}]
    # No participants at all.
    assert dialog_text.render([{"turns": turns}]) == (
        "====\nbot0\nx\n\nbot1\ny\n\n====\n")
    # Participants present but unusable, one way each.
    assert dialog_text.speaker_name({"participants": []}, 0) == "bot0"
    assert dialog_text.speaker_name({"participants": "child"}, 0) == "bot0"
    assert dialog_text.speaker_name({"participants": ["child"]}, 0) == "bot0"
    assert dialog_text.speaker_name({"participants": [{}]}, 0) == "bot0"
    assert dialog_text.speaker_name({"participants": [{"name": ""}]}, 0) == (
        "bot0")
    assert dialog_text.speaker_name(_dialog_record(["ann"], []), 1) == "bot1"


def test_unparseable_lines_are_skipped_with_a_warning(tmp_path):
    path = tmp_path / "mixed.jsonl"
    good = json.dumps(_TWO_DIALOGS[0], ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(good + "\n")
        f.write("{not json at all\n")      # e.g. a killed writer
        f.write("\n")                      # blank: ignored silently
        f.write("[1, 2, 3]\n")             # valid JSON, wrong shape
        f.write(good + "\n")

    warnings = []
    records = list(dialog_text.read_records(str(path), warn=warnings.append))

    # Both good records survive; the file is not abandoned.
    assert len(records) == 2
    assert len(warnings) == 2
    assert "mixed.jsonl:2" in warnings[0]
    assert "mixed.jsonl:4" in warnings[1]
    assert "not a JSON object" in warnings[1]


def test_warnings_go_to_stderr_not_the_output(tmp_path, capsys):
    path = tmp_path / "broken.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("{oops\n")
    list(dialog_text.read_records(str(path)))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "skipping unparseable line" in captured.err


def test_multiple_inputs_concatenate_in_order(tmp_path):
    first = _write_jsonl(tmp_path / "a.jsonl", [_TWO_DIALOGS[0]])
    second = _write_jsonl(tmp_path / "b.jsonl", [_TWO_DIALOGS[1]])
    out = str(tmp_path / "both.txt")

    assert dialog_text.main([first, second, "--out", out]) == 0
    with open(out, encoding="utf-8") as f:
        assert f.read() == _TWO_DIALOGS_TEXT

    # Reversed inputs -> reversed dialogs.
    dialog_text.main([second, first, "--out", out])
    with open(out, encoding="utf-8") as f:
        reversed_text = f.read()
    assert (reversed_text.index("Why is the sky blue?")
            < reversed_text.index("Can I ask you a question?"))


def test_out_file_is_utf8_even_for_typographic_model_output(tmp_path):
    fancy = "It’s “blue” — mostly. éè 日"
    path = _write_jsonl(
        tmp_path / "fancy.jsonl",
        [_dialog_record(["child"], [{"speaker": 0, "text": fancy}])])
    out = str(tmp_path / "fancy.txt")

    dialog_text.main([path, "--out", out])

    with open(out, encoding="utf-8") as f:
        assert fancy in f.read()
    # Written as UTF-8 bytes, not escapes or cp1252.
    with open(out, "rb") as f:
        assert "’".encode("utf-8") in f.read()


def test_stdout_is_the_default_destination(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "a.jsonl", _TWO_DIALOGS)
    assert dialog_text.main([path]) == 0
    assert capsys.readouterr().out == _TWO_DIALOGS_TEXT


def test_renders_what_the_harness_actually_writes(tmp_path):
    # End to end: record a dialog with the harness, then read it back.
    conf = _make_conf()
    out = str(tmp_path / "dialogs.jsonl")
    dialog_harness.run(conf, out, 1, session=_FakeSession(["0123456789"]))

    text = dialog_text.render(list(dialog_text.read_records(out)))
    assert text.startswith("====\nchild\nHi!\n\nteacher\n0123456789\n\n")
    assert text.endswith("====\n")
