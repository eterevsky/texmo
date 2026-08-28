"""Tests for `scripts/simplify_corpus.py`.

Same arrangement as `chat_eval_test.py` and `dialog_harness_test.py`:
the script lives in `scripts/` but `pyproject.toml` sets
`testpaths = ["texmo"]`, so the test lives here and reaches the script
through the mirror-image import shim.

No model, no server, no parquet: the source filter, the reply
validation (retry path included, via a fake session), the
resumability set, the render selection table and the degenerate-output
warning are all pure. The script's pyarrow import is guarded exactly
so this module imports in an environment without it.
"""
import json
import os
import sys

import pytest

# Appended, not prepended: scripts/ holds modules whose names collide
# with real packages (e.g. coverage.py).
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts"))
import simplify_corpus as sc


def _turns(*texts):
    """Source turns, User first and strictly alternating."""
    return [{"side": sc._SIDES[i % 2], "text": text}
            for i, text in enumerate(texts)]


def _reply(turns, simple=None, trivial=None):
    """A well-formed model reply for `turns`, as a JSON string."""
    out = []
    for i, turn in enumerate(turns):
        entry = {
            "i": i,
            "side": turn["side"],
            "simple": simple or f"simple {i}",
            "trivial": None,
        }
        if turn["side"] == sc._BOT:
            entry["trivial"] = trivial or f"trivial {i}"
        out.append(entry)
    return json.dumps(out)


def _log_turn(side, original="orig", simple="simp", trivial=None):
    return {"side": side, "original": original, "simple": simple,
            "trivial": trivial}


def _record(dialog_idx=0, turns=None):
    return {
        "dialog_idx": dialog_idx,
        "turns": turns if turns is not None else [
            _log_turn(sc._USER),
            _log_turn(sc._BOT, trivial="triv"),
        ],
        "model": "m.gguf",
        "prompt_version": sc._PROMPT_VERSION,
        "elapsed_s": 1.0,
    }


class _FakeResponse:

    def __init__(self, text):
        self._text = text
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._text}}],
                "usage": {"completion_tokens": 7}}


class _FakeSession:
    """Hands out canned completion texts in order, recording bodies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.bodies = []

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        text = self.replies.pop(0) if self.replies else ""
        return _FakeResponse(text)


# ------------------------------------------------------- source filter


def test_collapse_flattens_multiline_utterances():
    assert sc.collapse("  hello  ") == ("hello", False)
    assert sc.collapse("one\ntwo") == ("one two", True)
    assert sc.collapse("one\r\n   two") == ("one two", True)


def test_dialog_turns_renames_in_first_appearance_order():
    stats = sc.SourceStats()
    turns = sc.dialog_turns(
        ["Hi", "Hello", "Bye"], ["Ada", "Bob", "Ada"], stats)
    assert [t["side"] for t in turns] == ["User", "Bot", "User"]
    assert [t["text"] for t in turns] == ["Hi", "Hello", "Bye"]


def test_dialog_turns_collapses_and_counts_newlines():
    stats = sc.SourceStats()
    turns = sc.dialog_turns(["a\nb", "c"], ["Ada", "Bob"], stats)
    assert turns[0]["text"] == "a b"
    assert stats.collapsed == 1


@pytest.mark.parametrize("dialogue,speakers,attr", [
    (["a", "b"], ["Ada"], "skip_mismatch"),
    ([], [], "skip_empty_dialog"),
    (["a", "b", "c"], ["Ada", "Bob", "Cy"], "skip_speakers"),
    (["a", "   "], ["Ada", "Bob"], "skip_empty_utterance"),
    (["a", None], ["Ada", "Bob"], "skip_empty_utterance"),
])
def test_dialog_turns_drops_and_counts_bad_rows(dialogue, speakers, attr):
    stats = sc.SourceStats()
    assert sc.dialog_turns(dialogue, speakers, stats) is None
    assert getattr(stats, attr) == 1


# -------------------------------------------------------------- prompt


def test_render_numbered_indexes_every_turn():
    assert sc.render_numbered(_turns("hi", "hello")) == (
        "0 User: hi\n1 Bot: hello")


def test_build_messages_carry_the_rules_and_the_dialog():
    messages = sc.build_messages(_turns("hi", "hello"))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "A2" in messages[0]["content"]
    assert "trivial" in messages[0]["content"]
    assert "0 User: hi" in messages[1]["content"]
    assert "exactly 2 objects" in messages[1]["content"]
    assert "was not usable" not in messages[1]["content"]


def test_build_messages_append_the_retry_suffix():
    messages = sc.build_messages(_turns("hi", "hello"), "turn 1 is missing")
    assert "was not usable" in messages[1]["content"]
    assert "turn 1 is missing" in messages[1]["content"]


# ---------------------------------------------------------- validation


def test_parse_accepts_a_well_formed_array():
    turns = _turns("hi", "hello there", "bye")
    parsed = sc.parse_simplification(_reply(turns), turns)
    assert [t["side"] for t in parsed] == ["User", "Bot", "User"]
    assert [t["original"] for t in parsed] == ["hi", "hello there", "bye"]
    assert parsed[1]["trivial"] == "trivial 1"
    # A User turn never needs a trivial phrase (s3 takes User from
    # `simple`), and none was offered.
    assert parsed[0]["trivial"] is None


def test_parse_unwraps_markdown_fences():
    turns = _turns("hi", "hello")
    fenced = f"```json\n{_reply(turns)}\n```"
    assert len(sc.parse_simplification(fenced, turns)) == 2


def test_parse_digs_the_array_out_of_prose():
    turns = _turns("hi", "hello")
    noisy = f"Sure, here you go:\n{_reply(turns)}\nHope that helps!"
    assert len(sc.parse_simplification(noisy, turns)) == 2


def test_parse_accepts_out_of_order_indexes():
    turns = _turns("hi", "hello")
    entries = json.loads(_reply(turns))
    reordered = json.dumps([entries[1], entries[0]])
    parsed = sc.parse_simplification(reordered, turns)
    assert [t["side"] for t in parsed] == ["User", "Bot"]
    assert parsed[0]["simple"] == "simple 0"


@pytest.mark.parametrize("reply,message", [
    ("no json here", "no JSON array"),
    ('{"i": 0}', "expected a JSON array"),
    ('[{"i": 0, "side": "User", "simple": "s"}]', "expected 2 objects"),
    ('[{"i": 0, "side": "User", "simple": "s", "trivial": null},'
     ' {"i": 0, "side": "Bot", "simple": "s", "trivial": "t"}]',
     "appears twice"),
    ('[{"i": 0, "side": "User", "simple": "s", "trivial": null},'
     ' {"i": 5, "side": "Bot", "simple": "s", "trivial": "t"}]',
     "out of range"),
    ('[{"i": 0, "side": "Bot", "simple": "s", "trivial": "t"},'
     ' {"i": 1, "side": "User", "simple": "s", "trivial": null}]',
     "expected 'User'"),
    ('[{"i": 0, "side": "User", "simple": "", "trivial": null},'
     ' {"i": 1, "side": "Bot", "simple": "s", "trivial": "t"}]',
     "'simple' is not a non-empty string"),
    ('[{"i": 0, "side": "User", "simple": "s", "trivial": null},'
     ' {"i": 1, "side": "Bot", "simple": "s", "trivial": null}]',
     "'trivial' is not a non-empty string"),
])
def test_parse_rejects_bad_replies(reply, message):
    turns = _turns("hi", "hello")
    with pytest.raises(ValueError, match=message):
        sc.parse_simplification(reply, turns)


def test_parse_keeps_a_trivial_offered_on_a_user_turn():
    turns = _turns("hi", "hello")
    reply = ('[{"i": 0, "side": "User", "simple": "s", "trivial": "Hi."},'
             ' {"i": 1, "side": "Bot", "simple": "s", "trivial": "t"}]')
    assert sc.parse_simplification(reply, turns)[0]["trivial"] == "Hi."


# ------------------------------------------------------- request/retry


def _process(replies, turns=None):
    turns = turns or _turns("hi", "hello")
    session = _FakeSession(replies)
    record = sc.process_dialog(
        session, "http://x", sc.build_participant("m.gguf", 0.3, 100),
        7, turns, "m.gguf", (1, 1))
    return record, session


def test_process_dialog_returns_a_record_on_the_first_try():
    record, session = _process([_reply(_turns("hi", "hello"))])
    assert record["dialog_idx"] == 7
    assert len(session.bodies) == 1
    assert [t["simple"] for t in record["turns"]] == ["simple 0", "simple 1"]
    assert record["model"] == "m.gguf"
    assert record["prompt_version"] == sc._PROMPT_VERSION
    assert "error" not in record
    # The house rule rides on every request.
    assert session.bodies[0]["chat_template_kwargs"] == {
        "enable_thinking": False}


def test_process_dialog_retries_once_with_the_error():
    record, session = _process(
        ["sorry, I can't do that", _reply(_turns("hi", "hello"))])
    assert "error" not in record
    assert len(session.bodies) == 2
    retry = session.bodies[1]["messages"][1]["content"]
    assert "was not usable" in retry
    assert "no JSON array" in retry


def test_process_dialog_records_the_error_after_two_failures():
    record, session = _process(["nope", "still nope"])
    assert len(session.bodies) == 2
    assert "no JSON array" in record["error"]
    assert "turns" not in record
    assert record["dialog_idx"] == 7


# --------------------------------------------------------- resumability


def test_existing_indices_covers_successes_and_failures(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        json.dumps(_record(0)) + "\n"
        + json.dumps({"dialog_idx": 3, "error": "boom"}) + "\n"
        + "not json at all\n"
        + json.dumps(_record(5)) + "\n",
        encoding="utf-8")
    assert sc.existing_indices(str(path)) == {0, 3, 5}


def test_existing_indices_of_a_missing_file_is_empty(tmp_path):
    assert sc.existing_indices(str(tmp_path / "nope.jsonl")) == set()


# ------------------------------------------------------------- render


@pytest.mark.parametrize("variant,expected", [
    ("s0", [("User", "original"), ("Bot", "original")]),
    ("s1", [("User", "original"), ("Bot", "simple")]),
    ("s2", [("User", "original"), ("Bot", "trivial")]),
    ("s3", [("User", "simple"), ("Bot", "trivial")]),
])
def test_select_text_follows_the_variant_table(variant, expected):
    turns = [
        _log_turn(sc._USER, "U-orig", "U-simp", "U-triv"),
        _log_turn(sc._BOT, "B-orig", "B-simp", "B-triv"),
    ]
    got = [sc.select_text(turn, variant) for turn in turns]
    assert got == [
        (f"{side[0]}-{field[:4]}", field) for side, field in expected]


def test_select_text_falls_back_when_a_field_is_missing():
    turn = _log_turn(sc._BOT, "B-orig", "B-simp", None)
    assert sc.select_text(turn, "s2") == ("B-simp", "simple")
    bare = {"side": sc._BOT, "original": "B-orig"}
    assert sc.select_text(bare, "s2") == ("B-orig", "original")


def test_render_corpus_sorts_skips_failures_and_formats():
    records = [
        _record(2, [_log_turn(sc._USER, "u2"), _log_turn(sc._BOT, "b2",
                                                         "s2", "t2")]),
        {"dialog_idx": 1, "error": "boom"},
        _record(0, [_log_turn(sc._USER, "u0"), _log_turn(sc._BOT, "b0",
                                                         "s0", "t0")]),
    ]
    text, stats = sc.render_corpus(records, "s1")
    assert text == ("User: u0\n\nBot: s0\n\n\nUser: u2\n\nBot: s2\n")
    assert stats == {"dialogs": 2, "turns": 4, "failed": 1,
                     "collapsed": 0, "fallbacks": 0}


def test_render_corpus_collapses_newlines_and_counts_them():
    records = [_record(0, [
        _log_turn(sc._USER, "u\nsplit"),
        _log_turn(sc._BOT, "b", "one\ntwo", "t"),
    ])]
    text, stats = sc.render_corpus(records, "s1")
    assert text == "User: u split\n\nBot: one two\n"
    assert stats["collapsed"] == 2


def test_render_corpus_of_nothing_is_empty():
    text, stats = sc.render_corpus([{"dialog_idx": 0, "error": "x"}], "s2")
    assert text == ""
    assert stats["dialogs"] == 0 and stats["failed"] == 1


# ------------------------------------------------------------ summary


def test_trivial_key_folds_case_and_punctuation():
    keys = {sc.trivial_key(t) for t in ["Yes.", "yes", "YES!", " Yes  "]}
    assert keys == {"yes"}


def _trivial_records(phrases):
    """One dialog per phrase: a User turn and a Bot turn."""
    return [
        _record(i, [_log_turn(sc._USER, "user text here"),
                    _log_turn(sc._BOT, "bot text", "simpler", phrase)])
        for i, phrase in enumerate(phrases)
    ]


def test_summarize_counts_chars_per_side_and_field():
    summary = sc.summarize(_trivial_records(["Yes.", "No."]))
    assert summary["dialogs"] == 2 and summary["failed"] == 0
    assert summary["turns"] == {"User": 2, "Bot": 2}
    assert summary["chars"]["User"]["original"] == len("user text here")
    # No trivial phrase on the User side.
    assert summary["chars"]["User"]["trivial"] is None
    assert summary["chars"]["Bot"]["simple"] == len("simpler")
    # "Yes." and "No.": 4 and 3 characters.
    assert summary["chars"]["Bot"]["trivial"] == 3.5


def test_summarize_counts_failures_without_reading_turns():
    records = _trivial_records(["Yes."]) + [{"dialog_idx": 9, "error": "x"}]
    summary = sc.summarize(records)
    assert summary["dialogs"] == 1 and summary["failed"] == 1


def test_summary_warns_loudly_on_a_degenerate_distribution():
    # 6 of 10 Bot turns say "Yes" -- an s2 corpus with one lesson in it.
    summary = sc.summarize(_trivial_records(
        ["Yes."] * 6 + ["No.", "I don't know.", "Me too.", "Sure!"]))
    assert summary["trivial_distinct"] == 5
    assert summary["trivial_top"] == "yes"
    assert summary["trivial_top_share"] == pytest.approx(0.6)
    assert any("WARNING" in line for line in sc.format_summary(summary))


def test_summary_is_quiet_on_a_varied_distribution():
    summary = sc.summarize(_trivial_records(
        ["Yes.", "No.", "I don't know.", "Me too.", "Sure!"]))
    assert summary["trivial_top_share"] == pytest.approx(0.2)
    assert not any("WARNING" in line for line in sc.format_summary(summary))
