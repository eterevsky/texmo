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
import random
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
    # s4 selects exactly like s3; the polar answers are an override
    # applied on top, in `render_dialog`.
    ("s4", [("User", "simple"), ("Bot", "trivial")]),
    # s5 too; its speech acts are added turns, not a selection.
    ("s5", [("User", "simple"), ("Bot", "trivial")]),
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
    assert stats == {"dialogs": 2, "turns": 4, "bot_turns": 2, "failed": 1,
                     "collapsed": 0, "fallbacks": 0, "polar": 0,
                     "polar_yes": 0, "polar_no": 0, "lowercased": 0,
                     "greeting": 0, "thanks": 0, "farewell": 0}


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


# -------------------------------------------- yes/no question detection


@pytest.mark.parametrize("text,expected", [
    ("Do you like tea?", "Do you like tea?"),
    ("That is sad. Can you come later?", "Can you come later?"),
    ("Really! Are you sure?", "Are you sure?"),
    ('He said "no". Is that true?', "Is that true?"),
    # Trailing punctuation after the question mark is tolerated.
    ('Is that true?"', "Is that true?"),
    ("I am fine.", None),
    ("Tell me about it", None),
])
def test_final_question_is_the_last_sentence_of_a_question_turn(
        text, expected):
    assert sc.final_question(text) == expected


@pytest.mark.parametrize("text", [
    "Do you like tea?",
    "Does she know?",
    "Did you steal a candy bar today?",
    "Are you coming to the party on Saturday?",
    "Is this the main water valve?",
    "Am I late?",
    "Was it good?",
    "Were they there?",
    "Can I try on your shirt?",
    "Could you help me?",
    "Will you come?",
    "Would you like some?",
    "Have you ever been there?",
    "Has he called?",
    "Had you eaten?",
    "Should I go?",
    "Don't you want to come?",
    "Isn’t that nice?",          # a curly apostrophe
    "That is sad. Can you come later?",   # the question is the 2nd sentence
    "So, do you like it?",                # comma lead-in
    "So did she say yes?",                # bare discourse marker
    "Hey Marcus, are you coming to the party?",   # name lead-in
    "Is it raining, do you know?",        # the comma is not a lead-in
])
def test_is_yes_no_question_accepts_auxiliary_initial_questions(text):
    assert sc.is_yes_no_question(text)


@pytest.mark.parametrize("text", [
    "What do you think?",
    "Well, what do you think?",
    "How much did it cost?",
    "Where do you live?",
    "Why did they do that?",
    "You like tea, right?",               # a tag question, not aux-initial
    "I am fine.",                         # not a question at all
    "Tell me about it.",
    "Do you like tea? What is your name?",   # the last sentence rules
    "May I have some?",                   # a bare request, not a question
    "Shall we go?",
    "If you have the time and you are free, can you help?",  # long lead-in
    "",
    "?",
])
def test_is_yes_no_question_rejects_everything_else(text):
    assert not sc.is_yes_no_question(text)


# ------------------------------------------------- polar classification


@pytest.mark.parametrize("reply,label", [
    ("yes", sc.YES),
    ("Yes", sc.YES),
    ("no", sc.NO),
    ("No.", sc.NO),
    ("other", sc.OTHER),
    ("Neither", sc.OTHER),
    ("<think>\n\n</think>\n\nyes", sc.YES),
    ("```\nno\n```", sc.NO),
])
def test_parse_polar_reads_a_one_word_answer(reply, label):
    assert sc.parse_polar(reply) == (label, False)


def test_parse_polar_digs_a_label_out_of_a_sentence():
    assert sc.parse_polar("The answer is yes.") == (sc.YES, True)


@pytest.mark.parametrize("reply", ["", "hmm", "42"])
def test_parse_polar_rejects_a_reply_without_a_label(reply):
    with pytest.raises(ValueError):
        sc.parse_polar(reply)


def test_polar_messages_share_a_constant_system_prefix():
    a = sc.polar_messages("Do you like tea?", "Yeah, I love it.")
    b = sc.polar_messages("Are you ready?", "Not yet.")
    assert a[0] == b[0]
    assert a[1]["content"] == "User: Do you like tea?\nBot: Yeah, I love it."


def test_polar_jobs_pairs_the_simple_question_with_the_original_reply():
    record = _record(4, [
        _log_turn(sc._USER, "Anyway, do you happen to like tea at all?",
                  "Do you like tea?"),
        _log_turn(sc._BOT, "Absolutely, I drink it daily.", "I like tea.",
                  "Yes, I do."),
        _log_turn(sc._USER, "What else do you drink?", "What else?"),
        _log_turn(sc._BOT, "Coffee.", "Coffee.", "Coffee."),
    ])
    assert sc.polar_jobs([record]) == [
        (4, 1, "Do you like tea?", "Absolutely, I drink it daily.")]


def test_polar_jobs_skips_failures_and_bot_turns_that_open_a_dialog():
    opener = _record(1, [_log_turn(sc._BOT, "Do you like tea?",
                                   "Do you like tea?", "Yes.")])
    assert sc.polar_jobs([opener, {"dialog_idx": 2, "error": "x"}]) == []


def test_polar_map_keeps_the_last_label_per_turn():
    records = [{"dialog_idx": 3, "turn": 1, "label": "yes"},
               {"dialog_idx": 3, "turn": 5, "label": "no"}]
    assert sc.polar_map(records) == {3: {1: "yes", 5: "no"}}


def test_polar_records_round_trip_through_a_log(tmp_path):
    path = tmp_path / "polar.jsonl"
    path.write_text(
        '{"dialog_idx": 3, "turn": 1, "label": "yes"}\n'
        '{"dialog_idx": 3, "turn": 3, "label": "other", "error": "boom"}\n'
        '{"no_keys": true}\n', encoding="utf-8")
    assert sc.polar_keys(str(path)) == {(3, 1), (3, 3)}
    summary = sc.summarize_polar(sc.read_polar_records(str(path)))
    assert summary["classified"] == 2
    assert summary[sc.YES] == 1 and summary[sc.OTHER] == 1
    assert summary["errors"] == 1


# ------------------------------------------------------------- render s4


def _s4_record():
    """One dialog: a polar question, a plain question, two Bot turns."""
    return _record(0, [
        _log_turn(sc._USER, "u0-orig", "Do you like tea?"),
        _log_turn(sc._BOT, "b1-orig", "b1-simp", "I like it."),
        _log_turn(sc._USER, "u2-orig", "What else?"),
        _log_turn(sc._BOT, "b3-orig", "b3-simp", "Coffee."),
    ])


@pytest.mark.parametrize("label,expected", [
    ("yes", "Yes."),
    ("no", "No."),
    # An unclassifiable turn keeps its trivial phrase -- s3 behaviour.
    ("other", "I like it."),
])
def test_render_s4_substitutes_only_the_labelled_turns(label, expected):
    text, stats = sc.render_corpus(
        [_s4_record()], "s4", {0: {1: label}})
    assert text == (f"User: Do you like tea?\n\nBot: {expected}\n\n"
                    f"User: What else?\n\nBot: Coffee.\n")
    if label == sc.OTHER:
        assert stats["polar"] == 0
    else:
        assert stats["polar"] == 1
        assert stats[f"polar_{label}"] == 1


def test_render_s4_without_a_polar_map_is_exactly_s3():
    record = _s4_record()
    s3, _ = sc.render_corpus([record], "s3")
    s4, stats = sc.render_corpus([record], "s4", {})
    assert s4 == s3
    assert stats["polar"] == 0


def test_render_s3_ignores_the_polar_map():
    text, stats = sc.render_corpus([_s4_record()], "s3", {0: {1: "yes"}})
    assert "Yes." not in text
    assert stats["polar"] == 0


def test_render_s4_ignores_a_label_on_a_user_turn():
    # A stale or mis-keyed label must never rewrite the User side.
    text, _ = sc.render_corpus([_s4_record()], "s4", {0: {0: "yes"}})
    assert text.startswith("User: Do you like tea?")


# ------------------------------------------------------------ render s5


class _NoActsRng:
    """A generator that selects no exchange at all."""

    def random(self) -> float:
        return 1.0

    def choices(self, population, weights=None):
        raise AssertionError("no phrase should be drawn")


class _AllActsRng:
    """Selects every exchange, and always the first phrase offered."""

    def random(self) -> float:
        return 0.0

    def choices(self, population, weights=None):
        return [population[0]]


def _rendered(*pairs):
    """Rendered turns -- what `add_speech_acts` operates on."""
    return [{"side": side, "text": text} for side, text in pairs]


def _pairs(turns):
    return [(turn["side"], turn["text"]) for turn in turns]


def _same_side_pairs(turns) -> int:
    return sum(turns[i]["side"] == turns[i + 1]["side"]
               for i in range(len(turns) - 1))


def _alternating_record(dialog_idx, n_turns, first=sc._USER):
    """A strictly alternating dialog, written like the corpus is.

    Capitals and closing full stops, so the case post-step has
    something to change.
    """
    sides = [first, sc._other_side(first)]
    return _record(dialog_idx, [
        _log_turn(sides[i % 2], f"Original {i}.", f"Simple {i}.",
                  f"Trivial {i}.")
        for i in range(n_turns)])


def test_add_speech_acts_selecting_nothing_changes_nothing():
    turns = _rendered(("User", "u0"), ("Bot", "b1"))
    out, added = sc.add_speech_acts(turns, _NoActsRng())
    assert out == turns
    # Pure: the caller's list is not the one that comes back.
    assert out is not turns
    assert added == {act: 0 for act in sc.SPEECH_ACTS}


def test_render_s5_with_nothing_selected_is_exactly_s4(monkeypatch):
    monkeypatch.setattr(sc, "speech_rng", lambda seed, idx: _NoActsRng())
    monkeypatch.setattr(sc, "case_rng", lambda seed, idx: _NoActsRng())
    record = _s4_record()
    polar = {0: {1: "yes"}}
    s4, stats4 = sc.render_corpus([record], "s4", polar)
    s5, stats5 = sc.render_corpus([record], "s5", polar)
    assert s5 == s4
    # The polar override is still s5's, and no exchange went in.
    assert stats5["polar"] == stats4["polar"] == 1
    assert all(stats5[act] == 0 for act in sc.SPEECH_ACTS)
    assert stats5["lowercased"] == 0


def test_speech_acts_wrap_a_user_first_dialog():
    turns = _rendered(("User", "u0"), ("Bot", "b1"))
    out, added = sc.add_speech_acts(turns, _AllActsRng())
    assert _pairs(out) == [
        ("User", "Hi!"), ("Bot", "Hi!"),
        ("User", "u0"), ("Bot", "b1"),
        # Thanks first, then the farewell that closes the dialog.
        ("User", "Thanks!"), ("Bot", "You're welcome!"),
        ("User", "Bye!"), ("Bot", "Bye!")]
    assert added == {act: 1 for act in sc.SPEECH_ACTS}


def test_speech_acts_mirror_a_bot_first_dialog():
    # The greeting must leave the original Bot turn still answering a
    # User turn, so a Bot-opened dialog is greeted by the Bot.
    turns = _rendered(("Bot", "b0"), ("User", "u1"))
    out, _ = sc.add_speech_acts(turns, _AllActsRng())
    assert _pairs(out) == [
        ("Bot", "Hi!"), ("User", "Hi!"),
        ("Bot", "b0"), ("User", "u1"),
        ("Bot", "Thanks!"), ("User", "You're welcome!"),
        ("Bot", "Bye!"), ("User", "Bye!")]


def test_the_closing_exchanges_answer_whoever_closed_the_dialog():
    turns = _rendered(("User", "u0"), ("Bot", "b1"), ("User", "u2"))
    out, _ = sc.add_speech_acts(turns, _AllActsRng())
    assert _pairs(out)[-4:] == [
        ("Bot", "Thanks!"), ("User", "You're welcome!"),
        ("Bot", "Bye!"), ("User", "Bye!")]


def test_add_speech_acts_is_seeded_and_reproducible():
    turns = _rendered(("User", "u0"), ("Bot", "b1"))
    first = sc.add_speech_acts(turns, sc.speech_rng(3, 11))
    assert sc.add_speech_acts(turns, sc.speech_rng(3, 11)) == first
    variants = {
        tuple(_pairs(sc.add_speech_acts(turns, sc.speech_rng(3, i))[0]))
        for i in range(50)}
    # Different dialogs get different exchanges out of the same seed.
    assert len(variants) > 5


def test_rendered_s5_dialogs_alternate_sides():
    records = [_alternating_record(i, 2 + i % 5, sc._SIDES[i % 2])
               for i in range(80)]
    text, stats = sc.render_corpus(records, "s5")
    dialogs = text.rstrip("\n").split("\n\n\n")
    assert len(dialogs) == len(records)
    for dialog in dialogs:
        sides = [line.split(":", 1)[0] for line in dialog.split("\n\n")]
        assert all(a != b for a, b in zip(sides, sides[1:])), dialog
    # The sample has to actually exercise all three exchanges.
    assert all(stats[act] > 0 for act in sc.SPEECH_ACTS)
    assert stats["bot_turns"] > 0


def test_add_speech_acts_adds_no_same_side_pair_to_a_ragged_dialog():
    # Some source dialogs have two turns from one side in a row; the
    # exchanges must not add any more of them.
    turns = _rendered(("User", "a"), ("Bot", "b"), ("Bot", "c"),
                      ("User", "d"))
    for seed in range(40):
        out, _ = sc.add_speech_acts(turns, random.Random(seed))
        assert _same_side_pairs(out) == _same_side_pairs(turns)


def test_add_speech_acts_of_an_empty_dialog_is_empty():
    out, added = sc.add_speech_acts([], random.Random(0))
    assert out == []
    assert added == {act: 0 for act in sc.SPEECH_ACTS}


def test_speech_act_shares_follow_the_constants():
    records = [_alternating_record(i, 4) for i in range(2000)]
    _, stats = sc.render_corpus(records, "s5")
    for act, fraction in ((sc.GREETING, sc._GREETING_FRACTION),
                          (sc.THANKS, sc._THANKS_FRACTION),
                          (sc.FAREWELL, sc._FAREWELL_FRACTION)):
        assert stats[act] / len(records) == pytest.approx(fraction, abs=0.04)


def test_every_greeting_and_farewell_has_a_reply_table():
    assert set(sc._GREETING_REPLIES) == {p for p, _ in sc._GREETINGS}
    assert set(sc._FAREWELL_REPLIES) == {p for p, _ in sc._FAREWELLS}


def test_format_speech_acts_reports_every_act():
    _, stats = sc.render_corpus(
        [_alternating_record(i, 4) for i in range(50)], "s5")
    lines = sc.format_speech_acts(stats, 1)
    assert all(any(act in line for line in lines) for act in sc.SPEECH_ACTS)
    assert "Bot turns" in lines[0]


# ------------------------------------------------- case mirroring (s5)


# `apply_case_style` draws one `random()` and nothing else, so the
# speech-act generators above serve as "never" and "always" here too.


@pytest.mark.parametrize("text,expected", [
    ("I am good.", "i am good"),
    ("Yes.", "yes"),
    # A full stop inside the utterance is not a closing one.
    ("It is late. Bye!", "it is late. bye!"),
    ("I like it. Do you?", "i like it. do you?"),
    # "!" and "?" carry a tone a full stop does not; they stay.
    ("Hi there!", "hi there!"),
    ("Really?", "really?"),
    # An ellipsis is not a full stop either.
    ("Wait...", "wait..."),
    ('He said "no".', 'he said "no"'),
])
def test_lowercase_drops_only_the_closing_full_stop(text, expected):
    assert sc.drop_final_period(text.lower()) == expected


def test_lowercase_dialog_leaves_the_speaker_labels_alone():
    turns = _rendered(("User", "Do you like tea?"), ("Bot", "Yes."))
    assert _pairs(sc.lowercase_dialog(turns)) == [
        ("User", "do you like tea?"), ("Bot", "yes")]


def test_apply_case_style_is_all_or_nothing():
    turns = _rendered(("User", "Hi!"), ("Bot", "I am good."))
    kept, lowered = sc.apply_case_style(turns, _NoActsRng())
    assert kept == turns and kept is not turns and lowered is False
    changed, lowered = sc.apply_case_style(turns, _AllActsRng())
    assert _pairs(changed) == [("User", "hi!"), ("Bot", "i am good")]
    assert lowered is True


def _lowered_line(line: str) -> str:
    """One rendered `Name: text` line as the case post-step leaves it."""
    name, _, text = line.partition(": ")
    return f"{name}: {sc.drop_final_period(text.lower())}"


def test_render_s5_leaves_the_untransformed_dialogs_byte_identical(
        monkeypatch):
    records = [_alternating_record(i, 2 + i % 4, sc._SIDES[i % 2])
               for i in range(120)]
    mixed, stats = sc.render_corpus(records, "s5")
    monkeypatch.setattr(sc, "case_rng", lambda seed, idx: _NoActsRng())
    plain, plain_stats = sc.render_corpus(records, "s5")

    assert plain_stats["lowercased"] == 0
    # The same speech acts either way: the two post-steps draw from
    # separate generators, so one does not shift the other.
    assert all(stats[act] == plain_stats[act] for act in sc.SPEECH_ACTS)
    left = mixed.rstrip("\n").split("\n\n\n")
    right = plain.rstrip("\n").split("\n\n\n")
    assert len(left) == len(right) == len(records)
    changed = 0
    for a, b in zip(left, right):
        if a == b:
            continue
        changed += 1
        assert a == "\n\n".join(
            _lowered_line(line) for line in b.split("\n\n"))
    assert changed == stats["lowercased"] > 0


def test_render_s5_lowercases_the_synthesized_turns_too(monkeypatch):
    monkeypatch.setattr(sc, "speech_rng", lambda seed, idx: _AllActsRng())
    monkeypatch.setattr(sc, "case_rng", lambda seed, idx: _AllActsRng())
    text, stats = sc.render_corpus([_s4_record()], "s5", {0: {1: "yes"}})
    # The greeting, the polar answer and the closing pair, all in the
    # style of the dialog they were added to.
    assert text == (
        "User: hi!\n\nBot: hi!\n\n"
        "User: do you like tea?\n\nBot: yes\n\n"
        "User: what else?\n\nBot: coffee\n\n"
        "User: thanks!\n\nBot: you're welcome!\n\n"
        "User: bye!\n\nBot: bye!\n")
    assert stats["lowercased"] == 1
    assert stats["polar"] == 1


def test_lowercase_share_follows_the_constant():
    records = [_alternating_record(i, 4) for i in range(2000)]
    _, stats = sc.render_corpus(records, "s5")
    assert stats["lowercased"] / len(records) == pytest.approx(
        sc._LOWERCASE_FRACTION, abs=0.04)


def test_s4_is_never_lowercased():
    records = [_alternating_record(i, 4) for i in range(50)]
    text, stats = sc.render_corpus(records, "s4")
    assert stats["lowercased"] == 0
    assert "User: Simple 0." in text


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
