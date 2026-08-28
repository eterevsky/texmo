"""Tests for `scripts/chat_eval.py`.

Same arrangement as `dialog_harness_test.py`: the script lives in
`scripts/` but `pyproject.toml` sets `testpaths = ["texmo"]`, so the
test lives here and reaches the script through the mirror-image import
shim.

Only the pure pieces are covered -- message assembly for both seats,
the resumability keys and the job plan, the repetition metrics, judge
output parsing, the c-without-b check, report building, and the
null-model phrase bot (draws, seeding, distillation, seat record). No
model, no server, no network: `grade_answer` is driven by a fake
session, and the phrase bot needs no endpoint at all.
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
import chat_eval


def _seed(idx=0, opener="Hi there!", script=None, speakers=None):
    script = script or [opener, "Hello!", "How are you?"]
    speakers = speakers or ["User", "Bot", "User"]
    return {
        "idx": idx,
        "opener": opener,
        "script": script,
        "speakers": speakers,
        "n_utterances": len(script),
    }


def _turns(*texts):
    """Alternating turns, examiner first (the forced opener)."""
    sides = [chat_eval._EXAMINER, chat_eval._STUDENT]
    return [
        {"side": sides[i % 2], "text": text, "forced": i == 0,
         "tokens": 0, "elapsed_s": 0.0}
        for i, text in enumerate(texts)
    ]


# ------------------------------------------------------ seeds / prompt


def test_render_script_labels_from_speakers_not_parity():
    # A side taking two turns in a row -- 2.2% of the seed file. Parity
    # would mislabel everything after the repeat.
    seed = _seed(
        script=["a", "b", "c", "d"],
        speakers=["User", "User", "Bot", "User"])
    assert chat_eval.render_script(seed) == (
        "User: a\nUser: b\nBot: c\nUser: d")


def test_examiner_system_prompt_carries_the_whole_script():
    seed = _seed(script=["one", "two", "three"],
                 speakers=["User", "Bot", "User"])
    prompt = chat_eval.examiner_system_prompt(seed)
    for utterance in seed["script"]:
        assert utterance in prompt
    assert "1-2 short sentences" in prompt


def test_load_seeds_takes_a_prefix(tmp_path):
    path = tmp_path / "seeds.jsonl"
    path.write_text(
        "\n".join(json.dumps(_seed(idx=i)) for i in range(5)),
        encoding="utf-8")
    seeds = chat_eval.load_seeds(str(path), 3)
    assert [s["idx"] for s in seeds] == [0, 1, 2]


# ---------------------------------------------------- message assembly


def test_examiner_messages_alternate_from_a_user_turn():
    # Gemma-family templates raise on a leading assistant message, so
    # the kickoff user turn is what makes the examiner seat legal.
    messages = chat_eval.examiner_messages(
        "SYS", _turns("opener", "reply", "second"))
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant"]
    assert messages[0]["content"] == "SYS"
    assert messages[1]["content"] == chat_eval._KICKOFF
    assert messages[2]["content"] == "opener"
    assert messages[3]["content"] == "reply"


def test_student_messages_end_on_the_user_turn():
    messages = chat_eval.student_messages(_turns("opener", "reply", "second"))
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "second"
    # texmo models have no system channel: none is sent.
    assert all(m["role"] != "system" for m in messages)


def test_student_messages_take_a_system_prompt_for_the_good_anchor():
    messages = chat_eval.student_messages(_turns("opener"), "BE NICE")
    assert messages[0] == {"role": "system", "content": "BE NICE"}
    assert messages[1]["role"] == "user"


def test_empty_utterances_become_a_placeholder_in_messages_only():
    turns = _turns("opener", "", "second")
    assert chat_eval.student_messages(turns)[1]["content"] == (
        chat_eval._EMPTY_PLACEHOLDER)
    assert chat_eval.examiner_messages("SYS", turns)[3]["content"] == (
        chat_eval._EMPTY_PLACEHOLDER)
    # The record itself keeps the empty string: an empty answer is a
    # failing answer, not a missing one.
    assert turns[1]["text"] == ""


def test_render_dialog_labels_sides_and_stops_at_upto():
    turns = _turns("opener", "reply", "second", "reply2")
    assert chat_eval.render_dialog(turns, 1) == "User: opener\nBot: reply"


def test_judge_messages_carry_criteria_and_the_dialog():
    messages = chat_eval.judge_messages(_turns("opener", "reply"), 1)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "substantive" in messages[0]["content"]
    assert "I don't know." in messages[0]["content"]
    assert "Bot: reply" in messages[1]["content"]
    assert chat_eval._RETRY_SUFFIX[:20] not in messages[1]["content"]


def test_judge_messages_append_the_retry_suffix():
    messages = chat_eval.judge_messages(
        _turns("opener", "reply"), 1, "no JSON object in the reply")
    assert "could not be parsed as JSON" in messages[1]["content"]
    assert "no JSON object in the reply" in messages[1]["content"]


def test_student_positions_are_one_based():
    turns = _turns("o", "r1", "e2", "r2", "e3")
    assert chat_eval.student_positions(turns) == [(1, 1), (2, 3)]


# ----------------------------------------------------- temperature sweep


def test_parse_temperatures_single_and_list():
    assert chat_eval.parse_temperatures("0.5") == [0.5]
    assert chat_eval.parse_temperatures("0.3, 0.5,0.7") == [0.3, 0.5, 0.7]
    # Duplicates would collide on the resumability key.
    assert chat_eval.parse_temperatures("0.5,0.5") == [0.5]


def test_parse_temperatures_rejects_garbage():
    with pytest.raises(ValueError):
        chat_eval.parse_temperatures("")
    with pytest.raises(ValueError):
        chat_eval.parse_temperatures("-1")
    with pytest.raises(ValueError):
        chat_eval.parse_temperatures("hot")


def test_plan_jobs_sweeps_seeds_within_a_temperature():
    seeds = [_seed(idx=i) for i in range(2)]
    jobs = chat_eval.plan_jobs(seeds, [0.3, 0.7], False, False, 0.3)
    assert [(j[1]["idx"], j[2]) for j in jobs] == [
        (0, 0.3), (1, 0.3), (0, 0.7), (1, 0.7)]
    assert all(j[0] is None for j in jobs)


def test_plan_jobs_runs_anchors_once_at_the_anchor_temperature():
    seeds = [_seed(idx=i) for i in range(5)]
    jobs = chat_eval.plan_jobs(seeds, [0.3, 0.7], True, True, 0.5)
    anchors = [j for j in jobs if j[0] is not None]
    assert len(anchors) == 2 * chat_eval._ANCHOR_SEEDS
    assert {j[0] for j in anchors} == {"good", "garbage"}
    assert {j[2] for j in anchors} == {0.5}
    assert {j[1]["idx"] for j in anchors} == {0, 1, 2}


# ---------------------------------------------------------- resumability


def _dialog_record(seed_idx, temperature, anchor=None):
    return {"seed_idx": seed_idx, "anchor": anchor,
            "student_temperature": temperature, "turns": []}


def test_record_key_separates_anchors_and_temperatures():
    keys = {chat_eval.record_key(r) for r in [
        _dialog_record(0, 0.3),
        _dialog_record(0, 0.7),
        _dialog_record(0, 0.3, "good"),
        _dialog_record(0, 0.3, "garbage"),
    ]}
    assert len(keys) == 4


def test_grade_key_adds_the_answer_position():
    record = dict(_dialog_record(7, 0.5), position=3)
    assert chat_eval.grade_key(record) == (None, 7, 0.5, 3)


def test_existing_keys_drives_the_skip(tmp_path):
    path = tmp_path / "dialogs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in [_dialog_record(0, 0.3), _dialog_record(1, 0.3)]:
            f.write(json.dumps(record) + "\n")
    done = chat_eval.existing_keys(str(path), chat_eval.record_key)
    seeds = [_seed(idx=i) for i in range(2)]
    jobs = chat_eval.plan_jobs(seeds, [0.3, 0.7], False, False, 0.3)
    todo = [j for j in jobs if (j[0], j[1]["idx"], j[2]) not in done]
    assert [(j[1]["idx"], j[2]) for j in todo] == [(0, 0.7), (1, 0.7)]


def test_existing_keys_on_a_missing_file_is_empty(tmp_path):
    assert chat_eval.existing_keys(
        str(tmp_path / "nope.jsonl"), chat_eval.record_key) == set()


def test_existing_keys_survives_a_truncated_line(tmp_path):
    path = tmp_path / "dialogs.jsonl"
    path.write_text(
        json.dumps(_dialog_record(0, 0.3)) + '\n{"seed_idx": 1, "anch',
        encoding="utf-8")
    assert chat_eval.existing_keys(
        str(path), chat_eval.record_key) == {(None, 0, 0.3)}


# ----------------------------------------------------- repetition metrics


def test_repeated_ngram_fraction():
    assert chat_eval.repeated_ngram_fraction("one two three four") == 0.0
    # Too short to have a 3-gram at all.
    assert chat_eval.repeated_ngram_fraction("hi") == 0.0
    stuck = chat_eval.repeated_ngram_fraction("a b c a b c a b c a b c")
    assert stuck > 0.5


def test_longest_repeated_substring():
    assert chat_eval.longest_repeated_substring("abcabc") == 3
    assert chat_eval.longest_repeated_substring("abcdef") == 0
    assert chat_eval.longest_repeated_substring("") == 0


def test_repetition_stats_shape():
    stats = chat_eval.repetition_stats("hello hello hello")
    assert stats["chars"] == 17
    # Three words are exactly one 3-gram, so the n-gram rate cannot see
    # this repeat -- the substring measure is what catches it.
    assert stats["rep3"] == 0.0
    assert stats["lrs_ratio"] > 0.5
    assert chat_eval.repetition_stats("")["lrs_ratio"] == 0.0


# -------------------------------------------------- judge output parsing


def test_parse_plain_json():
    grade = chat_eval.parse_judge_output(
        '{"a": true, "b": false, "c": false, "examiner_defect": null, '
        '"comment": "fine"}')
    assert (grade["a"], grade["b"], grade["c"]) == (True, False, False)
    assert grade["examiner_defect"] is None
    assert grade["comment"] == "fine"


def test_parse_strips_markdown_fences():
    grade = chat_eval.parse_judge_output(
        '```json\n{"a": true, "b": true, "c": true, '
        '"examiner_defect": null, "comment": ""}\n```')
    assert grade["c"] is True


def test_parse_strips_a_leaked_thinking_block():
    grade = chat_eval.parse_judge_output(
        '<think>hmm, the reply is fine</think>\n'
        '{"a": true, "b": true, "c": false, "examiner_defect": null, '
        '"comment": "ok"}')
    assert grade["b"] is True


def test_parse_tolerates_prose_around_the_object():
    grade = chat_eval.parse_judge_output(
        'Here is my grade:\n{"a": false, "b": false, "c": false, '
        '"comment": "nonsense"}\nHope that helps!')
    assert grade["a"] is False
    assert grade["examiner_defect"] is None


def test_parse_accepts_yes_no_strings():
    grade = chat_eval.parse_judge_output(
        '{"a": "yes", "b": "no", "c": "no", "comment": "x"}')
    assert (grade["a"], grade["b"], grade["c"]) == (True, False, False)


@pytest.mark.parametrize("value", ['"  "', '"null"', '"None."', '"n/a"',
                                   'null'])
def test_parse_normalizes_a_non_defect_to_none(value):
    # Judges write the string "null" about as often as JSON null, and
    # an untouched one would report a defect on every single answer.
    grade = chat_eval.parse_judge_output(
        '{"a": true, "b": true, "c": true, "examiner_defect": '
        + value + "}")
    assert grade["examiner_defect"] is None


def test_parse_keeps_a_real_defect_under_either_key():
    # The prompt asks for `user_problem`; `examiner_defect` (the record
    # field name) stays accepted as an alias.
    for key in ("user_problem", "examiner_defect"):
        grade = chat_eval.parse_judge_output(
            '{"a": true, "b": true, "c": true, '
            f'"{key}": "User repeated its question."}}')
        assert grade["examiner_defect"] == "User repeated its question."


def test_parse_rejects_missing_and_unparseable():
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('{"a": true, "b": true}')
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output("I would say the reply is fine.")
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('{"a": "maybe", "b": true, "c": true}')
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('["a", "b"]')


def test_c_without_b_is_flagged():
    assert chat_eval.is_inconsistent({"a": True, "b": False, "c": True})
    assert not chat_eval.is_inconsistent({"a": True, "b": True, "c": True})
    assert not chat_eval.is_inconsistent({"a": True, "b": False, "c": False})
    # A judge error carries None everywhere and is not an inconsistency.
    assert not chat_eval.is_inconsistent({"a": None, "b": None, "c": None})


# ------------------------------------------------- null model (phrase bot)


def _phrase_student(pairs, seed=0):
    texts = [t for t, _ in pairs]
    weights = [w for _, w in pairs]
    return chat_eval.PhraseStudent(
        texts, weights,
        chat_eval.phrase_participant("student", "p.json", seed, len(texts)),
        seed)


def test_phrase_student_draws_follow_the_weights():
    student = _phrase_student([("often", 80), ("rarely", 20)])
    draws = [student.reply(None, [], None)[0] for _ in range(4000)]
    # 4000 draws at p=0.8 has sd ~0.6%, so 3 points is many sigma;
    # the seed makes it deterministic anyway.
    assert abs(draws.count("often") / len(draws) - 0.8) < 0.03


def test_phrase_student_ignores_the_context_and_costs_nothing():
    student = _phrase_student([("only", 1)])
    # No session, no URL, no timeout -- and the turns are not read.
    text, tokens, elapsed = student.reply(
        None, _turns("opener", "reply", "second"), None)
    assert (text, tokens, elapsed) == ("only", 0, 0.0)


def _draws(student, n=50):
    return [student.reply(None, [], None)[0] for _ in range(n)]


def test_phrase_student_is_reproducible_per_seed():
    pairs = [("a", 3), ("b", 2), ("c", 1)]
    first = _draws(_phrase_student(pairs))
    assert len(set(first)) > 1
    # One stream per run, so the same seed replays the whole run.
    assert _draws(_phrase_student(pairs)) == first
    assert _draws(_phrase_student(pairs, seed=7)) != first


def test_phrase_student_needs_a_phrase():
    with pytest.raises(ValueError):
        chat_eval.PhraseStudent([], [], {}, 0)


def test_phrase_participant_records_the_provenance():
    seat = chat_eval.phrase_participant(
        "student", "data/eval/phrases-s3-top20.json", 0, 20)
    assert seat["kind"] == "phrases"
    assert seat["phrases_path"] == "data/eval/phrases-s3-top20.json"
    assert seat["phrase_seed"] == 0
    assert seat["n_phrases"] == 20
    # No sampling params: neither exists for this seat.
    assert "temperature" not in seat and "manifest" not in seat


def _write_phrase_file(tmp_path, pairs):
    path = tmp_path / "phrases.json"
    path.write_text(
        json.dumps({"phrases": [{"text": t, "weight": w} for t, w in pairs]}),
        encoding="utf-8")
    return str(path)


def test_load_phrase_file_round_trip(tmp_path):
    path = _write_phrase_file(tmp_path, [("hi", 10), ("bye", 3)])
    phrases, weights = chat_eval.load_phrase_file(path)
    # Raw counts, not normalized: `random.choices` normalizes.
    assert phrases == ["hi", "bye"]
    assert weights == [10.0, 3.0]


def test_load_phrase_file_rejects_empty_and_non_positive(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"phrases": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        chat_eval.load_phrase_file(str(empty))
    with pytest.raises(ValueError):
        chat_eval.load_phrase_file(_write_phrase_file(tmp_path, [("x", 0)]))


# ------------------------------------------------------- phrase distillation


def _dialog(seed_idx, *texts, anchor=None):
    return {"seed_idx": seed_idx, "anchor": anchor,
            "student_temperature": None, "turns": _turns(*texts)}


def test_count_answers_strips_and_skips_anchors():
    dialogs = [
        _dialog(0, "q1", "  Hi.  ", "q2", "Hi."),
        _dialog(1, "q1", "Bye.", "q2", "Hi."),
        _dialog(2, "q1", "ANCHOR", "q2", "ANCHOR", anchor="good"),
    ]
    counts = chat_eval.count_answers(dialogs)
    assert counts == {"Hi.": 3, "Bye.": 1}


def test_phrase_table_carries_coverage_and_provenance():
    counts = chat_eval.count_answers([
        _dialog(0, "q", "a", "q", "a"),
        _dialog(1, "q", "a", "q", "b"),
        _dialog(2, "q", "c", "q", "d"),
    ])
    table = chat_eval.phrase_table(counts, 2, "d.jsonl")
    assert table["source_dialogs"] == "d.jsonl"
    assert table["n_answers"] == 6
    assert table["n_distinct"] == 4
    assert table["top_n"] == 2
    # "a" three times plus one of the singletons: 4 of 6.
    assert table["coverage"] == pytest.approx(4 / 6, abs=1e-4)
    assert table["phrases"][0] == {"text": "a", "weight": 3}
    assert len(table["phrases"]) == 2


def test_phrases_subcommand_writes_a_file_the_bot_can_load(tmp_path):
    dialogs_path = tmp_path / "dialogs.jsonl"
    dialogs_path.write_text(
        "\n".join(json.dumps(r) for r in [
            _dialog(0, "q", "Hi.", "q", "Hi."),
            _dialog(1, "q", "Hi.", "q", "Bye."),
            _dialog(2, "q", "NOPE", "q", "NOPE", anchor="garbage"),
        ]),
        encoding="utf-8")
    out_path = tmp_path / "phrases.json"
    args = chat_eval.build_parser().parse_args([
        "phrases", "--dialogs", str(dialogs_path), "--top", "5",
        "--out", str(out_path)])
    assert args.func(args) == 0
    phrases, weights = chat_eval.load_phrase_file(str(out_path))
    assert phrases == ["Hi.", "Bye."]
    assert weights == [3.0, 1.0]
    # The anchor dialog contributes nothing.
    assert "NOPE" not in phrases


# ------------------------------------------------- grading with a fake seat


class _FakeResponse:

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Hands out canned completion contents, recording every body."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.bodies = []

    def post(self, url, json=None, timeout=None):
        self.bodies.append(json)
        content = self.contents.pop(0) if self.contents else ""
        return _FakeResponse({
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": 7},
        })


def _judge_seat():
    return {"model": "judge.gguf", "temperature": 0.0, "max_tokens": 400,
            "extra": chat_eval._NO_THINKING}


def test_grade_answer_retries_once_then_succeeds():
    session = _FakeSession([
        "I think it is fine, honestly.",
        '{"a": true, "b": true, "c": false, "examiner_defect": null, '
        '"comment": "deflection"}',
    ])
    grade = chat_eval.grade_answer(
        session, "http://x", _judge_seat(), _turns("opener", "reply"), 1,
        (1, 1))
    assert grade["judge_error"] is None
    assert (grade["a"], grade["b"], grade["c"]) == (True, True, False)
    assert len(session.bodies) == 2
    # The retry carries the correcting suffix, and the house rule rides
    # on every request.
    assert "could not be parsed" in session.bodies[1]["messages"][1][
        "content"]
    assert session.bodies[0]["chat_template_kwargs"] == {
        "enable_thinking": False}


def test_grade_answer_gives_up_after_the_retry():
    session = _FakeSession(["nope", "still nope"])
    grade = chat_eval.grade_answer(
        session, "http://x", _judge_seat(), _turns("opener", "reply"), 1,
        (1, 1))
    assert grade["a"] is None
    assert grade["judge_error"]
    assert grade["raw"] == "still nope"
    assert len(session.bodies) == 2


# ------------------------------------------------------------- reporting


def _grade(a, b, c, temperature=0.5, position=1, anchor=None, seed_idx=0,
           answer="something", defect=None, error=None):
    return {
        "a": a, "b": b, "c": c,
        "examiner_defect": defect, "comment": "",
        "judge_error": error, "raw": "",
        "seed_idx": seed_idx, "anchor": anchor,
        "student_temperature": temperature, "position": position,
        "answer": answer, "repetition": chat_eval.repetition_stats(answer),
        "judge_inconsistency": chat_eval.is_inconsistent(
            {"b": b, "c": c}),
        "judge_model": "judge.gguf",
    }


def test_summarize_ignores_judge_errors_in_the_rates():
    grades = [_grade(True, True, True), _grade(None, None, None,
                                               error="no JSON")]
    summary = chat_eval.summarize(grades)
    assert summary["n"] == 2
    assert summary["n_valid"] == 1
    assert summary["rates"]["a"] == 100.0
    assert summary["errors"] == 1


def test_summarize_deflection_is_b_minus_c():
    grades = [_grade(True, True, True), _grade(True, True, False)]
    summary = chat_eval.summarize(grades)
    assert summary["rates"]["b"] == 100.0
    assert summary["rates"]["c"] == 50.0
    assert summary["deflection"] == 50.0


def test_verdict_needs_every_target():
    passing = chat_eval.summarize([_grade(True, True, True)] * 10)
    assert chat_eval.verdict(passing) == "PASS"
    # a and b at 100%, c at 0% -> c misses its 50% target.
    failing = chat_eval.summarize([_grade(True, True, False)] * 10)
    assert chat_eval.verdict(failing) == "FAIL"


def test_report_has_one_row_per_temperature_and_the_anchors():
    grades = (
        [_grade(True, True, True, temperature=0.3)] * 4
        + [_grade(True, False, False, temperature=0.7, position=2)] * 4
        + [_grade(True, True, True, anchor="good", temperature=0.3)]
        + [_grade(False, False, False, anchor="garbage", temperature=0.3,
                  answer="qqq qqq qqq", defect="User repeated itself")]
    )
    report = chat_eval.build_report("d.jsonl", "g.jsonl", grades)
    assert "| 0.3 | 4 | 100.0% | 100.0% | 100.0% |" in report
    assert "| 0.7 | 4 |" in report
    assert "| good |" in report and "| garbage |" in report
    assert "User repeated itself" in report
    assert "PASS" in report and "FAIL" in report


def test_report_survives_an_empty_main_set():
    report = chat_eval.build_report(
        "d.jsonl", "g.jsonl", [_grade(True, True, True, anchor="good")])
    assert "0 student answers" in report
    assert "n/a" in report
