"""Tests for `scripts/chat_eval.py`.

Same arrangement as `dialog_harness_test.py`: the script lives in
`scripts/` but `pyproject.toml` sets `testpaths = ["texmo"]`, so the
test lives here and reaches the script through the mirror-image import
shim.

Only the pure pieces are covered -- message assembly for both seats,
the resumability keys and the job plan, the style mixing (the case
transform, the seeded per-seed draw, the forced greeting/farewell turn
structure), the repetition metrics, judge output parsing, the
c-without-b check, report building, and the null-model phrase bot
(draws, seeding, distillation, seat record) and the ELIZA reference
student. No model, no server, no network: `grade_answer` and
`run_dialog` are driven by a fake session, and neither reference
student needs an endpoint at all.
"""
import json
import os
import sys
import threading

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


def test_pass_a_sees_the_utterance_and_nothing_else():
    # The whole point of the split: context cannot leak into (a).
    turns = _turns("what is your favourite colour", "reply", "e2", "reply2")
    messages = chat_eval.judge_messages("a", turns, 3)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "reply2" in messages[1]["content"]
    for other in ("what is your favourite colour", "e2", "User:", "Bot:"):
        assert other not in messages[1]["content"]
    # Nor does the system prompt smuggle the conversation back in.
    assert "conversation so far" not in messages[0]["content"].lower()
    assert "word salad" in messages[0]["content"]


def test_pass_b_sees_only_the_pair():
    turns = _turns("opener", "reply", "the question", "the answer")
    content = chat_eval.judge_messages("b", turns, 3)[1]["content"]
    assert "User: the question" in content
    assert "Bot: the answer" in content
    assert "opener" not in content and "reply" not in content


def test_pass_b_pairs_with_the_last_user_turn_not_the_previous_index():
    # Two student turns in a row: the question answered is the last
    # User line, not turns[upto - 1].
    turns = [
        {"side": chat_eval._EXAMINER, "text": "the question"},
        {"side": chat_eval._STUDENT, "text": "first"},
        {"side": chat_eval._STUDENT, "text": "second"},
    ]
    content = chat_eval.judge_messages("b", turns, 2)[1]["content"]
    assert "User: the question" in content
    assert "Bot: second" in content


def test_pass_a_ignores_capitalization_and_a_missing_final_period():
    # A lower-case examiner turn is answered in kind by a model that
    # learned to mirror, and that must not cost it (a).
    system = chat_eval.judge_messages("a", _turns("hi", "i am good"), 1)[0][
        "content"]
    assert "Capitalization" in system
    assert '"i am good"' in system


def test_pass_c_counts_a_minimal_direct_answer_as_substantive():
    # The laconic-answer ruling: "Yes." to a polar question resolves it.
    system = chat_eval.judge_messages("c", _turns("o", "Yes."), 1)[0][
        "content"]
    assert "RESOLVES the question" in system
    assert "a direct answer is never a deflection" in system
    assert '"Do you like cats?"' in system
    assert "counter-questions" in system


def test_pass_c_carries_the_whole_dialog_and_the_substantive_example():
    turns = _turns("opener", "reply", "e2", "reply2")
    messages = chat_eval.judge_messages("c", turns, 3)
    assert "substantive" in messages[0]["content"]
    assert "I don't know." in messages[0]["content"]
    assert "user_problem" in messages[0]["content"]
    assert "User: opener\nBot: reply\nUser: e2\nBot: reply2" in (
        messages[1]["content"])


def test_pass_prompts_ask_for_their_own_key_only():
    systems = {p: chat_eval.judge_messages(p, _turns("o", "r"), 1)[0][
        "content"] for p in chat_eval._PASSES}
    assert '"a": <true or false>' in systems["a"]
    assert '"b"' not in systems["a"] and '"c"' not in systems["a"]
    assert '"b": <true or false>' in systems["b"]
    assert '"c": <true or false>' in systems["c"]
    # `comment` first in all three: the keys are generated in order, so
    # it is the judge's one-line chain of thought.
    for system in systems.values():
        assert system.index('"comment"') < system.index("true or false")


def test_pass_system_prompts_are_stable_across_answers():
    # The prompt cache keys on the shared prefix: two different answers
    # must produce the same system bytes, or every request re-evaluates
    # the whole prompt.
    first = chat_eval.judge_messages("a", _turns("o", "one"), 1)[0]
    second = chat_eval.judge_messages("a", _turns("o", "two"), 1)[0]
    assert first["content"] == second["content"]


@pytest.mark.parametrize("pass_name", chat_eval._PASSES)
def test_judge_messages_append_the_retry_suffix(pass_name):
    messages = chat_eval.judge_messages(
        pass_name, _turns("opener", "reply"), 1,
        "no JSON object in the reply")
    assert "could not be parsed as JSON" in messages[1]["content"]
    assert "no JSON object in the reply" in messages[1]["content"]
    # The suffix names the keys of *this* pass, and rides in the user
    # message so the cached system prefix is untouched.
    assert f'"{pass_name}" (true/false)' in messages[1]["content"]
    assert "could not be parsed" not in messages[0]["content"]


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


# ------------------------------------------------------------ style mixing


def _styled(case=None, greeting=None, greeting_seen=False, farewell=None,
            farewell_seen=False):
    return {"case": case or chat_eval.PLAIN, "greeting": greeting,
            "greeting_seen": greeting_seen, "farewell": farewell,
            "farewell_seen": farewell_seen}


def test_apply_case_style_is_the_corpus_rewrite():
    lower = chat_eval.LOWER
    assert chat_eval.apply_case_style("I am Good.", lower) == "i am good"
    # A full stop *inside* the utterance is not a closing one.
    assert chat_eval.apply_case_style("It is late. Bye!", lower) == (
        "it is late. bye!")
    # "!" and "?" carry a tone a full stop does not, and an ellipsis is
    # not a full stop -- all three survive.
    assert chat_eval.apply_case_style("Hi!", lower) == "hi!"
    assert chat_eval.apply_case_style("Really?", lower) == "really?"
    assert chat_eval.apply_case_style("Well...", lower) == "well..."
    # Only one stop comes off, and `plain` is the identity.
    assert chat_eval.apply_case_style("Yes. No.", lower) == "yes. no"
    assert chat_eval.apply_case_style("I am Good.", chat_eval.PLAIN) == (
        "I am Good.")


def test_assign_style_is_reproducible_per_seed_index():
    # Keyed on the seed index alone: the same seed carries the same
    # style across temperatures, students and reruns.
    assert chat_eval.assign_style(7) == chat_eval.assign_style(7)
    mine = [chat_eval.assign_style(i) for i in range(50)]
    other = [chat_eval.assign_style(i, seed=99) for i in range(50)]
    assert mine != other


def test_assign_style_hits_its_shares_and_keeps_them_orthogonal():
    styles = [chat_eval.assign_style(i) for i in range(4000)]

    def share(test):
        return sum(bool(test(s)) for s in styles) / len(styles)

    assert abs(share(lambda s: s["case"] == chat_eval.LOWER) - 0.30) < 0.03
    greet = share(lambda s: s["greeting"])
    assert abs(greet - 0.40) < 0.03
    assert abs(share(lambda s: s["farewell"]) - 0.15) < 0.03
    # Independent draws from separate streams: the greeting share
    # inside the lower-case slice is the overall one.
    lows = [s for s in styles if s["case"] == chat_eval.LOWER]
    assert abs(sum(bool(s["greeting"]) for s in lows) / len(lows)
               - greet) < 0.05


def test_assign_style_marks_the_unseen_forms():
    styles = [chat_eval.assign_style(i) for i in range(2000)]
    greetings = {s["greeting"] for s in styles if s["greeting"]}
    farewells = {s["farewell"] for s in styles if s["farewell"]}
    # Both blocks of both tables are reachable.
    assert {"Hi!", "Hiya!"} <= greetings
    assert {"Bye!", "Catch you later!"} <= farewells
    for style in styles:
        assert style["greeting_seen"] == (
            style["greeting"] in chat_eval._TRAINED_GREETINGS)
        assert style["farewell_seen"] == (
            style["farewell"] in chat_eval._TRAINED_FAREWELLS)
    unseen = [s for s in styles if s["greeting"] == "Hiya!"]
    assert unseen and not any(s["greeting_seen"] for s in unseen)


def test_forced_openers_put_the_greeting_before_the_seed_opener():
    seed = _seed(opener="What is your favourite colour?")
    assert chat_eval.forced_openers(seed, _styled()) == [seed["opener"]]
    assert chat_eval.forced_openers(seed, _styled(greeting="Hi!")) == [
        "Hi!", seed["opener"]]


def test_style_tag_is_readable_in_the_progress_line():
    assert chat_eval.style_tag(None) == "-"
    assert chat_eval.style_tag(_styled()) == "plain"
    assert chat_eval.style_tag(
        _styled(case=chat_eval.LOWER, greeting="Hi!", farewell="Bye!")) == (
            "lower+greet+bye")


def test_answers_in_kind_uses_the_probes_accepted_sets():
    greeting, farewell = chat_eval.GREETING, chat_eval.FAREWELL
    assert chat_eval.answers_in_kind(greeting, "Hi!")
    assert chat_eval.answers_in_kind(greeting, "  hello  ")
    assert chat_eval.answers_in_kind(greeting, "Good morning.")
    assert not chat_eval.answers_in_kind(greeting, "I am good.")
    assert chat_eval.answers_in_kind(farewell, "See you later!")
    assert chat_eval.answers_in_kind(farewell, "you too")
    assert not chat_eval.answers_in_kind(farewell, "Hi!")


def test_mirrors_lower_is_no_capitals_and_no_closing_stop():
    assert chat_eval.mirrors_lower("hi")
    assert chat_eval.mirrors_lower("bye!")
    assert chat_eval.mirrors_lower("i am good")
    assert not chat_eval.mirrors_lower("i am good.")
    assert not chat_eval.mirrors_lower("Hi")
    assert not chat_eval.mirrors_lower("   ")


def _generate_args(argv=()):
    return chat_eval.build_parser().parse_args(
        ["generate", "--student", "m.json", "--examiner", "e.gguf", *argv])


def test_style_mix_is_on_by_default_and_can_be_switched_off():
    args = _generate_args()
    assert args.style_mix is True
    assert args.style_seed == chat_eval._DEFAULT_STYLE_SEED
    assert _generate_args(["--no-style-mix"]).style_mix is False


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


def test_parse_plain_json_per_pass():
    a = chat_eval.parse_judge_output('{"comment": "fine", "a": true}', ("a",))
    assert a["a"] is True and a["comment"] == "fine"
    b = chat_eval.parse_judge_output('{"comment": "x", "b": false}', ("b",))
    assert b["b"] is False
    c = chat_eval.parse_judge_output(
        '{"comment": "y", "c": false, "user_problem": null}', ("c",))
    assert c["c"] is False
    assert c["examiner_defect"] is None


def test_parse_ignores_keys_another_pass_owns():
    # A judge that volunteers all three still gets graded on the one
    # key this pass asked for.
    grade = chat_eval.parse_judge_output(
        '{"comment": "x", "a": true, "b": false, "c": false}', ("a",))
    assert grade["a"] is True
    assert "b" not in grade and "c" not in grade


def test_parse_strips_markdown_fences():
    grade = chat_eval.parse_judge_output(
        '```json\n{"c": true, "user_problem": null, "comment": ""}\n```',
        ("c",))
    assert grade["c"] is True


def test_parse_strips_a_leaked_thinking_block():
    grade = chat_eval.parse_judge_output(
        '<think>hmm, the reply is fine</think>\n'
        '{"b": true, "comment": "ok"}', ("b",))
    assert grade["b"] is True


def test_parse_tolerates_prose_around_the_object():
    grade = chat_eval.parse_judge_output(
        'Here is my grade:\n{"a": false, "comment": "nonsense"}\n'
        'Hope that helps!', ("a",))
    assert grade["a"] is False
    assert grade["examiner_defect"] is None


def test_parse_accepts_yes_no_strings():
    assert chat_eval.parse_judge_output(
        '{"a": "yes", "comment": "x"}', ("a",))["a"] is True
    assert chat_eval.parse_judge_output(
        '{"b": "no", "comment": "x"}', ("b",))["b"] is False


@pytest.mark.parametrize("value", ['"  "', '"null"', '"None."', '"n/a"',
                                   'null'])
def test_parse_normalizes_a_non_defect_to_none(value):
    # Judges write the string "null" about as often as JSON null, and
    # an untouched one would report a defect on every single answer.
    grade = chat_eval.parse_judge_output(
        '{"c": true, "examiner_defect": ' + value + "}", ("c",))
    assert grade["examiner_defect"] is None


def test_parse_keeps_a_real_defect_under_either_key():
    # The prompt asks for `user_problem`; `examiner_defect` (the record
    # field name) stays accepted as an alias.
    for key in ("user_problem", "examiner_defect"):
        grade = chat_eval.parse_judge_output(
            f'{{"c": true, "{key}": "User repeated its question."}}', ("c",))
        assert grade["examiner_defect"] == "User repeated its question."


def test_parse_rejects_missing_and_unparseable():
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('{"comment": "no verdict"}', ("a",))
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output("I would say the reply is fine.", ("a",))
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('{"a": "maybe"}', ("a",))
    with pytest.raises(ValueError):
        chat_eval.parse_judge_output('["a", "b"]', ("a",))


def test_c_without_b_counts_a_pass_disagreement():
    assert chat_eval.is_c_without_b({"b": False, "c_raw": True})
    assert not chat_eval.is_c_without_b({"b": True, "c_raw": True})
    assert not chat_eval.is_c_without_b({"b": False, "c_raw": False})
    # A judge error on b is unknown, not a disagreement.
    assert not chat_eval.is_c_without_b({"b": None, "c_raw": True})


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
        "student", "evals/phrases-s3-top20.json", 0, 20)
    assert seat["kind"] == "phrases"
    assert seat["phrases_path"] == "evals/phrases-s3-top20.json"
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


# ------------------------------------------- reference student (ELIZA)


def _eliza_student(seed=0):
    return chat_eval.ElizaStudent(
        chat_eval.eliza_participant("student", seed), seed)


def _said(*texts):
    """Turns ending on an examiner utterance, which is what a student
    is asked to answer."""
    return _turns(*texts)


def test_tidy_eliza_is_spacing_only():
    # The port joins its output words with spaces; " ?" and " ." are
    # the joiner's, not the script's.
    assert chat_eval.tidy_eliza("Your  job ?") == "Your job?"
    assert chat_eval.tidy_eliza(
        "Earlier you said your job .") == "Earlier you said your job."
    # A rule appending "?" to a phrase that already ended in one.
    assert chat_eval.tidy_eliza("Oh, I like computers??") == \
        "Oh, I like computers?"
    # No word is touched.
    assert chat_eval.tidy_eliza("I don't know, really!") == \
        "I don't know, really!"


def test_eliza_student_reflects_the_users_own_words():
    student = _eliza_student()
    text, tokens, elapsed = student.reply(
        None, _said("I am sad about my job."), None)
    # Deterministic: the matcher has no randomness, and the reply is
    # built out of the user's own words with the pronouns flipped.
    assert text == "Your job?"
    # No server behind this seat.
    assert tokens == 0 and elapsed >= 0.0


def test_eliza_student_reads_the_last_examiner_turn_not_the_last_turn():
    student = _eliza_student()
    text, _, _ = student.reply(
        None, _said("Hello.", "irrelevant", "My mother is worried."), None)
    assert text == "Tell me more about your family."


def test_eliza_student_cycles_its_reassembly_rules_within_a_session():
    student = _eliza_student()
    turns = _said("I am sad about my job.")
    replies = [student.reply(None, turns, None)[0] for _ in range(3)]
    # Same input, three different answers: each decomposition rule
    # walks its reassembly list in order. That state is per session.
    assert len(set(replies)) == 3
    assert replies[0] == "Your job?"


def test_eliza_student_reset_starts_a_new_session():
    student = _eliza_student()
    turns = _said("I am sad about my job.")
    first = student.reply(None, turns, None)[0]
    student.reply(None, turns, None)
    student.reset(0)
    # The rule cursor and the memory stack are gone with the session.
    assert student.reply(None, turns, None)[0] == first
    # The seed index only picks the stream; the matching is the same.
    student.reset(5)
    assert student.reply(None, turns, None)[0] == first


def test_eliza_student_answers_a_quit_word_with_the_signoff():
    # `respond` returns None on a bare quit word, but the eval always
    # runs its 10 turns, so the seat must still say something.
    student = _eliza_student()
    assert student.reply(None, _said("Bye"), None)[0] == \
        "Goodbye. Thank you for talking to me."


def test_eliza_participant_records_the_provenance():
    seat = chat_eval.eliza_participant("student", 3)
    assert seat["kind"] == "eliza"
    assert seat["model"] == "eliza"
    assert seat["script"] == "doctor.txt"
    assert seat["eliza_seed"] == 3
    assert "wadetb/eliza" in seat["source"]
    # No sampling params: neither exists for this seat.
    assert "temperature" not in seat and "manifest" not in seat


def test_every_student_seat_takes_a_reset():
    # `run_dialog` calls it unconditionally, once per dialog.
    for student in (chat_eval.UrlStudent("http://x", {}),
                    _phrase_student([("a", 1)]),
                    _eliza_student()):
        student.reset(0)


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


class _ScriptedSession:
    """Answers by looking the reply up, not by call order.

    What the concurrency tests need: with N threads in flight the
    request order is not defined, so the canned answer has to be keyed
    on the request itself.
    """

    def __init__(self, replies):
        self.replies = replies
        self.bodies = []
        self.lock = threading.Lock()

    def post(self, url, json=None, timeout=None):
        with self.lock:
            self.bodies.append(json)
        text = json["messages"][-1]["content"]
        content = ""
        for needle, reply in self.replies.items():
            if needle in text:
                content = reply
                break
        return _FakeResponse({
            "choices": [{"message": {"content": content}}],
            "usage": {"completion_tokens": 7},
        })


def _judge_seat():
    return {"model": "judge.gguf", "temperature": 0.0, "max_tokens": 400,
            "extra": chat_eval._JUDGE_EXTRA}


# ------------------------------------------- dialog shape (fake examiner)


def _examiner_seat():
    return {"name": "examiner", "model": "e.gguf", "system_prompt": "SYS",
            "temperature": 0.7, "max_tokens": 96,
            "extra": chat_eval._NO_THINKING}


_OPENER = "What is your favourite colour?"


def _run_dialog(style, examiner_lines=None, answer="I am good.",
                n_turns=10):
    """One dialog against a canned examiner and a one-phrase student."""
    session = _FakeSession(
        examiner_lines or [f"Examiner line {i}." for i in range(1, 9)])
    student = _phrase_student([(answer, 1)])
    turns = chat_eval.run_dialog(
        session, _seed(idx=3, opener=_OPENER), _examiner_seat(), "http://x",
        student, n_turns, (1, 1), style)
    return turns, session


def _sides(turns, side):
    return [t["text"] for t in turns if t["side"] == side]


def test_run_dialog_without_style_is_the_old_shape():
    turns, session = _run_dialog(None)
    assert len(turns) == 10
    assert turns[0]["text"] == _OPENER and turns[0]["forced"] is True
    assert all(t["forced"] is False for t in turns[1:])
    assert set(turns[0]) == {"side", "text", "forced", "tokens", "elapsed_s"}
    # Four generated examiner turns, five student answers.
    assert len(session.bodies) == 4
    assert len(chat_eval.student_positions(turns)) == 5


def test_run_dialog_forces_the_greeting_first_and_the_opener_second():
    turns, session = _run_dialog(_styled(greeting="Hi there!",
                                         greeting_seen=True))
    assert len(turns) == 10
    assert (turns[0]["text"], turns[0]["forced"]) == ("Hi there!", True)
    assert (turns[2]["text"], turns[2]["forced"]) == (_OPENER, True)
    # The student still answers five times; the first answer answers the
    # greeting, and one generated examiner turn paid for it.
    assert len(chat_eval.student_positions(turns)) == 5
    assert turns[1]["side"] == chat_eval._STUDENT
    assert len(session.bodies) == 3


def test_run_dialog_forces_the_farewell_as_the_last_examiner_turn():
    turns, session = _run_dialog(_styled(farewell="See you later!",
                                         farewell_seen=True))
    assert len(turns) == 10
    assert turns[0]["text"] == _OPENER
    assert (turns[8]["text"], turns[8]["forced"]) == ("See you later!", True)
    # The student's final answer answers the goodbye.
    assert turns[9]["side"] == chat_eval._STUDENT
    assert len(session.bodies) == 3


def test_run_dialog_lowercases_every_examiner_utterance_only():
    style = _styled(case=chat_eval.LOWER, greeting="Hi!", greeting_seen=True,
                    farewell="Bye!", farewell_seen=True)
    turns, _ = _run_dialog(
        style, examiner_lines=["That is Nice. I agree.", "Tell me More!"])
    examiner = _sides(turns, chat_eval._EXAMINER)
    assert examiner[0] == "hi!"
    assert examiner[1] == _OPENER.lower()
    assert examiner[-1] == "bye!"
    # A mid-text stop survives; "!" and "?" survive; nothing keeps a
    # capital or a closing full stop.
    assert "that is nice. i agree" in examiner
    assert "tell me more!" in examiner
    for text in examiner:
        assert text == text.lower() and not text.endswith(".")
    # The student's own replies are never rewritten.
    assert _sides(turns, chat_eval._STUDENT) == ["I am good."] * 5


def test_grade_pass_retries_once_then_succeeds():
    session = _FakeSession([
        "I think it is fine, honestly.",
        '{"comment": "deflection", "a": true}',
    ])
    grade = chat_eval.grade_pass(
        session, "http://x", _judge_seat(), "a", _turns("opener", "reply"),
        1, (1, 1))
    assert grade["judge_error"] is None
    assert grade["a"] is True
    assert len(session.bodies) == 2
    # The retry carries the correcting suffix, and the house rules ride
    # on every request.
    assert "could not be parsed" in session.bodies[1]["messages"][1][
        "content"]
    for body in session.bodies:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["cache_prompt"] is True


def test_grade_pass_gives_up_after_the_retry():
    session = _FakeSession(["nope", "still nope"])
    grade = chat_eval.grade_pass(
        session, "http://x", _judge_seat(), "b", _turns("opener", "reply"),
        1, (1, 1))
    assert grade["b"] is None
    assert grade["judge_error"]
    assert grade["raw"] == "still nope"
    assert len(session.bodies) == 2


# ------------------------------------------------- the three-pass grade


def _pass_results(a=True, b=True, c=True, defect=None, errors=()):
    def one(key, value, comment):
        result = {key: value, "comment": comment, "raw": f"raw-{key}",
                  "judge_error": f"{key} broke" if key in errors else None,
                  "elapsed_s": 0.2}
        if key == "c":
            result["examiner_defect"] = defect
        return result
    return {"a": one("a", a, "grammar"), "b": one("b", b, "fit"),
            "c": one("c", c, "substance")}


def test_merge_gates_c_on_b():
    # c is the headline substantive rate: pass C is not asked about
    # fit, so a reply pass B rejected cannot be substantive.
    grade = chat_eval.merge_grades(_pass_results(b=False, c=True))
    assert grade["c_raw"] is True
    assert grade["c"] is False
    assert grade["c_without_b"] is True
    kept = chat_eval.merge_grades(_pass_results(b=True, c=True))
    assert (kept["c_raw"], kept["c"], kept["c_without_b"]) == (
        True, True, False)


def test_merge_keeps_every_passs_comment_and_the_defect():
    grade = chat_eval.merge_grades(_pass_results(defect="User repeated"))
    assert grade["comment_a"] == "grammar"
    assert grade["comment_b"] == "fit"
    assert grade["comment_c"] == "substance"
    assert grade["examiner_defect"] == "User repeated"
    assert set(grade["raw"]) == set(chat_eval._PASSES)
    assert grade["elapsed_s"]["a"] == 0.2


def test_merge_leaves_c_unknown_when_a_pass_failed():
    grade = chat_eval.merge_grades(
        _pass_results(a=None, b=None, errors=("b",)))
    assert grade["b"] is None
    assert grade["c"] is None
    assert grade["c_without_b"] is False
    assert "b: b broke" in grade["judge_error"]
    # An independent pass that did come back is still usable.
    partial = chat_eval.merge_grades(
        _pass_results(a=None, errors=("a",)))
    assert partial["a"] is None and partial["b"] is True


# ------------------------------------------------------- pass concurrency


def _pass_tasks(n):
    return [((None, i, 0.5, 1), _turns("q%d" % i, "answer%d" % i), 1)
            for i in range(n)]


@pytest.mark.parametrize("parallel", [1, 4])
def test_run_pass_aggregates_by_key_not_by_order(parallel):
    tasks = _pass_tasks(9)
    session = _ScriptedSession({
        f"answer{i}": '{"comment": "c", "a": %s}'
                      % ("true" if i % 2 else "false")
        for i in range(9)
    })
    results = chat_eval.run_pass(
        "a", tasks, "http://x", _judge_seat(), (1, 1), parallel,
        session_factory=lambda: session)
    assert set(results) == {t[0] for t in tasks}
    assert [results[(None, i, 0.5, 1)]["a"] for i in range(9)] == [
        bool(i % 2) for i in range(9)]
    assert len(session.bodies) == 9
    # Every request of a pass shares the same system prefix -- that is
    # what the server's prompt cache reuses.
    systems = {body["messages"][0]["content"] for body in session.bodies}
    assert len(systems) == 1


def test_run_pass_gives_each_thread_its_own_session():
    sessions = []

    def factory():
        session = _ScriptedSession({"answer": '{"comment": "c", "b": true}'})
        sessions.append(session)
        return session

    chat_eval.run_pass("b", _pass_tasks(8), "http://x", _judge_seat(),
                       (1, 1), 4, session_factory=factory)
    assert 1 <= len(sessions) <= 4
    assert sum(len(s.bodies) for s in sessions) == 8


# ------------------------------------------------------------- reporting


def _grade(a, b, c, temperature=0.5, position=1, anchor=None, seed_idx=0,
           answer="something", defect=None, error=None, c_raw=None,
           style=None):
    c_raw = c if c_raw is None else c_raw
    grade = chat_eval.merge_grades({
        "a": {"a": a, "comment": "", "raw": "", "judge_error": error,
              "elapsed_s": 0.2},
        "b": {"b": b, "comment": "", "raw": "", "judge_error": None,
              "elapsed_s": 0.1},
        "c": {"c": c_raw, "comment": "", "raw": "", "judge_error": None,
              "examiner_defect": defect, "elapsed_s": 0.3},
    })
    grade.update({
        "seed_idx": seed_idx, "anchor": anchor,
        "student_temperature": temperature, "position": position,
        "answer": answer, "style": style,
        "repetition": chat_eval.repetition_stats(answer),
        "judge_model": "judge.gguf",
    })
    return grade


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


def test_summarize_counts_pass_disagreements_and_latency():
    grades = ([_grade(True, False, False, c_raw=True)] * 3
              + [_grade(True, True, True)] * 7)
    summary = chat_eval.summarize(grades)
    assert summary["c_without_b"] == 3
    # c_raw is what pass C said; c is c_raw AND b.
    assert summary["c_raw"] == 100.0
    assert summary["rates"]["c"] == 70.0
    assert summary["latency"]["c"] == pytest.approx(0.3)


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


def test_report_lines_for_disagreements_and_pass_cost():
    grades = [_grade(True, False, False, c_raw=True)] * 2 + [
        _grade(True, True, True)] * 2
    report = chat_eval.build_report(
        "d.jsonl", "g.jsonl", grades, None, {"a": 12.0, "b": 8.5, "c": 30.0})
    assert "pass disagreements (c_raw without b): 2" in report
    assert "| a | 0.2s | 12.0s |" in report
    assert "| c | 0.3s | 30.0s |" in report


def test_report_calls_out_the_phrase_bots_pass_a_as_calibration():
    grades = [_grade(True, False, False, temperature=None)] * 9 + [
        _grade(False, False, False, temperature=None)]
    report = chat_eval.build_report("d.jsonl", "g.jsonl", grades, "phrases")
    assert "Judge calibration" in report
    assert "**pass A: 90.0%**" in report
    # A texmo student gets no such section: pass A measures the student
    # there.
    assert "Judge calibration" not in chat_eval.build_report(
        "d.jsonl", "g.jsonl", grades, None)


# --------------------------------------------------- reporting the styles


def test_report_slices_a_b_c_by_case():
    grades = ([_grade(True, True, True, seed_idx=i, style=_styled())
               for i in range(4)]
              + [_grade(False, True, False, seed_idx=10 + i,
                        style=_styled(case=chat_eval.LOWER))
                 for i in range(4)])
    report = chat_eval.build_report("d.jsonl", "g.jsonl", grades)
    assert "| plain | 4 | 100.0% | 100.0% | 100.0% |" in report
    assert "| lower | 4 | 0.0% | 100.0% | 0.0% |" in report
    # The unsliced tables are untouched by the new section.
    assert "| 0.5 | 8 | 50.0% | 100.0% | 50.0% |" in report


def test_speech_act_answers_take_the_first_and_the_last_answer():
    style = _styled(greeting="Hi!", greeting_seen=True,
                    farewell="Catch you later!")
    grades = [_grade(True, True, True, seed_idx=1, position=p, answer=a,
                     style=style)
              for p, a in ((1, "Hello!"), (2, "I like cats."), (3, "Bye!"))]
    rows = chat_eval.speech_act_answers(grades)
    assert [(r["act"], r["answer"], r["ok"], r["seen"]) for r in rows] == [
        ("greeting", "Hello!", True, True),
        ("farewell", "Bye!", True, False)]
    # A dialog with no forced act contributes nothing.
    assert chat_eval.speech_act_answers(
        [_grade(True, True, True, style=_styled())]) == []


def test_report_scores_speech_acts_by_seen_and_unseen():
    seen = _styled(greeting="Hi!", greeting_seen=True)
    unseen = _styled(case=chat_eval.LOWER, greeting="Hiya!")
    grades = [
        _grade(True, True, True, seed_idx=i, position=1, style=seen,
               answer="Hi!" if i < 3 else "I like cats.")
        for i in range(4)
    ] + [
        _grade(True, True, True, seed_idx=10 + i, position=1, style=unseen,
               answer="hi" if i < 2 else "Hi.")
        for i in range(4)
    ]
    report = chat_eval.build_report("d.jsonl", "g.jsonl", grades)
    # 3 of 4 answered in kind, and none of these dialogs was lower case.
    assert "| greeting | seen | 4 | 75.0% | 25.0% | 0 | n/a | n/a |" in report
    # All four say "hi" one way or another, but only two mirror the case.
    assert ("| greeting | unseen | 4 | 100.0% | 0.0% | 4 | 50.0% | 50.0% |"
            in report)
    assert "| greeting | all | 8 | 87.5% |" in report
    assert "'Hi!' x3" in report


def test_report_and_judging_survive_dialogs_without_a_style(tmp_path):
    # An old dialogs file: no `style` key anywhere.
    path = tmp_path / "old.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in
                  [_judge_dialog(0), _judge_dialog(1)]),
        encoding="utf-8")
    dialogs = chat_eval.limit_seeds(chat_eval.read_records(str(path)), None)
    todo = chat_eval.plan_answers(dialogs, set())
    assert len(todo) == 4
    # The grade record run_judge would write carries style None.
    grades = [
        _grade(True, True, True, seed_idx=record["seed_idx"],
               position=position, style=record.get("style"))
        for _, record, position, _ in todo
    ]
    assert all(g["style"] is None for g in grades)
    report = chat_eval.build_report("d.jsonl", "g.jsonl", grades)
    assert "No `style` on these dialogs" in report
    assert chat_eval.speech_act_answers(grades) == []
    # Everything else in the report is exactly what it always was.
    assert "| 0.5 | 4 | 100.0% | 100.0% | 100.0% |" in report


def test_student_kind_comes_from_the_first_non_anchor_dialog():
    dialogs = [
        {"seed_idx": 0, "anchor": "good", "turns": [],
         "participants": {"student": {"kind": "llama"}}},
        {"seed_idx": 0, "anchor": None, "turns": [],
         "participants": {"student": chat_eval.phrase_participant(
             "student", "p.json", 0, 3)}},
    ]
    assert chat_eval.student_kind(dialogs) == "phrases"
    assert chat_eval.student_kind([{"seed_idx": 0, "anchor": None,
                                    "turns": [], "participants": {}}]) is None


# ------------------------------------------------- judge job planning


def _judge_dialog(seed_idx, anchor=None, temperature=0.5):
    return {"seed_idx": seed_idx, "anchor": anchor,
            "student_temperature": temperature,
            "turns": _turns("q1", "a1", "q2", "a2")}


def test_limit_seeds_takes_the_first_n_distinct_seeds_and_all_anchors():
    dialogs = ([_judge_dialog(i, temperature=t)
                for t in (0.3, 0.5) for i in range(4)]
               + [_judge_dialog(0, anchor="good")])
    kept = chat_eval.limit_seeds(dialogs, 2)
    assert {r["seed_idx"] for r in kept if r["anchor"] is None} == {0, 1}
    # Both temperatures of a kept seed survive, and so does the anchor.
    assert len([r for r in kept if r["anchor"] is None]) == 4
    assert any(r["anchor"] == "good" for r in kept)
    assert chat_eval.limit_seeds(dialogs, None) == dialogs


def _judge_args(argv=()):
    return chat_eval.build_parser().parse_args(
        ["judge", "--dialogs", "d.jsonl", "--judge", "j.gguf", *argv])


def test_phase_server_conf_sizes_the_two_phases_differently():
    args = _judge_args(["--parallel-ab", "16", "--ctx-ab", "1024",
                        "--parallel-c", "4", "--ctx-size", "4096"])
    ab, ab_parallel = chat_eval.phase_server_conf(args, "ab")
    c, c_parallel = chat_eval.phase_server_conf(args, "c")
    # -np divides -c among the slots on this build, so the total is
    # scaled back up and the flags stay per-request.
    assert ab["args"] == ["-c", "16384", "-np", "16"]
    assert c["args"] == ["-c", "16384", "-np", "4"]
    assert (ab_parallel, c_parallel) == (16, 4)


def test_generate_still_gets_a_single_slot_server():
    args = chat_eval.build_parser().parse_args(
        ["generate", "--student", "m.json", "--examiner", "e.gguf"])
    assert chat_eval._server_conf(args)["args"] == ["-c", "4096"]


def test_judge_seats_cap_the_short_passes():
    args = _judge_args(["--judge-max-tokens", "400"])
    seats = chat_eval.judge_seats(args)
    assert seats["a"]["max_tokens"] == chat_eval._JUDGE_SHORT_MAX_TOKENS
    assert seats["b"]["max_tokens"] == chat_eval._JUDGE_SHORT_MAX_TOKENS
    assert seats["c"]["max_tokens"] == 400
    for seat in seats.values():
        # The per-request system prompt is not part of the seat.
        assert "system_prompt" not in seat
        assert seat["extra"]["cache_prompt"] is True


def test_plan_answers_skips_what_is_already_graded():
    dialogs = [_judge_dialog(0), _judge_dialog(1)]
    done = {chat_eval.record_key(dialogs[0]) + (1,)}
    todo = chat_eval.plan_answers(dialogs, done)
    assert [(key[1], position) for key, _, position, _ in todo] == [
        (0, 2), (1, 1), (1, 2)]
    # The turn index is what the judge grades, the position what the
    # report groups by.
    assert [index for *_, index in todo] == [3, 1, 3]
