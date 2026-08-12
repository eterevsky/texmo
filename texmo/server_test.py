import os
import time
from queue import Queue

import matplotlib
import pytest
from flask import Flask, render_template
from markupsafe import escape
from werkzeug.datastructures import MultiDict

# Index rendering draws the Pareto graph; never to a display.
matplotlib.use('Agg')

from texmo.common import INF
from texmo.configuration import Configuration, Template
from texmo.db import DbReader, DbWriter
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.run import Run
import texmo.search as search_mod
from texmo.server import SearchServer


def _make_app():
    return Flask(
        "texmo",
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )


_BLANK_ROW = {'name': 'main', 'regex': '', 'default_spec': '',
              'share': '100'}


def _render(**overrides):
    defaults = dict(
        default_spec="",
        weights="32-1024",
        num_layers="0-3",
        length="128",
        batch="32-256",
        precision={p: (p == Precision.FP32) for p in Precision},
        lr="0.001-0.1",
        decay_types={'none': True, 'exp': True, 'cosine': True},
        steps="256-1024",
        time="1-16",
        top=[
            {
                'spec': 'bytes|dense.32.gelu',
                'weights': 8448,
                'precision': 'fp32',
                'data': '32×128',
                'lr': '1/128',
                'steps': 256,
                'score': '5.123 (3)',
                'time': '1.23 s on test',
                'cmd': "uv run texmo.py train -s 'bytes|dense.32.gelu'",
            },
        ],
        graph='',
        systems=[],
        selected_system=None,
        entry_form=[_BLANK_ROW],
        error=None,
    )
    defaults.update(overrides)
    with _make_app().test_request_context():
        return render_template('index.html', **defaults)


def _render_fastest(**overrides):
    defaults = dict(
        confs=[],
        loss_graph=None,
        time_graph=None,
    )
    defaults.update(overrides)
    with _make_app().test_request_context():
        return render_template('fastest.html', **defaults)


def test_index_template_renders():
    """Verify index.html renders without errors."""
    html = _render()
    assert '<title>TexMo</title>' in html
    assert 'dense.32.gelu' in html


def test_index_system_dropdown_all_systems_default():
    html = _render(systems=['rpi', 'whitebox'], selected_system=None)
    # All three options present
    assert 'All systems' in html
    assert '>rpi<' in html
    assert '>whitebox<' in html
    # "All systems" should be selected
    assert '<option value="" selected>All systems</option>' in html


def test_index_system_dropdown_specific_selected():
    html = _render(systems=['rpi', 'whitebox'], selected_system='rpi')
    assert '<option value="rpi" selected>rpi</option>' in html
    # "All systems" should NOT be selected
    assert '<option value="" selected>' not in html


def test_index_no_systems():
    """With an empty systems list, only 'All systems' should appear."""
    html = _render(systems=[], selected_system=None)
    assert 'All systems' in html


def test_index_copy_command_link():
    """Rows should include a copy link with the train command."""
    html = _render()
    assert 'copy-link' in html
    # The pipe in the spec is quoted so it's shell-safe.
    assert "bytes|dense.32.gelu&#39;" in html


def test_index_default_spec_field_present():
    html = _render(default_spec="bits.1+bp|norm-lrnn.4.2")
    assert 'name="default_spec"' in html
    assert 'bits.1+bp|norm-lrnn.4.2' in html


def test_index_no_error_banner_by_default():
    html = _render(error=None)
    assert "Couldn't apply template" not in html


def test_index_renders_error_banner_when_set():
    html = _render(error="example failure")
    assert "Couldn't apply template" in html
    assert 'example failure' in html


def test_fastest_empty_renders_message():
    html = _render_fastest()
    assert '<title>TexMo fastest near-best' in html
    assert 'No confs match' in html


def test_fastest_renders_table_and_graphs():
    html = _render_fastest(
        confs=[{
            'spec': 'bytes|rnn.8.tanh',
            'weights': 80,
            'precision': 'bf16',
            'data': '16×64',
            'lr': '1/16',
            'steps': 128,
            'score': '6.123 (4)',
            'time': '230 ms on rpi',
            'cmd': "uv run texmo.py train",
        }],
        loss_graph='bG9zcw==',  # base64("loss")
        time_graph='dGltZQ==',  # base64("time")
    )
    assert 'rnn.8.tanh' in html
    assert '230 ms on rpi' in html
    assert 'bG9zcw==' in html
    assert 'dGltZQ==' in html


def _make_template(spec="bytes|dense.32.gelu"):
    return Template(
        spec=spec,
        lr=None, length=None, batch=None,
        steps=None, max_weights=(2, INF),
        precision=list(Precision),
    )


def test_search_server_add_run_writes_median_estimate(tmp_path):
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )

    model = parse_model2(
        "bytes|dense.32.gelu", precision=Precision.FP32)
    conf = Configuration(
        model=model, lr=0.1, length=128, batch=32, steps=256, decay=1.0,
    )
    run = Run(
        system="testbench", step_loss=[0.1, 0.2],
        loss=3.0, train_time=12.5,
    )

    server.add_run({"conf": conf.to_dict(), "run": run.to_dict()})
    server.join()

    with DbReader(path) as reader:
        conf_id = reader.get_conf_id(conf)
        assert conf_id is not None
        est = reader.get_time_estimate(conf_id, "testbench")
    assert est is not None
    time_s, source = est
    assert source == "median"
    assert time_s == pytest.approx(12.5)


def _form_params(entries=(), **overrides):
    """Build a minimal /update form.

    A `MultiDict`, like the real `request.form`: the sub-search table
    posts four parallel repeated fields, and `entries` -- a list of
    `(name, regex, default_spec, share)` tuples -- is spread across
    them one value per row, in row order.
    """
    params = {
        'default_spec': '',
        'weights': '2-inf',
        'num_layers': '',
        'length': '',
        'batch': '',
        'steps': '',
        'lr': '',
        'time': '',
        'fp32': 'on',
        'decay_none': 'on',
    }
    params.update(overrides)
    form = MultiDict(params)
    for name, regex, default_spec, share in entries:
        form.add('entry_name', name)
        form.add('entry_regex', regex)
        form.add('entry_default_spec', default_spec)
        form.add('entry_share', share)
    return form


def test_search_server_update_invalid_regex_renders_error_no_crash(tmp_path):
    """A sub-search regex that the auto-finder can't resolve must
    produce an error-banner re-render, not a 500. The live template
    stays untouched so the search keeps running.
    """
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    original_template = server.template
    original_templates = server.templates

    with _make_app().test_request_context():
        # Regex containing a layer that's never in the auto-finder
        # seed list, with no default spec to fall back on.
        result = server.update(_form_params(
            entries=[('a', '.*norm-lrnn.*', '', '100')]))

    # Returns the rendered index page (a string), not a redirect.
    assert isinstance(result, str)
    assert "Couldn't apply template" in result
    # The user's submitted regex is preserved in the form.
    assert '.*norm-lrnn.*' in result
    # Live configuration unchanged.
    assert server.template is original_template
    assert server.templates is original_templates


def test_search_server_update_default_spec_overrides_unresolvable_regex(tmp_path):
    """If a sub-search regex doesn't resolve but the row carries a
    valid default spec, the update succeeds and that is the seed.
    """
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=[
                ('a', '.*norm-lrnn.*', 'bits.1+bp|gru.4-norm-lrnn.2.2',
                 '100'),
            ]))

        # Successful update returns a redirect, not an HTML string.
        assert not isinstance(result, str)
        entry = server.templates.by_name('a')
        assert entry.default_spec == 'bits.1+bp|gru.4-norm-lrnn.2.2'
        assert str(entry.default.model) == 'bits.1+bp|gru.4-norm-lrnn.2.2'
    finally:
        server.join()


def test_search_server_add_run_does_not_overwrite_median_with_prediction(tmp_path):
    """A predicted estimate must not later replace a median estimate."""
    path = str(tmp_path / "test.db")

    # Pre-seed a 'predicted' estimate for the conf we're about to run.
    model = parse_model2(
        "bytes|dense.32.gelu", precision=Precision.FP32)
    conf = Configuration(
        model=model, lr=0.1, length=128, batch=32, steps=256, decay=1.0,
    )
    pre_writer = DbWriter(path)
    conf_id = pre_writer.find_or_add_conf(conf)
    pre_writer.upsert_time_estimates(
        [(conf_id, "testbench", 99.0, "predicted")])
    pre_writer.close()

    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )

    run = Run(
        system="testbench", step_loss=[0.1, 0.2],
        loss=3.0, train_time=4.5,
    )
    server.add_run({"conf": conf.to_dict(), "run": run.to_dict()})
    server.join()

    with DbReader(path) as reader:
        time_s, source = reader.get_time_estimate(conf_id, "testbench")
    assert source == "median"
    assert time_s == pytest.approx(4.5)




# -- graceful predictor loading (Model2 transition prep) --------------

from texmo.predict.persist import predictor_path, save_predictor
from texmo.server import _load_predictor, _probe_timing


class _FakeReader:
    """Predictors live as files next to the DB; the reader only
    contributes its path."""

    def __init__(self, path: str):
        self.path = path


def _fake_db(tmp_path) -> _FakeReader:
    return _FakeReader(str(tmp_path / 'db.sqlite'))


def test_load_predictor_none_when_absent(tmp_path):
    assert _load_predictor(
        _fake_db(tmp_path), 'loss', lambda m: None) is None


def test_load_predictor_returns_model_when_probe_ok(tmp_path):
    reader = _fake_db(tmp_path)
    save_predictor(reader.path, 'loss', {'sentinel': 42})
    loaded = _load_predictor(reader, 'loss', lambda m: None)
    assert loaded == {'sentinel': 42}


def test_load_predictor_ignores_incompatible_model(tmp_path):
    """An old model deserializes but raises on first use (e.g. a
    feature-dim change) -> ignored so the caller refits."""
    reader = _fake_db(tmp_path)
    save_predictor(reader.path, 'loss', {'old': 'layout'})

    def probe(_):
        raise RuntimeError("shape mismatch")
    assert _load_predictor(reader, 'loss', probe) is None


def test_load_predictor_ignores_deserialization_failure(tmp_path):
    reader = _fake_db(tmp_path)
    with open(predictor_path(reader.path, 'loss'), 'wb') as f:
        f.write(b'not a pickle')
    assert _load_predictor(reader, 'loss', lambda m: None) is None


def test_probe_timing_runs_only_on_matching_precision():
    conf = Configuration(
        parse_model2('bytes|dense.8.gelu', precision=Precision.FP32),
        lr=0.01, length=64, batch=16, steps=128, decay=1.0)
    calls = []

    class _M:
        def __init__(self, pairs):
            self._pairs = pairs

        def keys(self):
            return self._pairs

        def predict_batch(self, system, confs):
            calls.append(system)

    _probe_timing(_M([('s', Precision.FP32)]), conf)
    assert calls == ['s']  # matching precision -> featurized

    other = next(p for p in Precision if p != Precision.FP32)
    calls.clear()
    _probe_timing(_M([('s', other)]), conf)
    assert calls == []  # no matching pair -> no-op, no exception


def test_stale_select_answered_with_none_instantly(tmp_path):
    """A Select whose client read-timeout has expired is skipped (no
    conf produced -- it would go to a closed socket) but still gets
    exactly one None response, so queue accounting holds and live
    requests behind it are served promptly."""
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )

    with server.confs_by_system_lock:
        server.confs_by_system["ghost"] = Queue()
    server.requests_queue.put(search_mod.Select(
        system="ghost", enqueued_at=time.monotonic() - 999.0))
    # Skipped without running select_conf: answered well before a
    # real selection could complete on a cold server.
    assert server.confs_by_system["ghost"].get(timeout=30) is None

    # The thread keeps serving live requests afterwards.
    result = server.select({"system": "live"})
    assert result["system"] == "live"
    server.join()


# --- weighted template sets in the UI ---------------------------------

_TEMPLATES_JSON = (
    '[{"name": "main", "share": 60},'
    ' {"name": "rnn", "share": 40, "regex": ".*\\\\|rnn\\\\..*",'
    '  "default_spec": "bytes|rnn.8.tanh"}]'
)


def _entry_row_count(html: str) -> int:
    """Rows in the editable sub-search table (the `<template>` clone
    source sits outside `#entry-rows`, so it isn't counted)."""
    body = html.split('id="entry-rows"', 1)[1].split('</tbody>', 1)[0]
    return body.count('name="entry_name"')


def test_index_has_no_base_spec_field():
    """Specs live in the sub-search rows and nowhere else: two boxes
    for one thing is what this replaced."""
    html = _render()
    assert 'name="spec"' not in html
    # The shared bounds all stay in the main template form.
    for field in ('weights', 'num_layers', 'length', 'batch', 'steps',
                  'lr', 'time', 'default_spec'):
        assert f'name="{field}"' in html


def test_index_single_template_renders_one_blank_row():
    """No sub-searches configured: one `main` row with a blank regex --
    the unrestricted search that was already running, now visible and
    editable -- and no realized-share table for a single entry."""
    html = _render(show_entries=False, entries=[], entry_names=[],
                   entry_form=[_BLANK_ROW])
    assert '<caption>Sub-searches</caption>' not in html
    assert _entry_row_count(html) == 1
    assert 'value="main"' in html
    assert 'addEntryRow()' in html
    assert 'id="entry-row-template"' in html


def test_index_renders_entry_form_rows():
    html = _render(entry_form=[
        {'name': 'main', 'regex': '', 'default_spec': '', 'share': '60'},
        {'name': 'rnn', 'regex': r'.*\|rnn\..*',
         'default_spec': 'bytes|rnn.8.tanh', 'share': '40'},
    ])
    assert _entry_row_count(html) == 2
    for field in ('entry_name', 'entry_regex', 'entry_default_spec',
                  'entry_share'):
        assert f'name="{field}"' in html
    assert 'value="main"' in html and 'value="rnn"' in html
    assert 'value="60"' in html and 'value="40"' in html
    assert 'bytes|rnn.8.tanh' in html
    assert 'removeEntryRow(this)' in html
    # The normalization hint the shares get recomputed into.
    assert 'updateShares()' in html
    assert 'normalized' in html


def test_index_stacks_the_two_spec_inputs_in_one_column():
    """Real regexes and specs run 40-80 characters, so the two share a
    wide column, one full-width line each, instead of sitting in narrow
    side-by-side cells."""
    html = _render(entry_form=[_BLANK_ROW])
    row = html.split('id="entry-rows"', 1)[1].split('</tbody>', 1)[0]
    cell = row.split('class="c-specs"', 1)[1].split('</td>', 1)[0]
    # Both inputs in the SAME cell, each inside its own label.
    assert cell.count('name="entry_regex"') == 1
    assert cell.count('name="entry_default_spec"') == 1
    assert cell.count('<label>') == 2
    assert 'input.c-regex,' in html and 'width: 100%;' in html


def test_index_renders_subsearch_table_and_filter():
    html = _render(
        show_entries=True,
        entry_names=['main', 'rnn'],
        selected_entry='rnn',
        entries=[
            {'name': 'main', 'regex': '(unrestricted)', 'share': '60%',
             'realized': '68%', 'selects': 34,
             'default': 'bits.1+bp|'},
            {'name': 'rnn', 'regex': '.*rnn.*', 'share': '40%',
             'realized': '32%', 'selects': 16,
             'default': 'bytes|rnn.8.tanh'},
        ],
    )
    assert '<caption>Sub-searches</caption>' in html
    # Nominal vs realized, side by side, plus the entry's default conf.
    assert '60%' in html and '68%' in html
    # The raw select count -- the sample size behind the realized %.
    assert '<th>Selects</th>' in html
    assert '<td>34</td>' in html and '<td>16</td>' in html
    assert 'bytes|rnn.8.tanh' in html
    # Per-entry frontier filter, with the current selection kept, side
    # by side with the system filter in one row of controls.
    assert '<option value="rnn" selected>rnn</option>' in html
    assert 'class="filters"' in html
    assert 'form.filters label' in html


def _templates_file(tmp_path, text=_TEMPLATES_JSON) -> str:
    path = tmp_path / 'templates.json'
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_search_server_cli_spec_seeds_the_first_row(tmp_path):
    """`--spec` still works, but as the first entry's regex: the base
    template the entries share carries no spec of its own."""
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template("bytes|dense.32.gelu"),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        assert server.templates.is_single
        entry = server.templates.entries[0]
        assert entry.name == 'main'
        assert entry.spec == 'bytes|dense.32.gelu'
        assert entry.template.spec == 'bytes|dense.32.gelu'
        assert entry.default is server.default
        # ...and the base is unrestricted, so nothing filters twice.
        assert server.template.spec_pattern is None
        assert server.templates.rows() == [{
            'name': 'main', 'regex': 'bytes|dense.32.gelu',
            'default_spec': '', 'share': '100'}]
    finally:
        server.join()


def test_search_server_without_spec_is_one_unrestricted_entry(tmp_path):
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        entry = server.templates.entries[0]
        assert entry.spec is None
        assert entry.template is server.template
        assert entry.default is server.default
    finally:
        server.join()


def test_search_server_loads_a_template_set(tmp_path):
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
        templates_json=_TEMPLATES_JSON,
    )
    try:
        assert server.templates.names == ['main', 'rnn']
        rnn = server.templates.by_name('rnn')
        assert str(rnn.default.model) == 'bytes|rnn.8.tanh'
        # Entries share every bound but the spec filter.
        assert rnn.template.max_weights.max == server.template.max_weights.max
        # The search thread got the same set.
        for _ in range(200):
            if server.search_thread.search is not None:
                break
            time.sleep(0.01)
        assert server.search_thread.search.templates is server.templates
    finally:
        server.join()


_ROWS = [
    ('main', '', '', '60'),
    ('rnn', r'.*\|rnn\..*', 'bytes|rnn.8.tanh', '40'),
]


def test_search_server_update_installs_a_template_set(tmp_path):
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=_ROWS))
        assert not isinstance(result, str)   # redirect == accepted
        assert server.templates.names == ['main', 'rnn']
        assert str(server.templates.by_name('rnn').default.model) == (
            'bytes|rnn.8.tanh')
        # N rows in -> the same N rows rendered back, values intact.
        with _make_app().test_request_context():
            html = server.index()
        assert _entry_row_count(html) == 2
        assert '<caption>Sub-searches</caption>' in html
        assert 'bytes|rnn.8.tanh' in html
        assert 'value="60"' in html and 'value="40"' in html
    finally:
        server.join()


def test_search_server_update_ignores_all_blank_rows(tmp_path):
    """A stray row from the Add button (or the whole table left empty)
    is dropped, not rejected -- and an emptied table means the plain
    single unrestricted search."""
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        rows = _ROWS + [('', '', '', ''), ('  ', '', '', ' ')]
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=rows))
        assert not isinstance(result, str)
        assert server.templates.names == ['main', 'rnn']

        # Every row blank -> one unrestricted entry, rendered back as
        # a single blank `main` row (never as no rows at all).
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=[('', '', '', '')]))
            assert not isinstance(result, str)
            html = server.index()
        assert server.templates.is_single
        assert server.templates.entries[0].spec is None
        assert _entry_row_count(html) == 1
    finally:
        server.join()


def test_search_server_update_rejects_a_bad_template_set(tmp_path):
    """Every malformed set must land on the error banner and leave the
    running search on its previous configuration -- with the submitted
    values still in the fields."""
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
        templates_json=_TEMPLATES_JSON,
    )
    try:
        original = server.templates
        bad_sets = [
            # A row with any field filled must validate fully.
            [('', '', '', '60')],                       # no name
            [('a', '', '', '')],                        # no share
            [('a', '', '', 'lots')],                    # share not a number
            [('a', '', '', '-1')],                      # negative share
            [('a', '', '', '0'), ('b', '', '', '0')],   # nothing to draw
            [('a', '', '', '1'), ('a', '', '', '1')],   # duplicate names
            [('a', '.*(rnn', '', '1')],                 # uncompilable regex
            [('a', r'.*\|rnn\..*', 'bytes|dense.4.gelu', '1')],
        ]
        for rows in bad_sets:
            with _make_app().test_request_context():
                result = server.update(_form_params(entries=rows))
            assert isinstance(result, str), rows
            assert "Couldn't apply template" in result
            # Live state untouched...
            assert server.templates is original
            # ...and the rejected rows come back so the user can fix
            # them in place instead of retyping.
            assert _entry_row_count(result) == len(rows)
            for name, regex, default_spec, share in rows:
                for value in (name, regex, default_spec, share):
                    if value.strip():
                        assert f'value="{escape(value)}"' in result, value
    finally:
        server.join()


def test_search_server_index_filters_the_frontier_by_entry(tmp_path):
    """The top-confs view can be narrowed to one sub-search by applying
    its regex at query time."""
    path = str(tmp_path / "test.db")
    conf_dense = Configuration(
        model=parse_model2("bytes|dense.4.gelu", precision=Precision.FP32),
        lr=0.1, length=128, batch=32, steps=256, decay=1.0)
    conf_rnn = Configuration(
        model=parse_model2("bytes|rnn.8.tanh", precision=Precision.FP32),
        lr=0.1, length=128, batch=32, steps=256, decay=1.0)
    writer = DbWriter(path)
    for conf, loss in ((conf_dense, 0.3), (conf_rnn, 0.5)):
        for i in range(2):
            writer.add_run(conf, Run(
                system='a', step_loss=[0.1], loss=loss + 0.01 * i,
                train_time=2.0))
    writer.close()

    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
        templates_json=_TEMPLATES_JSON,
    )
    try:
        with _make_app().test_request_context():
            html_all = server.index()
            html_rnn = server.index(selected_entry='rnn')
        assert 'bytes|dense.4.gelu' in html_all
        # The rnn sub-search never sees the dense conf...
        assert 'bytes|dense.4.gelu' not in html_rnn
        # ...but its own conf is on ITS frontier even though the dense
        # one dominates globally.
        assert 'bytes|rnn.8.tanh' in html_rnn
    finally:
        server.join()


def test_search_server_single_entry_renders_one_row(tmp_path):
    """With no sub-searches configured the table still shows one row
    (blank regex = unrestricted) and no realized-share table; editing
    that row's regex is how the search gets restricted."""
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        assert server.templates.is_single
        with _make_app().test_request_context():
            html = server.index()
        assert _entry_row_count(html) == 1
        assert '<caption>Sub-searches</caption>' not in html

        with _make_app().test_request_context():
            result = server.update(_form_params(
                entries=[('main', r'bytes\|rnn\..*', '', '100')]))
            assert not isinstance(result, str)
            html = server.index()
        assert server.templates.is_single
        entry = server.templates.entries[0]
        assert entry.template.regex.pattern == r'bytes\|rnn\..*'
        # The base template the entries share stays unrestricted.
        assert server.template.spec_pattern is None
        assert _entry_row_count(html) == 1
    finally:
        server.join()
