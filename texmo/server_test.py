import csv
import gzip
import io
import json
import os
import pickle
import threading
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
from texmo.configuration import MAIN_ENTRY, Configuration, Template
from texmo import named_regexes
from texmo.db import DbReader, DbWriter
from texmo.spec_parser import parse_model2
from texmo.precision import Precision
from texmo.run import Run
import texmo.search as search_mod
import texmo.server as server_mod
from texmo.predict import loss_tree
from texmo.server import SearchServer


def _make_app():
    return Flask(
        "texmo",
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )


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
        entries=[_entry_display_row()],
        preset_names=[],
        reserved_regex_name=named_regexes.RESERVED_NAME,
        presets_path='named_regexes.json',
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


def test_search_server_add_run_writes_median_estimate(
        tmp_path, monkeypatch):
    # Stub the async startup LossRefit: if it loses the race with
    # this test's add_run, it sees a labeled run and does the real
    # 8000-step fit with join() waiting (see the frontier-filter
    # test).
    monkeypatch.setattr(loss_tree, "train_loss_model",
                        lambda reader: None)
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


def _form_params(entries=(), seed_rows=(), **overrides):
    """Build a minimal /update form.

    A `MultiDict`, like the real `request.form`: the sub-search table
    posts two parallel repeated fields, and `entries` -- a list of
    `(preset, share)` tuples -- is spread across them one value per
    row, in row order. The regex and default spec are no longer
    posted at all; the server resolves them from the preset file.

    `seed_rows` is what a browser posts for the Seed column: the row
    indices whose checkbox is ticked, and nothing at all for the rest
    (which is exactly why it can't be a positional field).
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
    for preset, share in entries:
        form.add('entry_preset', preset)
        form.add('entry_share', share)
    for i in seed_rows:
        form.add('entry_seed', str(i))
    return form


def _write_presets(tmp_path, monkeypatch, mapping):
    """Point `named_regexes.PATH` at a tmp preset file holding
    `mapping`. Never touches the real one at the repo root."""
    store = tmp_path / 'named_regexes.json'
    store.write_text(json.dumps(mapping), encoding='utf-8')
    monkeypatch.setattr(named_regexes, 'PATH', str(store))
    return store


def test_search_server_update_unresolvable_preset_regex_no_crash(
        tmp_path, monkeypatch):
    """A preset whose regex the auto-finder can't resolve must produce
    an error-banner re-render, not a 500. The live template stays
    untouched so the search keeps running.
    """
    _write_presets(tmp_path, monkeypatch, {
        # A layer that's never in the auto-finder seed list, with no
        # default spec to fall back on.
        'exotic': {'regex': '.*norm-lrnn.*'},
    })
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    original_template = server.template
    original_templates = server.templates

    with _make_app().test_request_context():
        result = server.update(_form_params(entries=[('exotic', '100')]))

    # Returns the rendered index page (a string), not a redirect.
    assert isinstance(result, str)
    assert "Couldn't apply template" in result
    # The row that failed comes back selected, so it can be fixed.
    assert 'exotic' in result
    # Live configuration unchanged.
    assert server.template is original_template
    assert server.templates is original_templates


def test_search_server_update_preset_default_spec_seeds_the_entry(
        tmp_path, monkeypatch):
    """A preset regex that doesn't resolve on its own is fine when the
    preset carries a default spec: that is the seed."""
    _write_presets(tmp_path, monkeypatch, {
        'exotic': {'regex': '.*norm-lrnn.*',
                   'default_spec': 'bits.1+bp|gru.4-norm-lrnn.2.2'},
    })
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(
                entries=[('exotic', '100')]))

        # Successful update returns a redirect, not an HTML string.
        assert not isinstance(result, str)
        # Entry names are preset names now.
        entry = server.templates.by_name('exotic')
        assert entry.spec == '.*norm-lrnn.*'
        assert entry.default_spec == 'bits.1+bp|gru.4-norm-lrnn.2.2'
        assert str(entry.default.model) == 'bits.1+bp|gru.4-norm-lrnn.2.2'
    finally:
        server.join()


def test_update_snapshots_the_preset_so_later_edits_do_not_leak(
        tmp_path, monkeypatch):
    """Resolution happens at Update; editing the file afterwards must
    not mutate a running search."""
    store = _write_presets(tmp_path, monkeypatch, {
        'recurrent': {'regex': '.*gru.*'},
    })
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            server.update(_form_params(entries=[('recurrent', '100')]))
        assert server.templates.by_name('recurrent').spec == '.*gru.*'

        store.write_text(
            json.dumps({'recurrent': {'regex': '.*lstm.*'}}),
            encoding='utf-8')
        # Still running what was applied...
        assert server.templates.by_name('recurrent').spec == '.*gru.*'
        # ...and the row says so, flagged as drifted from the file.
        with _make_app().test_request_context():
            html = server.index()
        assert '.*gru.*' in html
        assert 'has changed since this was applied' in html

        # Adopting the new pattern takes another Update.
        with _make_app().test_request_context():
            server.update(_form_params(entries=[('recurrent', '100')]))
        assert server.templates.by_name('recurrent').spec == '.*lstm.*'
    finally:
        server.join()


def test_update_rejects_a_preset_that_left_the_file(tmp_path, monkeypatch):
    """The name is the state, so a preset deleted from the file makes
    the row unresolvable -- error banner, live template untouched."""
    _write_presets(tmp_path, monkeypatch, {'gone': {'regex': '.*gru.*'}})
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            server.update(_form_params(entries=[('gone', '100')]))
        applied = server.templates

        _write_presets(tmp_path, monkeypatch, {'other': {'regex': '.*a.*'}})
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=[('gone', '100')]))

        assert isinstance(result, str)
        assert "Couldn't apply template" in result
        assert 'gone' in result
        assert server.templates is applied
    finally:
        server.join()


def test_update_rejects_the_same_preset_twice(tmp_path, monkeypatch):
    """Two rows on one preset would be two entries with one name."""
    _write_presets(tmp_path, monkeypatch, {'a': {'regex': '.*gru.*'}})
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(
                entries=[('a', '50'), ('a', '50')]))
        assert isinstance(result, str)
        assert "Couldn't apply template" in result
        assert 'own preset' in result
    finally:
        server.join()


def test_unrestricted_row_keeps_the_main_entry_identity(
        tmp_path, monkeypatch):
    """The virtual Unrestricted row is the historical `main` entry:
    same name, so coverage keys and select counters carry over."""
    _write_presets(tmp_path, monkeypatch, {'rec': {'regex': '.*gru.*'}})
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            server.update(_form_params(entries=[
                (named_regexes.RESERVED_NAME, '70'), ('rec', '30')]))
        assert server.templates.names == [MAIN_ENTRY, 'rec']
        main = server.templates.by_name(MAIN_ENTRY)
        assert main.spec is None  # no filter
        # Renders back onto the reserved option, not a stored name.
        with _make_app().test_request_context():
            html = server.index()
        row = _entry_row_html(html, 0)
        assert f'<option selected>{named_regexes.RESERVED_NAME}' in row
    finally:
        server.join()


def test_search_server_add_run_does_not_overwrite_median_with_prediction(
        tmp_path, monkeypatch):
    """A predicted estimate must not later replace a median estimate."""
    # Stub the async startup LossRefit: if it loses the race with
    # this test's add_run, it sees a labeled run and does the real
    # 8000-step fit with join() waiting (see the frontier-filter
    # test).
    monkeypatch.setattr(loss_tree, "train_loss_model",
                        lambda reader: None)
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


def _tiny_tree_model(default_conf, run_count):
    """A fast-fitted TreeLossModel whose probe covers `default_conf`."""
    data = [(default_conf, 2.0)] * 8
    st = loss_tree.discover_simple_types(data)
    params, _ = loss_tree.fit(
        data, st, steps=20, batch_size=4, seed=0, per_type_fwd=True)
    return loss_tree.TreeLossModel(
        params=params, simple_types=st, per_type_fwd=True,
        run_count=run_count)


def test_client_refit_grant_and_submit(tmp_path, monkeypatch):
    path = str(tmp_path / "db.sqlite")
    with DbWriter(path):
        pass
    monkeypatch.setattr(server_mod, "LOSS_REFIT_EVERY", 1)
    # Stub the async startup LossRefit: if it loses the race with
    # this test's add_run, it sees a labeled run and does the real
    # 8000-step fit with join() waiting (see the frontier-filter
    # test).
    monkeypatch.setattr(loss_tree, "train_loss_model",
                        lambda reader: None)
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
        loss_refit='workers',
    )
    try:
        model = parse_model2(
            "bytes|dense.32.gelu", precision=Precision.FP32)
        conf = Configuration(
            model=model, lr=0.1, length=128, batch=32, steps=256,
            decay=1.0)
        run = Run(system="tower", step_loss=None, loss=3.0,
                  train_time=2.0)

        # Below threshold: no job.
        assert server._maybe_grant_refit("tower") is None

        server.add_run({"conf": conf.to_dict(), "run": run.to_dict()})

        # A client gets the job; fire-and-forget: no second grant
        # until another LOSS_REFIT_EVERY runs land, regardless of
        # whether the first job ever reports back.
        assert server._maybe_grant_refit("tower") is not None
        assert server._maybe_grant_refit("other") is None
        server.add_run({"conf": conf.to_dict(), "run": run.to_dict()})
        assert server._maybe_grant_refit("other") is not None

        # Training data round-trips through gzip + CSV. The runs
        # land via the async writer queue, so poll briefly.
        for _ in range(200):
            payload, n = server.training_data()
            if n == 2:
                break
            time.sleep(0.05)
        assert n == 2
        text = gzip.decompress(payload).decode()
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0][0] == "spec"
        assert len(rows) == 3 and rows[1][0] == str(model)

        # Garbage body: probe rejects.
        d = server.submit_loss_model(
            {"system": "tower"}, b"not a pickle")
        assert not d["accepted"]

        # A real model publishes; its freshness (run_count = training
        # rows counted by the worker) travels inside the pickle.
        tree = _tiny_tree_model(server.default, run_count=2)
        d = server.submit_loss_model(
            {"system": "tower", "fit_s": "1.0"}, pickle.dumps(tree))
        assert d["accepted"]
        assert server.loss_model.is_ready()
        assert server.loss_model.current().run_count == 2

        # The same (now stale) model again: rejected.
        d = server.submit_loss_model(
            {"system": "tower"}, pickle.dumps(tree))
        assert not d["accepted"] and d["reason"] == "stale run_count"
    finally:
        server.join()


def test_server_mode_grants_no_refit_jobs(tmp_path, monkeypatch):
    """In 'server' mode the cadence lives on the model thread, so
    /select never hands a job out -- but the submit path stays open
    for a model a worker fitted anyway."""
    path = str(tmp_path / "db.sqlite")
    with DbWriter(path):
        pass
    monkeypatch.setattr(server_mod, "LOSS_REFIT_EVERY", 1)
    # Stub the async startup LossRefit: if it loses the race with
    # this test's add_run, it sees a labeled run and does the real
    # 8000-step fit with join() waiting (see the frontier-filter
    # test).
    monkeypatch.setattr(loss_tree, "train_loss_model",
                        lambda reader: None)
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
        loss_refit='server',
    )
    try:
        conf = Configuration(
            model=parse_model2(
                "bytes|dense.32.gelu", precision=Precision.FP32),
            lr=0.1, length=128, batch=32, steps=256, decay=1.0)
        run = Run(system="tower", step_loss=None, loss=3.0,
                  train_time=2.0)
        server.add_run({"conf": conf.to_dict(), "run": run.to_dict()})

        # Past the cadence, but /select still answers with work (or
        # None), never with a refit job.
        result = server.select({"system": "tower", "refit": "1"})
        assert "refit_loss" not in result

        # A valid, newer submission is still accepted in this mode.
        tree = _tiny_tree_model(server.default, run_count=7)
        d = server.submit_loss_model(
            {"system": "tower"}, pickle.dumps(tree))
        assert d["accepted"]
    finally:
        server.join()


def test_server_wires_writer_panic_path(tmp_path):
    """A silently dead writer thread is the failure this guards: the
    server must hand WriterThread a real fatal callback, and that
    callback must be safe to call before `serve` binds any port."""
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        assert server.writer_thread._on_fatal is not None
        # Nothing bound yet -> logs and returns rather than raising.
        server.writer_thread._on_fatal()
    finally:
        server.join()


def _preset_names_in_select(html: str) -> list[str]:
    """Option labels of the first preset dropdown in `html`."""
    return _preset_names_in_select_for_row(html, 0)


def _entry_row_html(html: str, i: int) -> str:
    """The i-th `<tr>` of the merged sub-search table."""
    body = html.split('id="entry-rows"', 1)[1].split('</tbody>', 1)[0]
    return body.split('<tr>')[i + 1]


def _preset_names_in_select_for_row(html: str, i: int) -> list[str]:
    block = _entry_row_html(html, i).split(
        '<select', 1)[1].split('</select>', 1)[0]
    return [
        chunk.split('>', 1)[1].split('<', 1)[0].strip()
        for chunk in block.split('<option')[1:]
    ]


def _entry_display_row(**overrides):
    """One merged-table row, shaped as `SearchServer._entry_rows`
    builds them."""
    row = dict(
        preset=named_regexes.RESERVED_NAME,
        options=[],
        share='100',
        seed=False,
        seed_left=None,
        regex='',
        default_spec='',
        stale='',
        name=MAIN_ENTRY,
        share_pct='100%',
        realized='—',
        selects=0,
        default='bytes|dense.32.gelu',
    )
    row.update(overrides)
    return row


def _seed_box(html: str, i: int) -> str:
    """The i-th row's Seed checkbox tag."""
    return _entry_row_html(html, i).split(
        'class="c-seed"', 1)[1].split('>', 1)[0]


def test_index_preset_dropdown_lists_unrestricted_first(tmp_path):
    """The dropdown IS the row's spec state: 'Unrestricted' first,
    then the stored names alphabetically, posted as entry_preset."""
    html = _render(entries=[_entry_display_row(
        options=['attention', 'recurrent'])])
    assert _preset_names_in_select(html) == [
        named_regexes.RESERVED_NAME, 'attention', 'recurrent']
    assert 'name="entry_preset"' in html
    # The regex is no longer enterable anywhere in the form.
    assert 'name="entry_regex"' not in html
    assert 'name="entry_default_spec"' not in html


def test_index_preset_dropdown_with_no_presets(tmp_path):
    html = _render(entries=[_entry_display_row()])
    assert _preset_names_in_select(html) == [named_regexes.RESERVED_NAME]


def test_index_rereads_presets_on_every_render(tmp_path, monkeypatch):
    """The file is hand-edited behind the server's back, so a refresh
    has to show the edit -- nothing may cache it."""
    store = _write_presets(
        tmp_path, monkeypatch, {'before': {'regex': '.*gru.*'}})
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            first = server.index()
            store.write_text(
                '{"after": {"regex": ".*lstm.*"}}', encoding='utf-8')
            second = server.index()
    finally:
        server.join()

    assert 'before' in _preset_names_in_select(first)
    assert _preset_names_in_select(second) == [
        named_regexes.RESERVED_NAME, 'after']


def test_index_row_keeps_an_unknown_name_selectable(tmp_path, monkeypatch):
    """An entry from --templates or --spec need not correspond to a
    preset; its row still renders, still round-trips, and says why
    Update would reject it."""
    _write_presets(tmp_path, monkeypatch, {'other': {'regex': '.*lstm.*'}})
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
        templates_json=_TEMPLATES_JSON,
    )
    try:
        with _make_app().test_request_context():
            html = server.index()
    finally:
        server.join()

    # 'rnn' comes from the templates file, not from the preset file.
    assert _preset_names_in_select_for_row(html, 1) == [
        named_regexes.RESERVED_NAME, 'rnn', 'other']
    assert 'no longer in' in html


def test_index_selects_a_preset_matching_a_cli_spec(tmp_path, monkeypatch):
    """`--spec` seeds an entry carrying a raw pattern. When that
    pattern is a preset's, the row starts on the preset instead of
    stranding the user on an unknown name."""
    _write_presets(
        tmp_path, monkeypatch, {'recurrent': {'regex': '.*gru.*'}})
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(spec='.*gru.*'),
        train_time=(1.0, 16.0), default_spec='bytes|gru.4',
    )
    try:
        with _make_app().test_request_context():
            html = server.index()
    finally:
        server.join()

    row = _entry_row_html(html, 0)
    assert '<option selected>recurrent' in row
    assert 'no longer in' not in row


def _compare_selects(html: str) -> list[list[str]]:
    """Option labels of each column's spec-filter dropdown."""
    out = []
    for prefix in ('t1', 't2'):
        block = html.split(
            f'name="{prefix}_preset"', 1)[1].split('</select>', 1)[0]
        out.append([
            chunk.split('>', 1)[1].split('<', 1)[0].strip()
            for chunk in block.split('<option')[1:]
        ])
    return out


def test_compare_page_offers_one_preset_selector_per_column(
        tmp_path, monkeypatch):
    """Each column has exactly one spec control now: the preset
    dropdown. No free-text regex box, no separate entry override."""
    _write_presets(tmp_path, monkeypatch, {
        'recurrent': {'regex': '.*gru.*'},
        'attention': {'regex': '.*attn.*'},
    })
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            html = server.compare(MultiDict())
    finally:
        server.join()

    assert _compare_selects(html) == [
        [named_regexes.RESERVED_NAME, 'attention', 'recurrent'],
        [named_regexes.RESERVED_NAME, 'attention', 'recurrent'],
    ]
    # The retired controls are gone for good.
    assert 'name="t1_spec"' not in html
    assert 'name="t1_entry"' not in html
    assert 'applyPreset' not in html


def test_compare_resolves_the_selected_preset(tmp_path, monkeypatch):
    """The name picked in the dropdown is what filters that column."""
    _write_presets(tmp_path, monkeypatch, {
        'rnn': {'regex': r'.*\|rnn\..*'},
    })
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
    )
    try:
        with _make_app().test_request_context():
            # A submitted form carries its checkboxes explicitly:
            # absent means unchecked, so defaults are not inherited.
            html = server.compare(MultiDict({
                'weights': '100000',
                't1_preset': named_regexes.RESERVED_NAME,
                't2_preset': 'rnn',
                't1_fp32': 'on', 't2_fp32': 'on',
                't1_decay_none': 'on', 't2_decay_none': 'on',
            }))
    finally:
        server.join()

    # Column 1 unfiltered sees the dense conf; column 2 is rnn-only.
    table1, table2 = html.split('Template 2 &mdash;', 1)
    assert 'bytes|dense.4.gelu' in table1
    assert 'bytes|dense.4.gelu' not in table2
    assert 'bytes|rnn.8.tanh' in table2
    # The selection round-trips into the form.
    assert _compare_selects(html)[1][:2] == [
        named_regexes.RESERVED_NAME, 'rnn']
    assert '<option selected>rnn' in html


def test_compare_survives_a_preset_that_left_the_file(
        tmp_path, monkeypatch):
    """A bookmarked compare URL must outlive an edit to the preset
    file: no 500, the name still shown, and a reason given."""
    _write_presets(tmp_path, monkeypatch, {'stays': {'regex': '.*gru.*'}})
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            html = server.compare(MultiDict({
                'weights': '100000',
                't1_preset': named_regexes.RESERVED_NAME,
                't2_preset': 'vanished',
            }))
    finally:
        server.join()

    assert "Couldn't run the comparison" in html
    assert 'vanished' in html
    # Still selectable, so the URL round-trips once the file is fixed.
    assert '<option selected>vanished' in html
    assert 'Template 1 &mdash;' not in html


def test_every_server_thread_is_a_daemon(tmp_path):
    """Every thread a server starts must be a daemon.

    Cleanup for a server dropped without `join()` rests on `__del__`
    (see the next test), which plain refcounting has to reach. Pair a
    *non-daemon* thread with that and any accidental strong reference
    back to the server -- a bound-method callback is enough -- stops
    being a leak and becomes a process that never exits, since
    interpreter shutdown joins non-daemon threads before it would
    ever collect the cycle. Daemons keep that failure mode benign;
    the graceful paths (`join`, `/stop`) post `Stop` and join
    regardless, which is where durability actually comes from."""
    path = str(tmp_path / "db.sqlite")
    before = set(threading.enumerate())
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        for t in (server.writer_thread, server.search_thread,
                  server.model_thread):
            assert t.daemon, f"{type(t).__name__} is not a daemon"
        # Anything else construction started, named or not.
        started = [t for t in threading.enumerate() if t not in before]
        assert started, "constructing a server should start threads"
        assert [t for t in started if not t.daemon] == []
    finally:
        server.join()


def test_dropped_server_stops_its_threads(tmp_path):
    """A server dropped without `join()` must still stop its threads.

    `__del__` posts the three Stop messages, so this only works while
    plain refcounting can free the server -- nothing may hold it in a
    reference cycle. (A bound method handed to WriterThread as the
    fatal callback did exactly that; back when SearchThread was the
    one non-daemon thread it wedged interpreter exit outright.) No
    `gc.collect()` here on purpose: collecting the cycle by hand
    would hide the very regression this guards."""
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    threads = [
        server.writer_thread, server.model_thread, server.search_thread,
    ]
    del server

    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), f"{t.name} outlived the server that owned it"


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
    return body.count('name="entry_preset"')


def test_index_has_no_base_spec_field():
    """Specs live in the sub-search rows and nowhere else: two boxes
    for one thing is what this replaced."""
    html = _render()
    assert 'name="spec"' not in html
    # The shared bounds all stay in the main template form.
    for field in ('weights', 'num_layers', 'length', 'batch', 'steps',
                  'lr', 'time', 'default_spec'):
        assert f'name="{field}"' in html


def test_index_single_template_renders_one_unrestricted_row():
    """No sub-searches configured: one row sitting on Unrestricted --
    the search that was already running, now visible and editable."""
    html = _render(show_entries=False, entry_names=[],
                   entries=[_entry_display_row()])
    assert _entry_row_count(html) == 1
    assert f'<option selected>{named_regexes.RESERVED_NAME}' in html
    assert 'addEntryRow()' in html
    assert 'id="entry-row-template"' in html


def test_index_renders_entry_form_rows():
    html = _render(entries=[
        _entry_display_row(share='60'),
        _entry_display_row(
            preset='rnn', options=['rnn'], share='40',
            regex=r'.*\|rnn\..*', default_spec='bytes|rnn.8.tanh',
            name='rnn', share_pct='40%', realized='32%', selects=16),
    ])
    assert _entry_row_count(html) == 2
    # Two posted fields per row, and only those two.
    for field in ('entry_preset', 'entry_share'):
        assert f'name="{field}"' in html
    assert '<option selected>rnn' in html
    assert 'value="60"' in html and 'value="40"' in html
    assert 'bytes|rnn.8.tanh' in html
    assert 'removeEntryRow(this)' in html
    # The normalization hint the shares get recomputed into.
    assert 'updateShares()' in html
    assert 'normalized' in html


def test_index_shows_the_two_specs_read_only_in_one_column():
    """Real regexes and specs run 40-80 characters, so the two share a
    wide column, one full-width line each. They are display-only now:
    a row's pattern comes from the preset file, never from the form."""
    html = _render(entries=[_entry_display_row(
        preset='rnn', options=['rnn'], regex=r'.*\|rnn\..*',
        default_spec='bytes|rnn.8.tanh', name='rnn')])
    cell = _entry_row_html(html, 0).split(
        'class="c-specs"', 1)[1].split('</td>', 1)[0]
    assert cell.count('<label>') == 2
    assert '<input' not in cell
    assert cell.index('c-regex') < cell.index('class="c-spec"')
    assert 'code.c-regex,' in html and 'width: 100%;' in html


def test_index_renders_subsearch_stats_and_filter():
    """The stats that used to sit in their own table live in the edit
    row itself, so what a sub-search IS and how it is DOING read
    together."""
    html = _render(
        show_entries=True,
        entry_names=['main', 'rnn'],
        selected_entry='rnn',
        entries=[
            _entry_display_row(
                share='60', share_pct='60%', realized='68%', selects=34,
                default='bits.1+bp|'),
            _entry_display_row(
                preset='rnn', options=['rnn'], share='40',
                regex='.*rnn.*', name='rnn', share_pct='40%',
                realized='32%', selects=16,
                default='bytes|rnn.8.tanh'),
        ],
    )
    # Nominal vs realized, side by side, plus the entry's default conf.
    assert '60%' in html and '68%' in html
    # The raw select count -- the sample size behind the realized %.
    assert '34 selects' in html and '16 selects' in html
    assert 'bytes|rnn.8.tanh' in html
    # Each live row links to its own frontier view.
    assert 'href="/?entry=rnn"' in html
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
            'default_spec': '', 'share': '100', 'seed': False}]
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


# The presets the multi-entry tests below resolve against.
_PRESETS = {
    'rnn': {'regex': r'.*\|rnn\..*', 'default_spec': 'bytes|rnn.8.tanh'},
    'dense': {'regex': r'.*\|dense\..*'},
}

_ROWS = [
    ('Unrestricted', '60'),
    ('rnn', '40'),
]


def test_search_server_update_installs_a_template_set(
        tmp_path, monkeypatch):
    _write_presets(tmp_path, monkeypatch, _PRESETS)
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=_ROWS))
        assert not isinstance(result, str)   # redirect == accepted
        # The unrestricted row keeps the historical `main` identity;
        # every other entry is named for its preset.
        assert server.templates.names == [MAIN_ENTRY, 'rnn']
        assert server.templates.by_name('rnn').spec == r'.*\|rnn\..*'
        assert str(server.templates.by_name('rnn').default.model) == (
            'bytes|rnn.8.tanh')
        # N rows in -> the same N rows rendered back, values intact.
        with _make_app().test_request_context():
            html = server.index()
        assert _entry_row_count(html) == 2
        assert 'bytes|rnn.8.tanh' in html
        assert 'value="60"' in html and 'value="40"' in html
    finally:
        server.join()


def test_search_server_update_ignores_untouched_rows(
        tmp_path, monkeypatch):
    """A row straight from the Add button -- still on Unrestricted,
    with no share -- is dropped, not rejected; and a table with only
    those left means the plain single unrestricted search."""
    _write_presets(tmp_path, monkeypatch, _PRESETS)
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        rows = _ROWS + [('Unrestricted', ''), ('Unrestricted', '  ')]
        with _make_app().test_request_context():
            result = server.update(_form_params(entries=rows))
        assert not isinstance(result, str)
        assert server.templates.names == [MAIN_ENTRY, 'rnn']

        # Nothing left -> one unrestricted entry, rendered back as a
        # single Unrestricted row (never as no rows at all).
        with _make_app().test_request_context():
            result = server.update(
                _form_params(entries=[('Unrestricted', '')]))
            assert not isinstance(result, str)
            html = server.index()
        assert server.templates.is_single
        assert server.templates.entries[0].spec is None
        assert _entry_row_count(html) == 1
    finally:
        server.join()


def test_search_server_update_rejects_a_bad_template_set(
        tmp_path, monkeypatch):
    """Every malformed set must land on the error banner and leave the
    running search on its previous configuration -- with the submitted
    values still in the fields."""
    _write_presets(tmp_path, monkeypatch, _PRESETS)
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
        templates_json=_TEMPLATES_JSON,
    )
    try:
        original = server.templates
        bad_sets = [
            [('rnn', '')],                        # no share
            [('rnn', 'lots')],                    # share not a number
            [('rnn', '-1')],                      # negative share
            [('rnn', '0'), ('dense', '0')],       # nothing to draw
            [('rnn', '1'), ('rnn', '1')],         # same preset twice
            [('Unrestricted', '1'), ('Unrestricted', '1')],  # ditto
            [('vanished', '1')],                  # not in the file
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
            for preset, share in rows:
                if share.strip():
                    assert f'value="{escape(share)}"' in result, share
                assert f'<option selected>{escape(preset)}' in result
    finally:
        server.join()


# --- frontier seeding -------------------------------------------------


def test_server_shares_one_frontier_version_across_threads(tmp_path):
    """The writer thread publishes frontier moves and the search
    thread's seed queues invalidate on them. Two objects and neither
    half does anything -- with no error to show for it."""
    path = str(tmp_path / "db.sqlite")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        assert (server.writer_thread._frontier_version
                is server.frontier_version)
        for _ in range(200):
            if server.search_thread.search is not None:
                break
            time.sleep(0.01)
        assert (server.search_thread.search.frontier_version
                is server.frontier_version)
    finally:
        server.join()


def test_index_renders_the_seed_checkbox_column():
    """One checkbox per row, plus one in the Add-row template so a new
    row can be ticked before it is ever applied."""
    html = _render(entries=[
        _entry_display_row(share='60'),
        _entry_display_row(
            preset='rnn', options=['rnn'], share='40', seed=True,
            seed_left=7, regex=r'.*\|rnn\..*', name='rnn'),
    ])
    assert '>Seed</th>' in html
    assert 'name="entry_seed"' in html
    assert 'checked' not in _seed_box(html, 0)
    assert 'checked' in _seed_box(html, 1)
    # Each box carries its own row index -- see `_ENTRY_SEED_FIELD`.
    assert 'value="0"' in _seed_box(html, 0)
    assert 'value="1"' in _seed_box(html, 1)
    # Remaining queue is shown next to the entry's other live stats.
    assert '7 seeds queued' in html
    # The clone source has one too (rows + template = 3 in total).
    assert html.count('name="entry_seed"') == 3


def test_index_seed_queue_hidden_until_there_is_one():
    """An entry that has never had a seeded select shows no count --
    0 and 'not built yet' are different things."""
    html = _render(entries=[_entry_display_row(seed=True, seed_left=None)])
    assert 'seeds queued' not in html
    html = _render(entries=[_entry_display_row(seed=True, seed_left=0)])
    assert '0 seeds queued' in html


def test_entry_rows_from_form_reads_seed_by_row_index():
    """An unchecked checkbox posts nothing, so the flag can't be
    zipped by position like preset/share -- a single ticked row three
    down must not land on row 0."""
    rows = server_mod._entry_rows_from_form(_form_params(
        entries=[('a', '1'), ('b', '2'), ('c', '3')], seed_rows=[2]))
    assert [r['seed'] for r in rows] == [False, False, True]
    # Nothing ticked at all is the common case.
    rows = server_mod._entry_rows_from_form(_form_params(
        entries=[('a', '1'), ('b', '2')]))
    assert [r['seed'] for r in rows] == [False, False]


def test_update_round_trips_the_seed_checkbox(tmp_path, monkeypatch):
    """Seed is per-entry state like share: applied by Update, kept in
    the live set, rendered back onto the row."""
    _write_presets(tmp_path, monkeypatch, _PRESETS)
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(
                entries=_ROWS, seed_rows=[1]))
        assert not isinstance(result, str)
        assert server.templates.by_name(MAIN_ENTRY).seed is False
        assert server.templates.by_name('rnn').seed is True

        with _make_app().test_request_context():
            html = server.index()
        assert 'checked' not in _seed_box(html, 0)
        assert 'checked' in _seed_box(html, 1)

        # Unticking it is an ordinary Update too.
        with _make_app().test_request_context():
            assert not isinstance(
                server.update(_form_params(entries=_ROWS)), str)
        assert server.templates.by_name('rnn').seed is False
    finally:
        server.join()


def test_update_keeps_the_seed_tick_on_a_rejected_form(
        tmp_path, monkeypatch):
    """The rejection re-render exists so the user fixes what they
    submitted instead of retyping it -- the tick is part of that."""
    _write_presets(tmp_path, monkeypatch, _PRESETS)
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(spec=None),
        train_time=(1.0, 16.0), default_spec=None,
    )
    try:
        with _make_app().test_request_context():
            result = server.update(_form_params(
                entries=[('rnn', 'lots')], seed_rows=[0]))
        assert isinstance(result, str)
        assert "Couldn't apply template" in result
        assert 'checked' in _seed_box(result, 0)
    finally:
        server.join()


def test_search_server_index_filters_the_frontier_by_entry(
        tmp_path, monkeypatch):
    """The top-confs view can be narrowed to one sub-search by applying
    its regex at query time."""
    # The tmp DB is non-empty at construction, so the startup LossRefit
    # fallback would run the real 8000-step production fit (~70 s) with
    # join() waiting for it. Stub it -- this test is about the index
    # query, not the predictor.
    monkeypatch.setattr(loss_tree, "train_loss_model", lambda reader: None)
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


def test_search_server_single_entry_renders_one_row(
        tmp_path, monkeypatch):
    """With no sub-searches configured the table still shows one row
    (Unrestricted = no filter); putting that row on a preset is how
    the search gets restricted."""
    _write_presets(tmp_path, monkeypatch, {
        'rnn': {'regex': r'bytes\|rnn\..*'},
    })
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

        with _make_app().test_request_context():
            result = server.update(_form_params(entries=[('rnn', '100')]))
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
