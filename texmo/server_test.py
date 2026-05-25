import os

import pytest
from flask import Flask, render_template

from texmo.common import INF
from texmo.configuration import Configuration, Template
from texmo.db import DbReader, DbWriter
from texmo.model import build_model_def
from texmo.precision import Precision
from texmo.run import Run
from texmo.server import SearchServer


def _make_app():
    return Flask(
        "texmo",
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )


def _render(**overrides):
    defaults = dict(
        spec="bytes|dense.32.gelu",
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


def _make_template():
    return Template(
        spec="bytes|dense.32.gelu",
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

    model = build_model_def(
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


def _form_params(**overrides):
    """Build a minimal /update form-param dict."""
    params = {
        'spec': 'bytes|dense.32.gelu',
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
    return params


def test_search_server_update_invalid_regex_renders_error_no_crash(tmp_path):
    """A regex that the auto-finder can't resolve must produce an
    error-banner re-render, not a 500. The live template stays
    untouched so the search keeps running.
    """
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )
    original_template = server.template

    with _make_app().test_request_context():
        # Regex containing a layer that's never in the auto-finder
        # seed list and no default_spec to fall back on.
        result = server.update(_form_params(
            spec='.*norm-lrnn.*', default_spec=''))

    # Returns the rendered index page (a string), not a redirect.
    assert isinstance(result, str)
    assert "Couldn't apply template" in result
    # The user's submitted regex is preserved in the form.
    assert '.*norm-lrnn.*' in result
    # Live template unchanged.
    assert server.template is original_template


def test_search_server_update_default_spec_overrides_unresolvable_regex(tmp_path):
    """If the regex doesn't resolve but a valid default_spec is
    provided, the update succeeds and the default_spec is the seed.
    """
    path = str(tmp_path / "test.db")
    server = SearchServer(
        path, _make_template(),
        train_time=(1.0, 16.0), default_spec=None,
    )

    with _make_app().test_request_context():
        result = server.update(_form_params(
            spec='.*norm-lrnn.*',
            default_spec='bits.1+bp|gru.4-norm-lrnn.2.2',
        ))

    # Successful update returns a redirect, not an HTML string.
    assert not isinstance(result, str)
    assert server._default_spec == 'bits.1+bp|gru.4-norm-lrnn.2.2'
    assert str(server.default.model) == (
        'bits.1+bp|gru.4-norm-lrnn.2.2')


def test_search_server_add_run_does_not_overwrite_median_with_prediction(tmp_path):
    """A predicted estimate must not later replace a median estimate."""
    path = str(tmp_path / "test.db")

    # Pre-seed a 'predicted' estimate for the conf we're about to run.
    model = build_model_def(
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


