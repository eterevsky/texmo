import os

from flask import Flask, render_template

from texmo.precision import Precision


def _make_app():
    return Flask(
        "texmo",
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )


def _render(**overrides):
    defaults = dict(
        spec="bytes|dense.32.gelu",
        weights="32-1024",
        length="128",
        batch="32-256",
        precision={p: (p == Precision.FP32) for p in Precision},
        lr="0.001-0.1",
        decay="0.5-1",
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
        fastest=[],
        scaling_graph=None,
    )
    defaults.update(overrides)
    with _make_app().test_request_context():
        return render_template('index.html', **defaults)


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


def test_index_fastest_table_hidden_without_system():
    html = _render(
        systems=['rpi', 'whitebox'],
        selected_system=None,
        fastest=[{
            'spec': 'bytes|dense.32.gelu',
            'weights': 8448,
            'precision': 'fp32',
            'data': '32×128',
            'lr': '1/128',
            'steps': 256,
            'score': '5.123 (3)',
            'time': '1.23 s on rpi',
            'cmd': "uv run texmo.py train",
        }],
    )
    assert 'near-best loss' not in html


def test_index_fastest_table_shown_with_system():
    html = _render(
        systems=['rpi', 'whitebox'],
        selected_system='rpi',
        fastest=[{
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
        scaling_graph='dGVzdA==',  # base64("test")
    )
    assert 'near-best loss' in html
    assert 'rnn.8.tanh' in html
    # Scaling graph image
    assert 'dGVzdA==' in html
