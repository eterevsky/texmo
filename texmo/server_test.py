import os

from flask import Flask, render_template

from texmo.precision import Precision


def test_index_template_renders():
    """Verify index.html renders without errors."""
    app = Flask(
        "texmo",
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )

    precision = {p: (p == Precision.FP32) for p in Precision}

    with app.test_request_context():
        html = render_template(
            'index.html',
            spec="bytes|dense.32.gelu",
            weights="32-1024",
            length="128",
            batch="32-256",
            precision=precision,
            lr="0.001-0.1",
            decay="0.5-1",
            steps="256-1024",
            time="1-16",
            top=[
                {
                    'spec': 'bytes|dense.32.gelu',
                    'weights': 8448,
                    'precision': 'fp32',
                    'length': 128,
                    'batch': 32,
                    'learning': '1/128',
                    'steps': 256,
                    'score': '5.123 (3)',
                    'time': '1.23 s on test',
                },
            ],
            graph='',
        )

    assert '<title>TexMo</title>' in html
    assert 'dense.32.gelu' in html
