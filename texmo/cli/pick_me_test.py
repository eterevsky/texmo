import argparse
import types

import pytest

from texmo.cli.pick_me import init_args, server_url
from texmo.cli.train import conf_from_args


def _config():
    return types.SimpleNamespace(
        TOKENS_DIR="tokens", SERVER_HOST="localhost:5000", API_KEY="")


def _parse(argv):
    parser = argparse.ArgumentParser()
    init_args(parser, _config())
    return parser.parse_args(argv)


def test_pick_me_args_build_the_conf():
    args = _parse([
        "-s", "bytes|dense.16.gelu",
        "-b", "128", "-l", "64", "--lr", "1/32", "--cosine",
        "--steps", "4096", "--runs", "4",
    ])
    assert args.runs == 4
    assert args.server == "localhost:5000"
    conf = conf_from_args(args)
    assert str(conf.model) == "bytes|dense.16.gelu"
    assert conf.batch == 128
    assert conf.length == 64
    assert conf.lr == pytest.approx(1 / 32)
    assert conf.steps == 4096
    assert conf.cosine is True
    assert conf.decay == 1.0


def test_pick_me_runs_defaults_to_three():
    args = _parse(["-s", "bytes|dense.16.gelu"])
    assert args.runs == 3
    assert args.precision == "fp32"


def test_pick_me_cosine_requires_decay_one():
    args = _parse([
        "-s", "bytes|dense.16.gelu", "--cosine", "--decay", "0.1"])
    with pytest.raises(SystemExit):
        conf_from_args(args)


def test_server_url_forms():
    assert server_url("host:5000", "/pick_me") == "http://host:5000/pick_me"
    assert (server_url("https://x.example/", "/pick_me")
            == "https://x.example/pick_me")
