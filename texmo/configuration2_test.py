import logging
import os
from unittest import TestCase

from texmo.configuration2 import (
    Configuration2,
    Optimizer,
    Precision,
    Template,
    conf_neighbors,
)
from texmo.model3 import build_model
from texmo.tokens import set_tokens_dir
from texmo.common import INF


set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


def test_conf_neighbors():
    template = Template(
        spec=r"tokens\.(32|64)\.cw\.bh\|(dense|rec)\.\d+\.relu(-suffix\.\d+)?",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=(Precision.FP32,),
        optimizer=(Optimizer.ADAM,),
        decay=None,
    )

    conf = Configuration2(
        model=build_model("tokens.32.cw.bh|dense.1.relu"),
        lr=0.25,
        length=128,
        batch=256,
        steps=256,
        precision='fp32',
        optimizer='adam',
        decay=1/32,
    )
    assert set(conf_neighbors(conf, template)) == {
            conf.replace(model=build_model("tokens.32.cw.bh|dense.2.relu")),
            conf.replace(model=build_model("tokens.32.cw.bh|rec.1.relu")),
            conf.replace(model=build_model("tokens.32.cw.bh|dense.1.relu-suffix.2")),
            conf.replace(lr=0.125),
            conf.replace(lr=0.5),
            conf.replace(decay=1/16),
            conf.replace(decay=1/64),
            conf.replace(batch=128),
            conf.replace(model=build_model("tokens.64.cw.bh|dense.1.relu")),
        }

def test_dense_layer_neighbor():
    template = Template(
        spec=r"tokens\.32\.cw\.bh\|dense\.\d+\.\w+",
        lr=(0.25, 0.25),
        length=(128, 128),
        batch=(128, 128),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=(Precision.FP32,),
        optimizer='adam,fromage',
        decay=1,
    )

    conf = Configuration2(
        model=build_model("tokens.32.cw.bh|dense.1.relu"),
        lr=0.25,
        length=128,
        batch=128,
        steps=256,
        precision='fp32',
        optimizer=Optimizer.FROMAGE,
        decay=1,
    )


    assert set(conf_neighbors(conf, template)) == {
            conf.replace(model=build_model("tokens.32.cw.bh|dense.2.relu")),
            conf.replace(model=build_model("tokens.32.cw.bh|dense.1.gelu")),
            conf.replace(model=build_model("tokens.32.cw.bh|dense.1.tanh")),
        }

def test_conf_neighbors_suffix():
    template = Template(
        spec=r"tokens\.(32|64)\.cw\.bh\|suffix\.\d+",
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=(Precision.FP32,),
        optimizer=(Optimizer.ADAM,),
        decay=(0, 1),
    )

    conf = Configuration2(
        model=build_model("tokens.32.cw.bh|suffix.2"),
        lr=0.25,
        length=128,
        batch=256,
        steps=256,
        precision='fp32',
        optimizer=Optimizer.ADAM,
        decay=1,
    )
    assert set(conf_neighbors(conf, template)) == {
            conf.replace(model=build_model("tokens.32.cw.bh|suffix.4")),
            conf.replace(lr=0.125),
            conf.replace(lr=0.5),
            conf.replace(batch=128),
            conf.replace(model=build_model("tokens.64.cw.bh|suffix.2")),
            conf.replace(decay=1/2),
        }

def test_conf_neighbors_suffix_exact():
    template = Template(
        spec='tokens.32.cw.bh|suffix.2',
        lr=None,
        length=(128, 128),
        batch=(128, 256),
        steps=(256, 256),
        max_weights=(32, INF),
        precision=(Precision.FP32,),
        optimizer=(Optimizer.ADAM,),
        decay=(0, 1),
    )

    conf = Configuration2(
        model=build_model("tokens.32.cw.bh|suffix.2"),
        lr=0.25,
        length=128,
        batch=256,
        steps=256,
        precision='fp32',
        optimizer=Optimizer.ADAM,
        decay=1,
    )
    assert set(conf_neighbors(conf, template)) == {
            conf.replace(lr=0.125),
            conf.replace(lr=0.5),
            conf.replace(batch=128),
            conf.replace(decay=1/2),
        }

def test_conf_is_valid():
    conf = Configuration2(
        model=build_model("tokens.256.raw.b8|lstm.64-lstm.32"),
        lr=0.5,
        length=128,
        batch=256,
        steps=128,
        precision='fp32',
        optimizer='fromage',
        decay=1/32,
    )
    assert conf.is_valid()
