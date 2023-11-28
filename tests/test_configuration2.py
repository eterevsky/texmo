import logging
import os
from unittest import TestCase

from texmo.configuration2 import (
    Configuration2,
    Template,
    conf_neighbors,
)
from texmo.model3 import build_model
from texmo.tokens import set_tokens_dir
from texmo.common import INF


logging.disable(level=logging.ERROR)
set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


class Configuration2Test(TestCase):
    def test_conf_neighbors(self):
        template = Template(
            spec_regex=r"tokens\.(32|64)\|(dense|rec)\.\d+\.relu(-suffix\.\d+)?",
            lr=None,
            length=(128, 128),
            batch=(128, 256),
            steps=(256, 256),
            max_weights=(32, INF),
        )

        conf = Configuration2(
            model=build_model("tokens.32|dense.1.relu"),
            lr=0.25,
            length=128,
            batch=256,
            steps=256,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf.replace(model=build_model("tokens.32|dense.2.relu")),
                conf.replace(model=build_model("tokens.32|rec.1.relu")),
                conf.replace(model=build_model("tokens.32|dense.1.relu-suffix.2")),
                conf.replace(lr=0.125),
                conf.replace(lr=0.5),
                conf.replace(batch=128),
                conf.replace(model=build_model("tokens.64|dense.1.relu")),
            },
        )

    def test_dense_layer_neighbor(self):
        template = Template(
            spec_regex=r"tokens\.32\|dense\.\d+\.\w+",
            lr=(0.25, 0.25),
            length=(128, 128),
            batch=(128, 128),
            steps=(256, 256),
            max_weights=(32, INF),
        )

        conf = Configuration2(
            model=build_model("tokens.32|dense.1.relu"),
            lr=0.25,
            length=128,
            batch=128,
            steps=256,
        )

        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf.replace(model=build_model("tokens.32|dense.2.relu")),
                conf.replace(model=build_model("tokens.32|dense.1.gelu")),
                conf.replace(model=build_model("tokens.32|dense.1.tanh")),
            },
        )

    def test_conf_neighbors_suffix(self):
        template = Template(
            spec_regex=r"tokens\.(32|64)\|suffix\.\d+",
            lr=None,
            length=(128, 128),
            batch=(128, 256),
            steps=(256, 256),
            max_weights=(32, INF),
        )

        conf = Configuration2(
            model=build_model("tokens.32|suffix.2"),
            lr=0.25,
            length=128,
            batch=256,
            steps=256,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf.replace(model=build_model("tokens.32|suffix.4")),
                conf.replace(lr=0.125),
                conf.replace(lr=0.5),
                conf.replace(batch=128),
                conf.replace(model=build_model("tokens.64|suffix.2")),
            },
        )

    def test_conf_is_valid(self):
        conf = Configuration2(
            model=build_model("bytes|lstm.64-lstm.32"),
            lr=0.5,
            length=128,
            batch=256,
            steps=128,
        )
        self.assertTrue(conf.is_valid())
