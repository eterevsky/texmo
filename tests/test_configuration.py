import logging
from unittest import TestCase

from texmo.configuration import (Configuration, Template, conf_is_valid,
                                 conf_neighbors)
from texmo.model2 import build_model

logging.disable(level=logging.ERROR)


class ConfigurationTest(TestCase):
    def test_conf_neighbors(self):
        template = Template(
            spec_regex=r"(dense|rec)\.\d+\.relu(-suffix\.\d+)?",
            lr=None,
            sample_len=(0.2, 0.2),
            batch=(128, 256),
            t=(1, 1),
        )

        conf = Configuration(
            model=build_model(256, "dense.1.relu"),
            lr=0.25,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model(256, "dense.2.relu")),
                conf._replace(model=build_model(256, "rec.1.relu")),
                conf._replace(model=build_model(256, "dense.1.relu-suffix.2")),
                conf._replace(lr=0.125),
                conf._replace(lr=0.5),
                conf._replace(batch=128),
            },
        )

    def test_conf_neighbors_suffix(self):
        template = Template(
            spec_regex=r"suffix\.\d+",
            lr=None,
            sample_len=(128, 128),
            batch=(128, 256),
            t=(1, 1),
        )

        conf = Configuration(
            model=build_model(256, "suffix.2"),
            lr=0.25,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model(256, "suffix.4")),
                conf._replace(lr=0.125),
                conf._replace(lr=0.5),
                conf._replace(batch=128),
            },
        )

    def test_conf_is_valid(self):
        conf = Configuration(
            model=build_model(256, "lstm.64-lstm.32"),
            lr=0.5,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertTrue(conf_is_valid(conf))
