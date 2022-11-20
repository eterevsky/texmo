from unittest import TestCase

from texmo.configuration import (
    Configuration,
    Template,
    conf_neighbors,
    conf_is_valid,
)
from texmo.model2 import build_model


class ConfigurationTest(TestCase):
    def test_conf_neighbors(self):
        template = Template(
            spec_regex=r"(dense|rec)\.\d+\.relu(-suffix\.\d+)?",
            lr=None,
            sample_len=(0.2, 0.2),
            batch=(128, 256),
            regularization=(0.125, 0.125),
            init_scale=(1.0, 1.0),
            t=(1, 1),
        )

        conf = Configuration(
            id=None,
            model=build_model("dense.1.relu"),
            lr=0.25,
            sample_len=128,
            batch=256,
            regularization=0.125,
            init_scale=1.0,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model("dense.2.relu")),
                conf._replace(model=build_model("rec.1.relu")),
                conf._replace(model=build_model("dense.1.relu-suffix.2")),
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
            regularization=(0.125, 0.125),
            init_scale=(1.0, 1.0),
            t=(1, 1),
        )

        conf = Configuration(
            id=None,
            model=build_model("suffix.2"),
            lr=0.25,
            sample_len=128,
            batch=256,
            regularization=0.125,
            init_scale=1.0,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model("suffix.4")),
                conf._replace(lr=0.125),
                conf._replace(lr=0.5),
                conf._replace(batch=128),
            },
        )

    def test_conf_is_valid(self):
        conf = Configuration(
            id=None,
            model=build_model("lstm.64-lstm.32"),
            lr=0.5,
            sample_len=128,
            batch=256,
            regularization=0.125,
            init_scale=1.0,
            t=1,
        )
        self.assertTrue(conf_is_valid(conf))
