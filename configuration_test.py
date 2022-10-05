from unittest import main, TestCase

from configuration import Configuration, conf_neighbors, conf_is_valid
from spec import ModelSpec


class ConfigurationTest(TestCase):
    def test_conf_neighbors(self):
        conf = Configuration(
            id=None,
            spec=ModelSpec.parse("dense.1.relu"),
            lr=0.2,
            sample_len=128,
            batch=256,
            regularization=0.1,
            init_scale=1.0,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, ("size", "lr"))),
            {
                conf._replace(spec=ModelSpec.parse("dense.2.relu")),
                conf._replace(lr=0.1),
                conf._replace(lr=0.5),
            },
        )

    def test_conf_is_valid(self):
        conf = Configuration(
            id=None,
            spec=ModelSpec.parse("lstm.64-lstm.32"),
            lr=0.5,
            sample_len=128,
            batch=256,
            regularization=0.1,
            init_scale=1.0,
            t=1,
        )
        self.assertTrue(conf_is_valid(conf))