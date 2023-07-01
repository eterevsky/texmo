import logging
from unittest import TestCase

from texmo.configuration import (Configuration, Template, conf_is_valid,
                                 conf_neighbors)
from texmo.model2 import build_model
from texmo.tokens import set_tokens_dir

logging.disable(level=logging.ERROR)


class ConfigurationTest(TestCase):
    def test_conf_neighbors(self):
        set_tokens_dir("tokens")

        template = Template(
            spec_regex=r"(dense|rec)\.\d+\.relu(-suffix\.\d+)?",
            ntokens=(32, 64),
            token_type=("all", "bits1", "bits2", "bits4"),
            token_processing=("raw", "capswords"),
            lr=None,
            sample_len=(0.2, 0.2),
            batch=(128, 256),
            t=(1, 1),
        )

        conf = Configuration(
            model=build_model(32, "dense.1.relu"),
            ntokens=32,
            token_type="bits2",
            token_processing="raw",
            lr=0.25,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model(32, "dense.2.relu")),
                conf._replace(model=build_model(32, "rec.1.relu")),
                conf._replace(model=build_model(32, "dense.1.relu-suffix.2")),
                conf._replace(lr=0.125),
                conf._replace(lr=0.5),
                conf._replace(batch=128),
                conf._replace(ntokens=64, model=build_model(64, "dense.1.relu")),
                conf._replace(token_type="bits1"),
                conf._replace(token_type="bits4"),
                conf._replace(token_processing="capswords"),
            },
        )

    def test_conf_neighbors_suffix(self):
        set_tokens_dir("tokens")

        template = Template(
            spec_regex=r"suffix\.\d+",
            ntokens=(32, 64),
            token_type=("all", "bits1", "bits2", "bits4"),
            token_processing=("raw", "capswords"),
            lr=None,
            sample_len=(128, 128),
            batch=(128, 256),
            t=(1, 1),
        )

        conf = Configuration(
            model=build_model(32, "suffix.2"),
            ntokens=32,
            token_type="bits2",
            token_processing="raw",
            lr=0.25,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertEqual(
            set(conf_neighbors(conf, template)),
            {
                conf._replace(model=build_model(32, "suffix.4")),
                conf._replace(lr=0.125),
                conf._replace(lr=0.5),
                conf._replace(batch=128),
                conf._replace(ntokens=64, model=build_model(64, "suffix.2")),
                conf._replace(token_type="bits1"),
                conf._replace(token_type="bits4"),
                conf._replace(token_processing="capswords"),
            },
        )

    def test_conf_is_valid(self):
        conf = Configuration(
            model=build_model(32, "lstm.64-lstm.32"),
            ntokens=256,
            token_type="all",
            token_processing="raw",
            lr=0.5,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertTrue(conf_is_valid(conf))

    def test_conf_is_invalid1(self):
        conf = Configuration(
            model=build_model(256, "lstm.64-lstm.32"),
            ntokens=32,
            token_type="all",
            token_processing="raw",
            lr=0.5,
            sample_len=128,
            batch=256,
            t=1,
        )
        self.assertTrue(conf_is_valid(conf))

    def test_conf_is_invalid1(self):
        conf = Configuration(
            model=build_model(32, "lstm.64-lstm.32"),
            ntokens=32,
            token_type="all",
            token_processing="raw",
            lr=0.5,
            sample_len=128,
            batch=256,
            t=1,
        )
        # Can't have a tokenizer with 32 tokens and type "all"
        self.assertFalse(conf_is_valid(conf))
