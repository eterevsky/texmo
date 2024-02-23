import logging
from unittest import TestCase

from texmo.common import INF
from texmo.configuration2 import Configuration2, Template
from texmo.model3 import build_model
from texmo.resultdb import ResultDB
from texmo.search import Search

logging.disable(level=logging.ERROR)


class SearchTest(TestCase):
    def setUp(self):
        self._db = ResultDB()
        self._template = Template(
            spec_regex=None,
            lr=None,
            length=None,
            batch=None,
            steps=None,
            max_weights=(32, INF),
        )
        self._init_conf = Configuration2(
            build_model("tokens.2.raw.b1|"), lr=0.125, length=32, batch=1, steps=64
        )

    def test_init(self):
        search = Search(
            system="test",
            db=self._db,
            template=self._template,
            init_conf=self._init_conf,
            predictor=None,
            train_time=(1.0, 1.0),
        )
        conf = search.select_conf()
        self.assertEqual(conf, self._init_conf)
