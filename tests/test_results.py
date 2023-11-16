import logging
from unittest import TestCase, main
import numpy as np

from texmo.configuration2 import Configuration2, Template
from texmo.results import ResultSet
from texmo.model3 import build_model
from texmo.run import Run, LossTrendBase


class FakeLossTrend(LossTrendBase):
    version = 0

    def params(self) -> np.ndarray:
        return np.array([1.234])


class ResultSetTest(TestCase):
    def setUp(self):
        self._template = Template(
            spec_regex=None,
            lr=None,
            length=None,
            batch=None,
            steps=None,
            max_weights=None,
        )
        self._results = ResultSet(
            result_db=None, template=self._template, system="test"
        )

    def test_no_runs(self):
        self.assertIsNone(self._results.top_conf(max_time=1))
        self.assertFalse(list(self._results.top_by_neighbors_score(max_time=1)))

    def test_add_run(self):
        conf = Configuration2(
            build_model("tokens.2|"), lr=0.125, length=32, batch=1, steps=64
        )
        run = Run(
            system="test", loss=1.234, train_time=12.345, loss_trend=FakeLossTrend()
        )
        self._results.add_run(conf, run)

        self.assertIsNone(self._results.top_conf(max_time=1))
        top_conf_results = self._results.top_conf(max_time=20)
        self.assertIsNotNone(top_conf_results)
        self.assertEqual(top_conf_results.conf, conf)

        self.assertEqual(top_conf_results.median_score, 1.234)
        self.assertEqual(top_conf_results.neighbors_score, 1.234)
        self.assertEqual(top_conf_results.estimated_time(system="test"), 12.345)

        self.assertFalse(list(self._results.top_by_neighbors_score(max_time=1)))
        neighbors = list(self._results.top_by_neighbors_score(max_time=20))

        self.assertTrue(conf in (n.conf for n in neighbors))
        self.assertGreater(len(neighbors), 10)

        for n in neighbors:
            if n.conf == conf:
                continue
            self.assertIsNone(n.median_score)
            self.assertEqual(n.neighbors_score, 1.234)
            self.assertIsNotNone(n.estimated_time(system="test"))
    
    def test_add_run_other_system(self):
        conf = Configuration2(
            build_model("tokens.2|"), lr=0.125, length=32, batch=1, steps=64
        )
        run = Run(
            system="test", loss=1, train_time=10, loss_trend=FakeLossTrend()
        )
        self._results.add_run(conf, run)
        run = Run(
            system="other", loss=2, train_time=20, loss_trend=FakeLossTrend()
        )
        self._results.add_run(conf, run)

        conf_results = self._results.get_conf_results(conf)

        self.assertEqual(len(conf_results.runs), 2)
        self.assertEqual(conf_results.median_score, 1.5)
        self.assertEqual(conf_results.neighbors_score, 1.5)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertEqual(conf_results.estimated_time(system="other"), 20)

        conf2 = Configuration2(
            build_model("tokens.2|"), lr=0.25, length=32, batch=1, steps=64
        )
        conf_results = self._results.get_conf_results(conf2)
        self.assertEqual(len(conf_results.runs), 0)
        self.assertIsNone(conf_results.median_score)
        self.assertEqual(conf_results.neighbors_score, 1.5)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertEqual(conf_results.median_time(system="test"), 10)

        conf3 = Configuration2(
            build_model("tokens.2|"), lr=0.125, length=32, batch=1, steps=128
        )
        conf_results = self._results.get_conf_results(conf3)
        self.assertEqual(conf_results.estimated_time(system="test"), 20)
        self.assertEqual(conf_results.median_time(system="test"), 20)

        conf4 = Configuration2(
            build_model("tokens.4|"), lr=0.125, length=32, batch=1, steps=64
        )
        conf_results = self._results.get_conf_results(conf4)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertIsNone(conf_results.median_time(system="test"))


if __name__ == "__main__":
    main()
