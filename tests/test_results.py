import os
from unittest import TestCase, main

import numpy as np

from texmo.common import INF
from texmo.configuration2 import Configuration2, Template, conf_neighbors
from texmo.model3 import build_model
from texmo.resultdb import ResultDB
from texmo.results import ResultSet
from texmo.run import LossTrendBase, Run
from texmo.tokens import set_tokens_dir

set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


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
        self._results._check_consistency()

    def test_add_run(self):
        conf = Configuration2(
            build_model("tokens.2.raw.b1|"), lr=0.125, length=32, batch=1, steps=64
        )
        run = Run(
            system="test", loss=1.234, train_time=12.345, loss_trend=FakeLossTrend()
        )
        self._results.add_run(conf, run)
        self._results._check_consistency()

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
            build_model("tokens.2.raw.b1|"), lr=0.125, length=32, batch=1, steps=64
        )
        run = Run(system="test", loss=1, train_time=10, loss_trend=FakeLossTrend())
        self._results.add_run(conf, run)
        run = Run(system="other", loss=2, train_time=20, loss_trend=FakeLossTrend())
        self._results.add_run(conf, run)

        conf_results = self._results.get_conf_results(conf)

        self.assertEqual(len(conf_results.runs), 2)
        self.assertEqual(conf_results.median_score, 1.5)
        self.assertEqual(conf_results.neighbors_score, 1.5)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertEqual(conf_results.estimated_time(system="other"), 20)

        conf2 = Configuration2(
            build_model("tokens.2.raw.b1|"), lr=0.25, length=32, batch=1, steps=64
        )
        conf_results = self._results.get_conf_results(conf2)
        self.assertEqual(len(conf_results.runs), 0)
        self.assertIsNone(conf_results.median_score)
        self.assertEqual(conf_results.neighbors_score, 1.5)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertEqual(conf_results.median_time(system="test"), 10)

        conf3 = Configuration2(
            build_model("tokens.2.raw.b1|"), lr=0.125, length=32, batch=1, steps=128
        )
        conf_results = self._results.get_conf_results(conf3)
        self.assertEqual(conf_results.estimated_time(system="test"), 20)
        self.assertEqual(conf_results.median_time(system="test"), 20)

        conf4 = Configuration2(
            build_model("tokens.4.raw.b2|"), lr=0.125, length=32, batch=1, steps=64
        )
        conf_results = self._results.get_conf_results(conf4)
        self.assertEqual(conf_results.estimated_time(system="test"), 10)
        self.assertIsNone(conf_results.median_time(system="test"))
        self._results._check_consistency()

    def test_load_results_db(self):
        db = ResultDB()
        model = build_model("tokens.2.raw.b1|")
        conf = Configuration2(model=model, lr=0.125, length=32, batch=1, steps=64)
        conf_mod = Configuration2(model=model, lr=0.125, length=32, batch=2, steps=64)
        loss_trend = FakeLossTrend()

        run1 = Run(
            step_loss=None, loss=1, loss_trend=loss_trend, train_time=10, system="sys1"
        )
        db.add_run(conf, run1)

        run2 = Run(
            step_loss=None, loss=2, loss_trend=loss_trend, train_time=20, system="sys2"
        )
        db.add_run(conf, run2)

        run3 = Run(
            step_loss=None, loss=3, loss_trend=loss_trend, train_time=30, system="sys1"
        )
        db.add_run(conf_mod, run3)

        run4 = Run(
            step_loss=None, loss=4, loss_trend=loss_trend, train_time=40, system="sys2"
        )
        db.add_run(conf_mod, run4)

        results = ResultSet(result_db=db, template=self._template, system="sys1")
        results._check_consistency()

        conf_results = results.top_conf(max_time=INF, max_weights=INF)
        self.assertEqual(conf_results.conf, conf)
        self.assertEqual(len(conf_results.runs), 2)
        self.assertEqual(conf_results.median_score, 1.5)
        self.assertEqual(conf_results.neighbors_score, 2)  # [1, 2, 3.5]
        self.assertEqual(conf_results.median_time(system="sys1"), 10)
        self.assertEqual(conf_results.estimated_time(system="sys1"), 10)

        run5 = Run(
            step_loss=None, loss=5, loss_trend=loss_trend, train_time=50, system="sys1"
        )
        results.add_run(conf, run5)
        results._check_consistency()

        conf_results = results.get_conf_results(conf)
        self.assertEqual(len(conf_results.runs), 3)
        self.assertEqual(conf_results.median_score, 2)  # [1, 2, 5]
        self.assertEqual(conf_results.neighbors_score, 2.75)  # [1, 2, 3.5, 5]
        self.assertEqual(conf_results.median_time(system="sys1"), 30)  # 10, 50
        self.assertEqual(conf_results.estimated_time(system="sys1"), 30)

        conf_runs = list(db.get_confs_runs())
        self.assertEqual(len(conf_runs), 5)

    def test_median_time(self):
        db = ResultDB()
        model = build_model("tokens.2.raw.b1|")
        conf1 = Configuration2(model=model, lr=0.125, length=32, batch=1, steps=64)
        conf2 = Configuration2(model=model, lr=0.125, length=32, batch=1, steps=128)
        conf3 = Configuration2(model=model, lr=0.250, length=32, batch=1, steps=64)
        loss_trend = FakeLossTrend()

        self.assertTrue(conf2 in conf_neighbors(conf1, self._template))
        self.assertTrue(conf3 in conf_neighbors(conf1, self._template))

        run1 = Run(
            step_loss=None, loss=1, loss_trend=loss_trend, train_time=10, system="sys1"
        )
        db.add_run(conf1, run1)

        run2 = Run(
            step_loss=None, loss=2, loss_trend=loss_trend, train_time=20, system="sys1"
        )
        db.add_run(conf2, run2)

        run3 = Run(
            step_loss=None, loss=3, loss_trend=loss_trend, train_time=30, system="sys1"
        )
        db.add_run(conf3, run3)


        results = ResultSet(result_db=db, template=self._template, system="sys2")
        results._check_consistency()

        new_run = Run(
            step_loss=None, loss=4, loss_trend=loss_trend, train_time=40, system="sys2"
        )

        results.add_run(conf1, new_run)

        self.assertEqual(results.get_conf_results(conf1).median_time(system="sys2"), 40)
        self.assertEqual(results.get_conf_results(conf2).median_time(system="sys2"), 80)
        self.assertEqual(results.get_conf_results(conf3).median_time(system="sys2"), 40)

    def test_init_neighbors(self):
        db = ResultDB()
        model = build_model("tokens.2.raw.b1|")
        conf1 = Configuration2(model=model, lr=0.125, length=32, batch=1, steps=64)
        run1 = Run(
            step_loss=None, loss=1.5, loss_trend=FakeLossTrend(), train_time=10, system="sys1"
        )
        db.add_run(conf1, run1)

        template = Template('tokens\.2\.raw\.b1\|', lr=0.125, length=(32, 64), batch=(1, INF), steps=64, max_weights=None)

        results = ResultSet(result_db=db, template=template, system="sys1")
        results._check_consistency()

        confs = set(cr.conf for cr in results.top_by_neighbors_score(max_time=16))
        self.assertEqual(confs, {
            conf1,
            Configuration2(model=model, lr=0.125, length=32, batch=2, steps=64),
            Configuration2(model=model, lr=0.125, length=64, batch=1, steps=64),
        })


if __name__ == "__main__":
    main()
