import logging
import os
from unittest import TestCase, main

import numpy as np

from texmo.configuration import Configuration
from texmo.model3 import build_model
from texmo.predict import build_loss_trend
from texmo.resultdb import ResultDB
from texmo.run import Run
from texmo.tokens import set_tokens_dir

logging.disable(level=logging.ERROR)
set_tokens_dir(os.path.join(os.path.dirname(__file__), "../tokens"))


def create_conf_run(spec="tokens.256.cw.bh-emb.64|gru.64", batch=256, loss=3.123):
    model = build_model(spec)
    conf = Configuration(
        model=model,
        lr=0.25,
        length=128,
        batch=batch,
        steps=256,
        precision='fp32',

        decay=1,
    )
    loss_trend = build_loss_trend(None, 1, [1, 2, 3])
    run = Run(step_loss=None, loss=loss, loss_trend=loss_trend, train_time=12.345, system="test")
    return conf, run


class ResultDBTest(TestCase):
    def setUp(self):
        self.db = ResultDB()

    def test_create(self):
        cur = self.db._db.execute("SELECT * FROM conf")
        self.assertIsNone(cur.fetchone())

    def test_add_record_same_spec(self):
        conf1, run1 = create_conf_run()
        conf2, run2 = create_conf_run(loss=3.321)
        self.db.add_run(conf1, run1, update_neighbors=False)
        self.db.add_run(conf2, run2, update_neighbors=False)

        cur = self.db._db.execute("SELECT COUNT(*) AS count FROM conf")
        self.assertEqual(cur.fetchall()[0][0], 1)

        cur = self.db._db.execute("SELECT COUNT(*) AS count FROM run")
        self.assertEqual(cur.fetchall()[0][0], 2)

    def test_add_record_two_specs(self):
        conf1, run1 = create_conf_run()
        self.db.add_run(conf1, run1, update_neighbors=False)
        conf2, run2 = create_conf_run(spec="tokens.256.cw.bh-emb.64|lstm.64", loss=3.321)
        self.db.add_run(conf2, run2, update_neighbors=False)

        cur = self.db._db.execute("SELECT COUNT(*) AS count FROM conf")
        self.assertEqual(cur.fetchall()[0][0], 2)

        cur = self.db._db.execute("SELECT COUNT(*) AS count FROM run")
        self.assertEqual(cur.fetchall()[0][0], 2)

    def test_get_confs(self):
        conf1, run1 = create_conf_run(loss=1)
        self.db.add_run(conf1, run1)
        conf2, run2 = create_conf_run(loss=2)
        self.db.add_run(conf2, run2)
        conf3, run3 = create_conf_run(spec="tokens.256.cw.bh-emb.64|lstm.64", loss=3)
        self.db.add_run(conf3, run3)
        conf4, run4 = create_conf_run(spec="tokens.256.cw.bh-emb.64|lstm.64", loss=4)
        self.db.add_run(conf4, run4)

        conf_runs = self.db.get_confs_runs()
        all = set((conf, run.loss) for _, conf, run in conf_runs)

        self.assertEqual(
            all,
            {
                (conf1, 1),
                (conf2, 2),
                (conf3, 3),
                (conf4, 4),
            },
        )

    def test_step_loss(self):
        conf1, run1 = create_conf_run(loss=1)
        run1.add_step(1)
        run1.add_step(2)
        run1.add_step(3)
        self.db.add_run(conf1, run1)

        conf_runs = list(self.db.get_confs_runs())
        self.assertEqual(len(conf_runs), 1)

        np.testing.assert_array_equal(conf_runs[0][2].step_loss, [1, 2, 3])


if __name__ == "__main__":
    main()
