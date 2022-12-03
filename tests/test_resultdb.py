from datetime import datetime
import numpy as np
from unittest import main, TestCase

from texmo.configuration import Configuration, Template
from texmo.record import TrainingRecord
from texmo.resultdb import ResultDB
from texmo.model2 import build_model


def create_record(spec, loss, batch=256, reg=0.125):
    return TrainingRecord(
        timestamp=datetime.fromisoformat("2022-02-02T02:02:02"),
        model_spec=spec,
        weights=1024,
        steps=123,
        train_time_s=8,
        learning_rate=0.25,
        regularization=reg,
        train_sample_len=128,
        train_batch=batch,
        total_data=2**34,
        loss=loss,
        test_sample_len=1024,
        test_batch=1024,
        test_poisoned=True,
        init_scale=1.0,
        planned_time_s=8,
        final_time_s=8,
        loss_model_v=0,
        loss_model_params=None,
    )


class ResultDBTest(TestCase):
    def setUp(self):
        self.db = ResultDB(":memory:")

    def test_create(self):
        cur = self.db._db.execute("SELECT * FROM conf")
        self.assertIsNone(cur.fetchone())

    def test_add_record_same_spec(self):
        self.db.add_record(create_record("dense.16.relu", 3.123))
        self.db.add_record(create_record("dense.16.relu", 3.321))

        cur = self.db._db.execute("SELECT COUNT(*) FROM conf")
        self.assertEqual(cur.fetchall()[0][0], 1)

        cur = self.db._db.execute("SELECT COUNT(*) FROM run")
        self.assertEqual(cur.fetchall()[0][0], 2)

    def test_add_record_two_specs(self):
        self.db.add_record(create_record("dense.16.relu", 3.123))
        self.db.add_record(create_record("dense.16.tanh", 3.321))

        cur = self.db._db.execute("SELECT COUNT(*) FROM conf")
        self.assertEqual(cur.fetchall()[0][0], 2)

        cur = self.db._db.execute("SELECT COUNT(*) FROM run")
        self.assertEqual(cur.fetchall()[0][0], 2)

    def test_skip_invalid(self):
        self.db.add_record(create_record("gru.16.relu", 123), skip_invalid=True)
        self.db.add_record(
            create_record("dense.16.tanh", 3.321, batch=127), skip_invalid=True
        )

        cur = self.db._db.execute("SELECT COUNT(*) FROM conf")
        self.assertEqual(cur.fetchall()[0][0], 0)

        cur = self.db._db.execute("SELECT COUNT(*) FROM run")
        self.assertEqual(cur.fetchall()[0][0], 0)

    def test_get_confs(self):
        self.db.add_record(create_record("dense.16.tanh", 3.123))
        self.db.add_record(create_record("dense.16.tanh", 3.321))
        self.db.add_record(create_record("dense.16.relu", 3.321))
        self.db.add_record(create_record("dense.16.relu", 3.321, batch=64))
        self.db.add_record(
            create_record("dense.16.relu", 3.321, batch=64, reg=0.03125)
        )

        conf_runs = self.db.get_confs_runs(
            Template(sample_len=128, init_scale=1.0)
        )
        all = set((conf, run.loss) for _, conf, run in conf_runs)

        spec_tanh = build_model("dense.16.tanh")
        spec_relu = build_model("dense.16.relu")

        conf = Configuration(
            spec_relu,
            lr=0.25,
            sample_len=128,
            batch=256,
            regularization=0.125,
            init_scale=1.0,
            t=8,
        )

        self.assertEqual(
            all,
            {
                (conf._replace(model=spec_tanh), 3.123),
                (conf._replace(model=spec_tanh), 3.321),
                (conf, 3.321),
                (conf._replace(batch=64), 3.321),
                (conf._replace(batch=64, regularization=0.03125), 3.321),
            },
        )

        conf_runs = self.db.get_confs_runs(
            Template(sample_len=128, init_scale=1.0, regularization=0.125)
        )
        subset1 = set((conf, run.loss) for _, conf, run in conf_runs)

        self.assertEqual(
            subset1,
            {
                (conf._replace(model=spec_tanh), 3.123),
                (conf._replace(model=spec_tanh), 3.321),
                (conf, 3.321),
                (conf._replace(batch=64), 3.321),
            },
        )

        conf_runs = self.db.get_confs_runs(
            Template(sample_len=128, init_scale=1.0, batch=256)
        )
        subset2 = set((conf, run.loss) for _, conf, run in conf_runs)

        self.assertEqual(
            subset2,
            {
                (conf._replace(model=spec_tanh), 3.123),
                (conf._replace(model=spec_tanh), 3.321),
                (conf, 3.321),
            },
        )

    def test_step_loss(self):
        step_loss = [1, 2, 3]
        self.db.add_record(
            create_record("dense.16.relu", 3.123), step_loss=step_loss
        )

        for i, (conf, loss, step_loss) in enumerate(
            self.db.get_confs_runs(
                Template(sample_len=128), load_step_loss=True
            )
        ):
            self.assertEqual(i, 0)
            self.assertTrue((np.array(step_loss) == step_loss).all())


if __name__ == "__main__":
    main()
