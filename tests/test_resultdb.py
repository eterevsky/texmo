from unittest import main, TestCase

from texmo.configuration import Configuration, Template
from texmo.record import TrainingRecord
from texmo.resultdb import ResultDB
from texmo.spec import ModelSpec


def create_record(spec, loss, batch=256, reg=0.1):
    return TrainingRecord(
        timestamp="2022-02-02T02:02:02",
        model_spec=spec,
        weights=1024,
        steps=123,
        train_time_s=8,
        learning_rate=0.2,
        regularization=reg,
        train_sample_len=128,
        train_batch=batch,
        total_data=2**34,
        loss=loss,
        test_sample_len=1024,
        test_batch=1024,
        test_poisoned=True,
        init_scale=1.0,
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
        self.db.add_record(create_record("attention.16", 123), skip_invalid=True)
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
            create_record("dense.16.relu", 3.321, batch=64, reg=0.001)
        )

        confs = self.db.get_confs_runs(Template(sample_len=128, init_scale=1.0))
        all = set((conf._replace(id=None), loss) for conf, loss in confs)

        spec_tanh = ModelSpec.parse("dense.16.tanh")
        spec_relu = ModelSpec.parse("dense.16.relu")

        conf = Configuration(
            None,
            spec_relu,
            lr=0.2,
            sample_len=128,
            batch=256,
            regularization=0.1,
            init_scale=1.0,
            t=8,
        )

        self.assertEqual(
            all,
            {
                (conf._replace(spec=spec_tanh), 3.123),
                (conf._replace(spec=spec_tanh), 3.321),
                (conf, 3.321),
                (conf._replace(batch=64), 3.321),
                (conf._replace(batch=64, regularization=0.001), 3.321),
            },
        )

        confs = self.db.get_confs_runs(Template(sample_len=128, init_scale=1.0, regularization=0.1))
        subset1 = set((conf._replace(id=None), loss) for conf, loss in confs)

        self.assertEqual(
            subset1,
            {
                (conf._replace(spec=spec_tanh), 3.123),
                (conf._replace(spec=spec_tanh), 3.321),
                (conf, 3.321),
                (conf._replace(batch=64), 3.321),
            },
        )

        confs = self.db.get_confs_runs(Template(sample_len=128, init_scale=1.0, batch=256))
        subset2 = set((conf._replace(id=None), loss) for conf, loss in confs)

        self.assertEqual(
            subset2,
            {
                (conf._replace(spec=spec_tanh), 3.123),
                (conf._replace(spec=spec_tanh), 3.321),
                (conf, 3.321),
            },
        )


if __name__ == "__main__":
    main()
