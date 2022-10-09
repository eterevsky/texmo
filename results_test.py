from unittest import main, TestCase

from configuration import Configuration
from resultdb import ResultDB
from results import ResultSet
from spec import ModelSpec


INIT_CONF = Configuration(
    id=None,
    spec=ModelSpec.parse("dense.1.relu"),
    lr=0.2,
    sample_len=128,
    batch=256,
    regularization=0.1,
    init_scale=1.0,
    t=1,
)


class ResultSetTest(TestCase):
    def test_add_run(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1.5)

        conf_id = results.find_conf_id(INIT_CONF)

        cur = results._db.execute(
            "SELECT id, score, cluster_score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(set(map(tuple, cur)), {(conf_id, 1.5, 1.5)})

        cur = results._db.execute(
            "SELECT spec, batch FROM conf, neighbor WHERE conf1_id = ? AND conf2_id = conf.id",
            (conf_id,),
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {
                ("dense.2.relu", 256),
                ("dense.1.relu", 128),
                ("dense.1.relu", 512),
            },
        )

        cur = results._db.execute(
            "SELECT spec, batch FROM conf, neighbor WHERE conf2_id = ? AND conf1_id = conf.id",
            (conf_id,),
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {
                ("dense.2.relu", 256),
                ("dense.1.relu", 128),
                ("dense.1.relu", 512),
            },
        )

        cur = results._db.execute(
            "SELECT spec, batch, cluster_score FROM conf WHERE cluster_score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {
                ("dense.1.relu", 256, 1.5),
                ("dense.2.relu", 256, 1.5),
                ("dense.1.relu", 128, 1.5),
                ("dense.1.relu", 512, 1.5),
            },
        )

    def test_add_run2(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1)
        results.add_run(
            INIT_CONF._replace(spec=ModelSpec.parse("dense.2.relu")), 2
        )

        cur = results._db.execute(
            "SELECT spec, score, cluster_score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {("dense.1.relu", 1, 1.0), ("dense.2.relu", 2, 1.5)},
        )

    def test_add_run3(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1)
        results.add_run(INIT_CONF, 1)
        results.add_run(
            INIT_CONF._replace(spec=ModelSpec.parse("dense.2.relu")), 2
        )
        results.add_run(
            INIT_CONF._replace(spec=ModelSpec.parse("dense.2.relu")), 2
        )

        cur = results._db.execute(
            "SELECT spec, score, cluster_score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {("dense.1.relu", 1, 1), ("dense.2.relu", 2, 2)},
        )

    def test_update_all_scores(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1, False)
        results.add_run(INIT_CONF, 1.5, False)
        results.add_run(INIT_CONF, 2, False)
        results.add_run(INIT_CONF._replace(batch=512), 3, False)

        results.update_all_scores()

        cur = results._db.execute(
            "SELECT batch, score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(set(map(tuple, cur)), {(256, 1.5), (512, 3)})

    def test_update_all_neighbors(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1, False)
        results.update_all_scores()
        results.update_all_neighbors()

        cur = results._db.execute(
            "SELECT conf2_id FROM conf, neighbor " +
            "WHERE spec = 'dense.4.relu' AND conf.id = conf1_id"
        )
        self.assertIsNone(cur.fetchone())

    def test_update_all_cluster_scores(self):
        results = ResultSet(
            result_db=None, init_conf=INIT_CONF, vary=("size", "batch")
        )
        results.add_run(INIT_CONF, 1, False)
        results.add_run(INIT_CONF, 1, False)
        results.add_run(
            INIT_CONF._replace(spec=ModelSpec.parse("dense.2.relu")), 2, False
        )
        results.add_run(
            INIT_CONF._replace(spec=ModelSpec.parse("dense.2.relu")), 2, False
        )

        results.update_all_scores()
        results.update_all_neighbors()
        results.update_all_cluster_scores()

        cur = results._db.execute(
            "SELECT spec, score, cluster_score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {("dense.1.relu", 1, 1), ("dense.2.relu", 2, 2)},
        )


if __name__ == "__main__":
    main()
