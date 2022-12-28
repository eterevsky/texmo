import logging
from unittest import TestCase, main

from texmo import configuration
from texmo.configuration import Configuration, Template
from texmo.model2 import build_model
from texmo.results import ResultSet
from texmo.run import Run

logging.disable(level=logging.ERROR)


INIT_CONF = Configuration(
    model=build_model("dense.1.relu"),
    lr=0.2,
    sample_len=128,
    batch=256,
    t=1,
)


TEMPLATE = Template(
    spec_regex=r"dense\.\d+\.relu",
    lr=0.2,
    sample_len=128,
    batch=None,
    t=1,
)


class ResultSetTest(TestCase):
    def setUp(self):
        configuration._spec_neighbors = {}

    def test_add_run(self):
        results = ResultSet(result_db=None, template=TEMPLATE)
        results.add_run_conf(INIT_CONF, Run(loss=1.5), update_scores=True)

        conf_id = results.find_conf_id(INIT_CONF)

        cur = results._db.execute(
            "SELECT id, score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(set(map(tuple, cur)), {(conf_id, 1.5)})

    def test_add_run2(self):
        results = ResultSet(
            result_db=None,
            template=TEMPLATE,
        )
        results.add_run_conf(INIT_CONF, Run(loss=1), update_scores=True)
        results.add_run_conf(
            INIT_CONF._replace(model=build_model("dense.2.relu")),
            Run(loss=2),
            update_scores=True,
        )

        cur = results._db.execute(
            "SELECT spec, score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {("dense.1.relu", 1), ("dense.2.relu", 2)},
        )

    def test_add_run3(self):
        results = ResultSet(result_db=None, template=TEMPLATE)
        results.add_run_conf(INIT_CONF, Run(loss=1), update_scores=True)
        results.add_run_conf(INIT_CONF, Run(loss=1), update_scores=True)
        results.add_run_conf(
            INIT_CONF._replace(model=build_model("dense.2.relu")),
            Run(loss=2),
            update_scores=True,
        )
        results.add_run_conf(
            INIT_CONF._replace(model=build_model("dense.2.relu")),
            Run(loss=2),
            update_scores=True,
        )

        cur = results._db.execute(
            "SELECT spec, score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(
            set(map(tuple, cur)),
            {("dense.1.relu", 1), ("dense.2.relu", 2)},
        )

    def test_update_all_scores(self):
        results = ResultSet(result_db=None, template=TEMPLATE)
        results.add_run_conf(INIT_CONF, Run(loss=1), update_scores=False)
        results.add_run_conf(INIT_CONF, Run(loss=1.5), update_scores=False)
        results.add_run_conf(INIT_CONF, Run(loss=2), update_scores=False)
        results.add_run_conf(
            INIT_CONF._replace(batch=512), Run(loss=3), update_scores=False
        )

        results.update_all_scores()

        cur = results._db.execute(
            "SELECT batch, score FROM conf WHERE score IS NOT NULL"
        )
        self.assertEqual(set(map(tuple, cur)), {(256, 1.5), (512, 3)})

    def test_update_all_neighbors(self):
        results = ResultSet(result_db=None, template=TEMPLATE)
        results.add_run_conf(INIT_CONF, Run(loss=1), update_scores=False)
        results.update_all_scores()
        results.update_all_neighbors()

        cur = results._db.execute(
            "SELECT conf2_id FROM conf, neighbor "
            + "WHERE spec = 'dense.4.relu' AND conf.id = conf1_id"
        )
        self.assertIsNone(cur.fetchone())


if __name__ == "__main__":
    main()
