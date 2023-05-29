import logging
from unittest import TestCase
from datetime import datetime

from texmo.configuration import Configuration, Template, conf_to_string, conf_from_record
from texmo.model2 import build_model
from texmo.search import Search
from texmo.record import TrainingRecord
from texmo.run import Run

logging.disable(level=logging.ERROR)


INIT_CONF = Configuration(
    model=build_model("dense.1.relu"),
    lr=1.0,
    sample_len=128,
    batch=256,
    t=1,
)

TEMPLATE = Template(
    spec_regex="dense.1.relu",
    lr=(1.0, 1.0),
    sample_len=(128, 128),
    batch=(128, 256),
    t=(1, 1),
    max_weights=None,
)

def create_record(spec, loss, batch=256):
    return TrainingRecord(
        timestamp=datetime.fromisoformat("2022-02-02T02:02:02"),
        model_spec=spec,
        weights=1024,
        steps=123,
        train_time_s=1,
        learning_rate=1.0,
        regularization=0.0001,
        train_sample_len=128,
        train_batch=batch,
        total_data=2**34,
        loss=loss,
        test_sample_len=1024,
        test_batch=1024,
        test_poisoned=True,
        init_scale=1.0,
        planned_time_s=1,
        final_time_s=1,
        loss_model_v=0,
        loss_model_params=None,
    )


class SearchTest(TestCase):
    def test_init(self):
        search = Search(
            db=None,
            template=TEMPLATE,
            init_conf=INIT_CONF,
            min_max_weights=1024,
        )
        conf = search.select_conf()
        self.assertEqual(conf, INIT_CONF)

    def test_select_neighbor(self):
        search = Search(
            db=None,
            template=TEMPLATE,
            init_conf=INIT_CONF,
            min_max_weights=1024,
        )
        record = create_record("dense.1.relu", 4.5)
        step_loss = [4.5 + (122 - s) * 0.01 for s in range(0, 123)]
        run = Run(loss=4.5, step_loss=step_loss)
        search.add_run(record, INIT_CONF, run, None)

        i, neighbor, conf = search._select_neighbor(1, 1024)

        self.assertEqual(neighbor, INIT_CONF._replace(batch=128))

    def test_select_by_pred(self):
        search = Search(
            db=None,
            template=TEMPLATE,
            init_conf=INIT_CONF,
            min_max_weights=1024,
        )
        record = create_record("dense.1.relu", 4.5)
        step_loss = [4.5 + (122 - s) * 0.01 for s in range(0, 123)]
        run = Run(loss=4.5, step_loss=step_loss)
        search.add_run(record, INIT_CONF, run, None)

        i, conf = search._select_by_pred_score(1, 1024)

        self.assertEqual(conf, INIT_CONF._replace(batch=128))