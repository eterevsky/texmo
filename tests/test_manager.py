from unittest import TestCase

from texmo.configuration import Configuration
from texmo.dataset import build_fake_dataset
from texmo.manager import Manager
from texmo.model2 import build_model


class ManagerTest(TestCase):
    def test_train(self):
        dataset = build_fake_dataset()
        conf = Configuration(
            build_model("gru.16"),
            lr=0.125,
            sample_len=32,
            batch=8,
            regularization=0.125,
            init_scale=1.0,
            t=1,
        )
        manager = Manager(conf)
        manager.init()

        train_time = manager.train(
            steps=30, time_limit=None, train_set=dataset, quiet=True
        )
        self.assertGreater(train_time, 0.9)
        self.assertLess(train_time, 1.2)

        loss = manager.eval(dataset)
        self.assertLess(loss, 2)

        s = manager.continue_prefix("abac", 2)
        self.assertEqual(len(s), 6)
