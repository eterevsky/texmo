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
            t=1,
        )
        manager = Manager(conf)
        manager.init()

        manager.train(steps=30, time_limit=None, train_set=dataset, quiet=True)

        loss = manager.eval(dataset)
        self.assertLess(loss, 2)

        s = manager.continue_prefix("abac", 2)
        self.assertEqual(len(s), 6)
