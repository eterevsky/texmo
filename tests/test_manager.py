import logging
from unittest import TestCase

from texmo.configuration2 import Configuration2
from texmo.dataset import build_fake_dataset
from texmo.manager import Manager
from texmo.model3 import build_model

logging.disable(level=logging.ERROR)


class ManagerTest(TestCase):
    def test_train(self):
        dataset = build_fake_dataset()
        conf = Configuration2(
            build_model("bytes-emb.16|gru.32"),
            lr=0.125,
            length=32,
            batch=8,
            steps=64,
        )
        manager = Manager(conf)
        manager.init()

        manager.train(steps=None, time_limit=None, dataset=dataset, quiet=False)

        loss = manager.eval(dataset)
        self.assertLess(loss, 2)

        s = manager.continue_prefix("abac", 2, temperature=0.01)
        self.assertEqual(s, b"abacab")
