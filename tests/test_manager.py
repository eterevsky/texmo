import logging
from unittest import TestCase

from texmo.configuration import Configuration, Precision
from texmo.dataset import DataSet
from texmo.manager import Manager
from texmo.model3 import build_model

logging.disable(level=logging.ERROR)


def build_fake_dataset():
    s = b"a"
    for c in range(ord("b"), ord("m")):
        s = s + bytes([c]) + s
    return DataSet(data=s)



class ManagerTest(TestCase):
    def test_train(self):
        dataset = build_fake_dataset()
        conf = Configuration(
            build_model("tokens.256.raw.b8-emb.16|gru.32"),
            lr=0.03125,
            length=32,
            batch=8,
            steps=128,
            precision=Precision.FP32,
            decay=1,
        )
        manager = Manager(conf, system="test", dataset=dataset)
        manager.init()

        manager.train(steps=None, time_limit=None, quiet=False)

        loss = manager.eval()
        self.assertLess(loss, 2)

        s = manager.continue_prefix("abac", 2, temperature=0.01)
        self.assertEqual(s, b"abacab")
