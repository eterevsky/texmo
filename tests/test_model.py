import unittest
from unittest import TestCase

from texmo.model import Model


class ModelTest(TestCase):
    def test_no_params(self):
        class TestModel(Model):
            name = 'test1'

        m = TestModel()

        self.assertEqual(m.full_name, 'test1')
        self.assertEqual(m.serialize(), {'name': 'test1'})

    def test_one_param(self):
        class TestModel(Model):
            name = 'test2'
            def __init__(self, x):
                super().__init__(x=x)

        m = TestModel(128)

        self.assertEqual(m.x, 128)
        self.assertEqual(m.full_name, 'test2-128')
        self.assertEqual(m.serialize(), {'name': 'test2', 'x': 128})

    def test_two_params(self):
        class TestModel(Model):
            name = 'test3'
            def __init__(self, x, y=256):
                super().__init__(x=x, y=y)

        m = TestModel(128)

        self.assertEqual(m.x, 128)
        self.assertEqual(m.y, 256)
        self.assertEqual(m.full_name, 'test3-128-256')
        self.assertEqual(m.serialize(), {'name': 'test3', 'x': 128, 'y': 256})


if __name__ == '__main__':
    unittest.main()