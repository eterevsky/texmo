import unittest
from unittest import TestCase

from models import build_from_name


class ModelsTest(TestCase):
    def test_equal(self):
        model = build_from_name('equal')
        self.assertEqual(model.full_name, 'equal')

    def test_freq(self):
        model = build_from_name('freq')
        self.assertEqual(model.full_name, 'freq')
    
    def test_markov(self):
        model = build_from_name('markov-2')
        self.assertEqual(model.full_name, 'markov-2')
        self.assertEqual(model.serialize(), {'name': 'markov', 'suffix': 2})


if __name__ == '__main__':
    unittest.main()