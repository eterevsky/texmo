import logging
from unittest import TestCase, main

from texmo.configuration2 import Configuration2, Template
from texmo.results import ResultSet
from texmo.run import Run


class ResultSetTest(TestCase):
    def setUp(self):
        self._template = Template(
            spec_regex=None,
            lr=None,
            length=None,
            batch=None,
            steps=None,
            max_weights=None,
        )
        self._results = ResultSet(
            result_db=None, template=self._template, system="test"
        )

    def test_no_runs(self):
        self.assertIsNone(self._results.top_conf(max_time=1))
        self.assertFalse(list(self._results.top_by_neighbors_score(max_time=1)))


if __name__ == "__main__":
    main()
