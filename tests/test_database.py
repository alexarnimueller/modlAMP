import unittest
from os.path import dirname, join

import pytest

from modlamp.database import _read_db_config, query_apd, query_camp


class TestConnect(unittest.TestCase):

    def test_config_read(self):
        conf = _read_db_config(join(dirname(__file__), "../modlamp/data/db_config.json"))
        d = ["host", "password", "user", "database"]
        self.assertEqual(set(conf.keys()), set(d))


@pytest.mark.network
class TestDB(unittest.TestCase):
    # queried inside the test methods, not at class-definition time: as module-level
    # attributes an outage of either website broke collection of the whole test suite

    def test_query_apd(self):
        self.assertEqual(query_apd([15]), ["GLFDIVKKVVGALGSL"])

    def test_query_camp(self):
        self.assertEqual(query_camp([2705]), ["GLFDIVKKVVGALGSL"])
