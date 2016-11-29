import unittest
from os.path import join, dirname
from modlamp.database import query_apd, query_camp
from modlamp.database import _read_db_config


class TestConnect(unittest.TestCase):

    def test_config_read(self):
        conf = _read_db_config(join(dirname(__file__), '../modlamp/data/db_config.json'))
        d = ['host', 'password', 'user', 'database']
        self.assertEqual(set(conf.keys()), set(d))


class TestDB(unittest.TestCase):
    seq1 = query_apd([15])
    seq2 = query_camp([2705])

    def test_query(self):
        self.assertEqual(self.seq1, ['GLFDIVKKVVGALGSL'])
        self.assertEqual(self.seq2, ['GLFDIVKKVVGALGSL'])
