import unittest
from ..database import query_apd
from ..database import _read_db_config


class TestConnect(unittest.TestCase):

    def test_config_read(self):
        conf = _read_db_config(password='1234')
        d = ['host', 'password', 'user', 'database']
        self.assertEqual(set(conf.keys()), set(d))


class TestDB(unittest.TestCase):
	
    seq = query_apd([15])
	
    def test_query(self):
        self.assertEqual(self.seq, ['GLFDIVKKVVGALGSL'])
