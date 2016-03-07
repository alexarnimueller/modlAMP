import unittest

from database.query import _read_db_config

class Testconnect(unittest.TestCase):

	def test_config_read(self):
		conf = _read_db_config(password='1234')
		d = ['host', 'password', 'user', 'database']
		self.assertEqual(set(conf.keys()),set(d))

