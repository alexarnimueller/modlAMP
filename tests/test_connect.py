import unittest

from delta_db.connect_db import read_db_config

class Testconnect(unittest.TestCase):

	def test_config_read(self):
		conf = read_db_config()
		d = ['host', 'password', 'user', 'database']
		self.assertEqual(set(conf.keys()),set(d))