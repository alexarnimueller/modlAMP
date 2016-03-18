import unittest
from modlamp.datasets import load_helicalAMPset, load_AMPvsTMset

class TestHelicalSet(unittest.TestCase):

	data = load_helicalAMPset()

	def test_sequences(self):
		self.assertEqual('NPATLMMFFK', self.data.sequences[3])

	def test_targets(self):
		self.assertEqual([0, 0, 1, 1], [int(self.data.target[i]) for i in [0, 362, 363, 725]])


class TestTMSet(unittest.TestCase):

	data = load_AMPvsTMset()

	def test_sequences(self):
		self.assertEqual('HGSIGAGVDW', self.data.sequences[5])

	def test_targets(self):
		self.assertEqual([0, 0, 1, 1], [int(self.data.target[i]) for i in [0, 205, 206, 411]])