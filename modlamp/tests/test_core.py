import unittest
from modlamp.sequences import Random


class TestCore(unittest.TestCase):
	sequences = ['GLFDIVKKVVGALG', 'GLFDIVKKVVGALG', 'GLFDIVKKVVGALK', 'ABCDEFGHAJKLMNOPQRSTUVWXYZ']
	R = Random(10, 20, 1)
	R.sequences = sequences

	def test_filter_unnatural(self):
		self.R.filter_unnatural()
		self.assertNotIn('ABCDEFGHAJKLMNOPQRSTUVWXYZ', self.R.sequences)

	def test_mutate(self):
		self.R.mutate_AA(1, 1)
		self.assertNotEqual(self.sequences, self.R.sequences)

	def test_filter_aa(self):
		self.R.filter_aa(['G'])
		self.assertEqual(len(self.R.sequences), 0)

if __name__ == '__main__':
	unittest.main()
