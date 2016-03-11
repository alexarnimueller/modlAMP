import unittest
from modlamp.sequences import Centrosymmetric

class TestCentroSequences(unittest.TestCase):
	S = Centrosymmetric(1)
	S.generate_symmetric()

	def test_block_symmetry(self):
		self.assertEqual(self.S.sequences[0][0],self.S.sequences[0][6])
		self.assertEqual(self.S.sequences[0][1],self.S.sequences[0][5])
		self.assertEqual(self.S.sequences[0][2],self.S.sequences[0][4])

	def test_whole_symmetry(self):
		self.assertEqual(self.S.sequences[0][0:6],self.S.sequences[0][7:13])

	def test_length(self):
		self.assertIn(len(self.S.sequences[0]),(14,21))

if __name__ == '__main__':
	unittest.main()