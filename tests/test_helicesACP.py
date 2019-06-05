import unittest
from modlamp.sequences import HelicesACP


class TestHelicesACP(unittest.TestCase):

	S = HelicesACP(10, 18, 36)
	S.generate_sequences()

	def test_seq_num(self):
		self.assertEqual(len(self.S.sequences), 10)

	def test_seq_len(self):
		for seq in self.S.sequences:
			self.assertIn(len(seq), range(18, 37))


if __name__ == '__main__':
	unittest.main()
