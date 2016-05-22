import unittest
from modlamp.sequences import Helices

class TestHelices(unittest.TestCase):
	H = Helices(18, 40, 1)
	H.generate_helices()

	def test_seq_length(self):
		self.assertIn(len(self.H.sequences[0]),range(10,41))

	def test_first_placement(self):
		self.assertTrue(any(s in self.H.sequences[0][:4] for s in ('K','R')))

	def test_basic_AAs(self):
		if 'K' in self.H.sequences[0][:4]:
			p = self.H.sequences[0][:4].index('K')
		elif 'R' in self.H.sequences[0][:4]:
			p = self.H.sequences[0][:4].index('R')

		self.assertTrue(any(a in (self.H.sequences[0][p+3],self.H.sequences[0][p+4],self.H.sequences[0][p+6],self.H.sequences[0][p+7]) for a in ('K','R')))

if __name__ == '__main__':
	unittest.main()