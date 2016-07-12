import unittest
from modlamp.sequences import Random

class TestRandom(unittest.TestCase):

    S = Random(10,30,10)
    S.generate_sequences('randnoCM')

    def test_seq_num(self):
        self.assertEqual(len(self.S.sequences),10)

    def test_seq_len(self):
        for seq in self.S.sequences:
            self.assertIn(len(seq),range(10,31))

    def test_noCM(self):
        for s in self.S.sequences:
            self.assertNotIn('C', s)
            self.assertNotIn('M', s)

if __name__ == '__main__':
    unittest.main()